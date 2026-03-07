import torch
from typing import Optional
import logging
from dataclasses import dataclass

import int4_ext


logger = logging.getLogger(__name__)


@dataclass
class QuantMeta:
    scale: torch.Tensor
    last_dim: Optional[int] = None
    dtype: torch.dtype = torch.float16


@dataclass
class KVQuantMeta:
    k: Optional[QuantMeta] = None
    v: Optional[QuantMeta] = None


class QuantizedCompressor:
    def __init__(self, bits: int = 4):
        assert bits in (4, 8)
        self.bits = bits

    @torch.inference_mode()
    def compress(self, kv_cache: torch.Tensor, mode: str = "kv"):
        """
        kv_cache layout: (2, num_layers, block_size, num_kv_heads, head_size)
          - kv_cache[0] is K
          - kv_cache[1] is V

        mode:
          - "kv": compress both K and V
          - "k" : compress only K (V kept as-is)
          - "v" : compress only V (K kept as-is)

        Quantization is done per-layer (num_layers dimension).
        """
        assert mode in ("kv", "k", "v")
        if kv_cache.dim() < 3 or kv_cache.size(0) != 2:
            raise ValueError("kv_cache must have shape (2, num_layers, ..., head_size)")

        keys = kv_cache[0]
        values = kv_cache[1]
        ck, cv = keys, values
        meta = KVQuantMeta()

        if mode in ("kv", "k"):
            ck, meta_k = self._compress_tensor(keys, keys.dtype)
            meta.k = meta_k

        if mode in ("kv", "v"):
            cv, meta_v = self._compress_tensor(values, values.dtype)
            meta.v = meta_v

        return ck, cv, meta

    def _compress_tensor(self, x: torch.Tensor, orig_dtype: torch.dtype):
        # int8
        if self.bits == 8:
            q, scale = self._quantize_int8(x)
            return q, QuantMeta(scale=scale, dtype=orig_dtype)

        # int4
        q4, scale = self._quantize_int4(x)
        orig_D = x.shape[-1]
        packed = int4_ext.pack_int4(q4.contiguous())
        return packed, QuantMeta(scale=scale, last_dim=orig_D, dtype=orig_dtype)

    def _layerwise_abs_max(self, tensor: torch.Tensor):
        # tensor shape: (num_layers, ...)
        if tensor.dim() < 1:
            raise ValueError("tensor must have at least 1 dimension for layer-wise quantization")
        if tensor.dim() == 1:
            return tensor.abs()

        reduce_dims = tuple(range(1, tensor.dim()))
        return tensor.abs().amax(dim=reduce_dims, keepdim=True)

    def _quantize_int8(self, tensor: torch.Tensor):
        t = tensor.float()

        # Symmetric per-layer quantization
        abs_max = self._layerwise_abs_max(t)
        scale = abs_max / 127.0
        scale = torch.clamp(scale, min=1e-6)

        q = torch.round(t / scale).clamp(-127, 127).to(torch.int8)
        return q, scale.to(torch.float32)

    def _quantize_int4(self, tensor: torch.Tensor):
        t = tensor.float()

        # Symmetric per-layer quantization
        abs_max = self._layerwise_abs_max(t)
        scale = abs_max / 7.0
        scale = torch.clamp(scale, min=1e-6)

        q = torch.round(t / scale).clamp(-8, 7).to(torch.int8)
        return q, scale.to(torch.float32)

    @torch.inference_mode()
    def decompress(self, kv_cache: torch.Tensor, meta: KVQuantMeta):
        """
        Decompress a KV cache block and restore layout:
        (2, num_layers, block_size, num_kv_heads, head_size)
        """
        if kv_cache.dim() < 3 or kv_cache.size(0) != 2:
            raise ValueError("kv_cache must have shape (2, num_layers, ..., head_size/packed_head_size)")
        if not isinstance(meta, KVQuantMeta):
            raise TypeError("meta must be KVQuantMeta")

        keys = kv_cache[0]
        values = kv_cache[1]
        dk, dv = keys, values

        if meta.k is not None:
            dk = self._decompress_tensor(keys, meta.k)
        if meta.v is not None:
            dv = self._decompress_tensor(values, meta.v)

        return torch.stack([dk, dv], dim=0)


    @torch.inference_mode()
    def batch_decompress(self, kv_caches: torch.Tensor, metas):
        """
        Decompress batched KV cache blocks and restore layout:
        (num_blocks, 2, num_layers, block_size, num_kv_heads, head_size)

        Args:
            kv_caches: Tensor shaped (num_blocks, 2, num_layers, ..., head_size/packed_head_size)
            metas: sequence of KVQuantMeta with length == num_blocks,
                   or a single KVQuantMeta to be reused for all blocks.
        """
        if kv_caches.dim() < 4 or kv_caches.size(1) != 2:
            raise ValueError(
                "kv_caches must have shape (num_blocks, 2, num_layers, ..., head_size/packed_head_size)"
            )

        num_blocks = kv_caches.size(0)

        if isinstance(metas, KVQuantMeta):
            metas = [metas] * num_blocks

        if len(metas) != num_blocks:
            raise ValueError(f"len(metas)={len(metas)} must equal num_blocks={num_blocks}")
        if any(not isinstance(m, KVQuantMeta) for m in metas):
            raise TypeError("each meta in metas must be KVQuantMeta")

        restored_blocks = [
            self.decompress(kv_caches[i], metas[i]) for i in range(num_blocks)
        ]
        return torch.stack(restored_blocks, dim=0)

    def _decompress_tensor(self, x, meta: QuantMeta):
        scale = meta.scale.to(x.device, torch.float32)

        if self.bits == 8:
            out = x.float() * scale
            return out.to(meta.dtype)

        if meta.last_dim is None:
            raise ValueError("INT4 decompress requires last_dim in meta.")
        i4 = int4_ext.unpack_int4(x.contiguous(), meta.last_dim).float()
        out = i4 * scale
        return out.to(meta.dtype)
