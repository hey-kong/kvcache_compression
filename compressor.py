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

    def compress(self, keys: torch.Tensor, values: torch.Tensor, mode: str = "kv"):
        """
        mode:
          - "kv": compress both K and V
          - "k" : compress only K (V kept as-is)
          - "v" : compress only V (K kept as-is)
        """
        assert mode in ("kv", "k", "v")
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
        packed = int4_ext.pack_int4(q4.contiguous(), orig_D)
        return packed, QuantMeta(scale=scale, last_dim=orig_D, dtype=orig_dtype)

    def _quantize_int8(self, tensor: torch.Tensor):
        t = tensor.float()

        # Symmetric quantization
        abs_max = t.abs().amax()
        scale = abs_max / 127.0
        scale = torch.clamp(scale, min=1e-6)

        q = torch.round(t / scale).clamp(-127, 127).to(torch.int8)
        return q, scale.to(torch.float32)

    def _quantize_int4(self, tensor: torch.Tensor):
        t = tensor.float()

        # Symmetric quantization
        abs_max = t.abs().amax()
        scale = abs_max / 7.0
        scale = torch.clamp(scale, min=1e-6)

        q = torch.round(t / scale).clamp(-8, 7).to(torch.int8)
        return q, scale.to(torch.float32)

    def decompress(self, keys, values, meta: KVQuantMeta):
        dk, dv = keys, values

        if meta.k is not None:
            dk = self._decompress_tensor(keys, meta.k)
        if meta.v is not None:
            dv = self._decompress_tensor(values, meta.v)

        return dk, dv

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
