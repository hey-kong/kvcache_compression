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

    @torch.inference_mode()
    def compress_layer_batched(
            self,
            k_layer: torch.Tensor,
            v_layer: torch.Tensor,
            mode: str = "kv",
            seq_dim: int = 2,
            chunk_size: int = 16,
    ):
        """
        Batched chunk-wise compression for one layer.

        Args:
            k_layer, v_layer: tensors of shape like (B, H, T, D) but seq_dim may vary.
            mode:
              - "kv": compress both K and V
              - "k" : compress only K (V passthrough)
              - "v" : compress only V (K passthrough)
            seq_dim: which dim is sequence length T
            chunk_size: chunk length along T

        Returns:
            ck_layer, cv_layer, metas
              - metas: List[KVQuantMeta], length = ceil(T/chunk_size)
              - if not compressed, corresponding meta field is None and tensor passthrough
        """
        assert mode in ("kv", "k", "v")
        device = k_layer.device
        T = k_layer.shape[seq_dim]

        def move_seq_to_2(x):
            if seq_dim == 2:
                return x
            perm = list(range(x.ndim))
            perm[2], perm[seq_dim] = perm[seq_dim], perm[2]
            return x.permute(perm).contiguous()

        def move_back_from_2(x, ref):
            if seq_dim == 2:
                return x
            perm = list(range(ref.ndim))
            perm[2], perm[seq_dim] = perm[seq_dim], perm[2]
            inv = [0] * len(perm)
            for i, p in enumerate(perm):
                inv[p] = i
            return x.permute(inv).contiguous()

        k = move_seq_to_2(k_layer)  # (B, H, T, D)
        v = move_seq_to_2(v_layer)  # (B, H, T, D)
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError(f"Expect 4D (B,H,T,D) after permute, got k={k.shape}, v={v.shape}")

        B, H, T2, Dk = k.shape
        _, _, Tv, Dv = v.shape
        assert T2 == T == Tv, f"T mismatch: k={T2} v={Tv} expected={T}"
        orig_dtype_k = k_layer.dtype
        orig_dtype_v = v_layer.dtype

        n_chunks = (T + chunk_size - 1) // chunk_size
        metas = []

        # Determine whether to compress K/V
        do_k = (mode in ("kv", "k"))
        do_v = (mode in ("kv", "v"))

        # Allocate outputs (we may need packed last-dim for INT4, so delay allocation for int4)
        ck_out = None
        cv_out = None

        if self.bits == 8:
            # int8 keeps same shape
            ck_out = k if not do_k else torch.empty_like(k, dtype=torch.int8, device=device)
            cv_out = v if not do_v else torch.empty_like(v, dtype=torch.int8, device=device)
        else:
            # int4: if passthrough, keep original; if compressed, will be uint8 packed with smaller last dim
            ck_out = k if not do_k else None
            cv_out = v if not do_v else None

        for ci in range(n_chunks):
            s = ci * chunk_size
            e = min(T, (ci + 1) * chunk_size)

            meta = KVQuantMeta()

            # --------
            # K chunk
            # --------
            if do_k:
                k_chunk = k[:, :, s:e, :]  # (B,H,tc,Dk)
                t = k_chunk.float()

                abs_max = t.abs().amax()  # scalar
                if self.bits == 8:
                    scale = torch.clamp(abs_max / 127.0, min=1e-6).to(torch.float32)
                    q = torch.round(t / scale).clamp(-127, 127).to(torch.int8)

                    ck_out[:, :, s:e, :] = q
                    meta.k = QuantMeta(scale=scale, last_dim=None, dtype=orig_dtype_k)
                else:
                    scale = torch.clamp(abs_max / 7.0, min=1e-6).to(torch.float32)
                    q4 = torch.round(t / scale).clamp(-8, 7).to(torch.int8)

                    packed = int4_ext.pack_int4(q4.contiguous(), Dk)  # (B,H,tc,packed_D)
                    if ck_out is None:
                        packed_D = packed.shape[-1]
                        ck_out = torch.empty((B, H, T, packed_D), device=device, dtype=torch.uint8)
                    ck_out[:, :, s:e, :] = packed
                    meta.k = QuantMeta(scale=scale, last_dim=Dk, dtype=orig_dtype_k)
            else:
                meta.k = None

            # --------
            # V chunk
            # --------
            if do_v:
                v_chunk = v[:, :, s:e, :]  # (B,H,tc,Dv)
                t = v_chunk.float()

                abs_max = t.abs().amax()
                if self.bits == 8:
                    scale = torch.clamp(abs_max / 127.0, min=1e-6).to(torch.float32)
                    q = torch.round(t / scale).clamp(-127, 127).to(torch.int8)

                    cv_out[:, :, s:e, :] = q
                    meta.v = QuantMeta(scale=scale, last_dim=None, dtype=orig_dtype_v)
                else:
                    scale = torch.clamp(abs_max / 7.0, min=1e-6).to(torch.float32)
                    q4 = torch.round(t / scale).clamp(-8, 7).to(torch.int8)

                    packed = int4_ext.pack_int4(q4.contiguous(), Dv)
                    if cv_out is None:
                        packed_D = packed.shape[-1]
                        cv_out = torch.empty((B, H, T, packed_D), device=device, dtype=torch.uint8)
                    cv_out[:, :, s:e, :] = packed
                    meta.v = QuantMeta(scale=scale, last_dim=Dv, dtype=orig_dtype_v)
            else:
                meta.v = None

            metas.append(meta)

        # If passthrough, ensure dtype consistent (no copy)
        if not do_k:
            ck_out = k.to(orig_dtype_k, copy=False)
        if not do_v:
            cv_out = v.to(orig_dtype_v, copy=False)

        ck_out = move_back_from_2(ck_out, k_layer)
        cv_out = move_back_from_2(cv_out, v_layer)
        return ck_out, cv_out, metas

    @torch.inference_mode()
    def decompress_layer_batched(
            self,
            ck_layer: torch.Tensor,
            cv_layer: torch.Tensor,
            metas,
            seq_dim: int,
            chunk_size: int,
    ):
        device = ck_layer.device
        T = ck_layer.shape[seq_dim]

        def move_seq_to_2(x):
            if seq_dim == 2:
                return x
            perm = list(range(x.ndim))
            perm[2], perm[seq_dim] = perm[seq_dim], perm[2]
            return x.permute(perm).contiguous()

        def move_back_from_2(x, ref):
            if seq_dim == 2:
                return x
            perm = list(range(ref.ndim))
            perm[2], perm[seq_dim] = perm[seq_dim], perm[2]
            inv = [0] * len(perm)
            for i, p in enumerate(perm):
                inv[p] = i
            return x.permute(inv).contiguous()

        ck = move_seq_to_2(ck_layer)
        cv = move_seq_to_2(cv_layer)

        B, H, T2, _ = ck.shape
        assert T2 == T

        n_chunks = (T + chunk_size - 1) // chunk_size
        assert len(metas) == n_chunks, f"metas={len(metas)} n_chunks={n_chunks} T={T} chunk={chunk_size}"

        meta_k0 = metas[0].k
        meta_v0 = metas[0].v

        # output dtype: if not compressed, keep original dtype
        k_dtype = meta_k0.dtype if meta_k0 is not None else ck_layer.dtype
        v_dtype = meta_v0.dtype if meta_v0 is not None else cv_layer.dtype

        # build per-token scales only when compressed
        k_scale_tokens = None
        if meta_k0 is not None:
            k_scales = torch.stack([m.k.scale for m in metas], dim=0).to(device=device, dtype=torch.float32)
            k_scale_tokens = k_scales.repeat_interleave(chunk_size)[:T].view(1, 1, T, 1)

        v_scale_tokens = None
        if meta_v0 is not None:
            v_scales = torch.stack([m.v.scale for m in metas], dim=0).to(device=device, dtype=torch.float32)
            v_scale_tokens = v_scales.repeat_interleave(chunk_size)[:T].view(1, 1, T, 1)

        # -----------------
        # decompress / passthrough K
        # -----------------
        if meta_k0 is None:
            # not compressed
            dk = ck.to(k_dtype, copy=False)
        else:
            if self.bits == 8:
                # ck should be int8
                if ck.dtype != torch.int8:
                    raise TypeError(f"Expected int8 for INT8 K, got {ck.dtype}")
                dk = ck.float()
                dk = dk * k_scale_tokens  # must exist when meta_k0 is not None
                dk = dk.to(k_dtype)
            else:
                # INT4 packed => uint8
                if ck.dtype != torch.uint8:
                    raise TypeError(f"Expected uint8 packed for INT4 K, got {ck.dtype}")
                orig_D = int(meta_k0.last_dim)  # use meta, NOT D2*2
                i4k = int4_ext.unpack_int4(ck.contiguous(), orig_D).float()
                i4k = i4k * k_scale_tokens
                dk = i4k.to(k_dtype)

        # -----------------
        # decompress / passthrough V
        # -----------------
        if meta_v0 is None:
            dv = cv.to(v_dtype, copy=False)
        else:
            if self.bits == 8:
                if cv.dtype != torch.int8:
                    raise TypeError(f"Expected int8 for INT8 V, got {cv.dtype}")
                dv = cv.float()
                dv = dv * v_scale_tokens
                dv = dv.to(v_dtype)
            else:
                if cv.dtype != torch.uint8:
                    raise TypeError(f"Expected uint8 packed for INT4 V, got {cv.dtype}")
                orig_D = int(meta_v0.last_dim)
                i4v = int4_ext.unpack_int4(cv.contiguous(), orig_D).float()
                i4v = i4v * v_scale_tokens
                dv = i4v.to(v_dtype)

        dk = move_back_from_2(dk, ck_layer)
        dv = move_back_from_2(dv, cv_layer)
        return dk, dv
