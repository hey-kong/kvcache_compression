from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class HuffmanCodebook:
    encode_table: Dict[int, str]   # symbol -> bitstring
    decode_table: Dict[str, int]   # bitstring -> symbol


@dataclass
class CompressedChunk:
    """
    Compress one tensor chunk losslessly.

    For each BF16 element:
      - exponent (8 bits): Huffman compressed
      - sign + mantissa (1 + 7 = 8 bits): stored as one raw byte

    Fields:
      non_exp_bytes:
          raw bytes for [sign|mantissa], one uint8 per BF16 value
      exp_bitstream:
          Huffman-compressed exponent bitstream packed into bytes
      exp_num_symbols:
          number of exponent symbols
      exp_num_valid_bits:
          valid bits inside exp_bitstream
      codebook:
          Huffman codebook for this chunk
      orig_shape:
          original tensor shape
    """
    non_exp_bytes: bytes
    exp_bitstream: bytes
    exp_num_symbols: int
    exp_num_valid_bits: int
    codebook: HuffmanCodebook
    orig_shape: Tuple[int, ...]


class _HuffNode:
    __slots__ = ("freq", "symbol", "left", "right")

    def __init__(
        self,
        freq: int,
        symbol: Optional[int] = None,
        left: Optional["_HuffNode"] = None,
        right: Optional["_HuffNode"] = None,
    ):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right


class HuffmanCompressor:
    """
    Lossless compressor for BF16 KV cache.

    Input layout expected by compress():
        kv_cache: (2, num_layers, block_size, num_kv_heads, head_size)
          - kv_cache[0] is K
          - kv_cache[1] is V

    Compression policy:
      - Split BF16 into:
            sign (1 bit)
            exponent (8 bits)
            mantissa (7 bits)
      - Store [sign | mantissa] as one uint8 per element
      - Huffman-compress exponent stream
      - Compress layer-wise for K and V separately

    decompress() reconstructs a single CompressedChunk back to BF16 tensor.
    """

    def __init__(self):
        pass

    # =========================================================
    # Public API
    # =========================================================

    @torch.inference_mode()
    def compress(self, kv_cache: torch.Tensor) -> List[List[CompressedChunk]]:
        """
        Compress the full kv_cache layer-wise.

        Args:
            kv_cache:
                BF16 tensor with shape
                (2, num_layers, block_size, num_kv_heads, head_size)

        Returns:
            chunks:
                Nested list:
                    chunks[kv_idx][layer_idx] -> CompressedChunk
                where kv_idx=0 means K, kv_idx=1 means V
        """
        self._validate_kv_cache(kv_cache)

        _, num_layers, _, _, _ = kv_cache.shape
        out: List[List[CompressedChunk]] = [[], []]

        for kv_idx in range(2):  # 0: K, 1: V
            for layer_idx in range(num_layers):
                x = kv_cache[kv_idx, layer_idx]  # (block_size, num_kv_heads, head_size)
                chunk = self._compress_chunk(x)
                out[kv_idx].append(chunk)

        return out

    @torch.inference_mode()
    def decompress(
        self,
        chunk: CompressedChunk,
        device: Optional[torch.device | str] = None,
    ) -> torch.Tensor:
        """
        Decompress one CompressedChunk back to a BF16 tensor.

        Args:
            chunk:
                One compressed chunk, typically one (K/V, layer) slice.
            device:
                Target device for the restored tensor.
                If None, tensor is returned on CPU.

        Returns:
            Restored BF16 tensor with shape = chunk.orig_shape
        """
        non_exp = np.frombuffer(chunk.non_exp_bytes, dtype=np.uint8).copy()
        exp = self._huffman_decode_symbols(
            bitstream=chunk.exp_bitstream,
            num_valid_bits=chunk.exp_num_valid_bits,
            codebook=chunk.codebook,
            expected_num_symbols=chunk.exp_num_symbols,
        )

        if non_exp.size != exp.size:
            raise ValueError(
                f"Size mismatch during decompression: "
                f"non_exp.size={non_exp.size}, exp.size={exp.size}"
            )

        bits = self._merge_bf16_bits(non_exp, exp)
        return self._uint16_numpy_to_bf16_tensor(bits, chunk.orig_shape, device=device)

    # =========================================================
    # Optional helper APIs
    # =========================================================

    @torch.inference_mode()
    def reconstruct_kv_cache(
        self,
        chunks: List[List[CompressedChunk]],
        device: Optional[torch.device | str] = None,
    ) -> torch.Tensor:
        """
        Reconstruct the full kv_cache from nested chunks.

        Args:
            chunks:
                Output of self.compress(kv_cache)
            device:
                Target device

        Returns:
            kv_cache tensor with shape
            (2, num_layers, block_size, num_kv_heads, head_size)
        """
        if len(chunks) != 2:
            raise ValueError(f"Expected chunks length 2 for K/V, got {len(chunks)}")
        if len(chunks[0]) == 0:
            raise ValueError("Empty chunks[0]")

        num_layers = len(chunks[0])
        if len(chunks[1]) != num_layers:
            raise ValueError(
                f"K/V layer count mismatch: len(chunks[0])={len(chunks[0])}, "
                f"len(chunks[1])={len(chunks[1])}"
            )

        sample_shape = chunks[0][0].orig_shape
        if len(sample_shape) != 3:
            raise ValueError(f"Expected per-chunk shape rank 3, got {sample_shape}")

        block_size, num_kv_heads, head_size = sample_shape
        out = torch.empty(
            (2, num_layers, block_size, num_kv_heads, head_size),
            dtype=torch.bfloat16,
            device=device if device is not None else "cpu",
        )

        for kv_idx in range(2):
            for layer_idx in range(num_layers):
                restored = self.decompress(chunks[kv_idx][layer_idx], device=device)
                out[kv_idx, layer_idx] = restored

        return out

    def compressed_size_bytes(self, chunks: List[List[CompressedChunk]]) -> int:
        """
        Approximate compressed payload size in bytes.
        This excludes Python object overhead.
        """
        total = 0
        for kv_chunks in chunks:
            for c in kv_chunks:
                total += len(c.non_exp_bytes)
                total += len(c.exp_bitstream)

                # rough estimate for codebook payload
                for _, code in c.codebook.encode_table.items():
                    total += 1
                    total += 2
                    total += (len(code) + 7) // 8
        return total

    def original_size_bytes(self, kv_cache: torch.Tensor) -> int:
        return kv_cache.numel() * kv_cache.element_size()

    # =========================================================
    # Internal compression logic
    # =========================================================

    def _validate_kv_cache(self, kv_cache: torch.Tensor):
        if kv_cache.dtype != torch.bfloat16:
            raise TypeError(
                f"HuffmanCompressor only supports torch.bfloat16, "
                f"but got {kv_cache.dtype}"
            )
        if kv_cache.ndim != 5:
            raise ValueError(
                f"Expected kv_cache.ndim == 5, got shape={tuple(kv_cache.shape)}"
            )
        if kv_cache.shape[0] != 2:
            raise ValueError(
                f"Expected kv_cache.shape[0] == 2 for (K,V), got {kv_cache.shape[0]}"
            )

    def _compress_chunk(self, x: torch.Tensor) -> CompressedChunk:
        if x.dtype != torch.bfloat16:
            raise TypeError(f"_compress_chunk expects BF16 tensor, got {x.dtype}")

        orig_shape = tuple(x.shape)
        bits = self._bf16_tensor_to_uint16_numpy(x)
        non_exp, exp = self._split_bf16_bits(bits)

        codebook = self._build_huffman_codebook(exp)
        exp_bitstream, exp_num_valid_bits = self._huffman_encode_symbols(exp, codebook)

        return CompressedChunk(
            non_exp_bytes=non_exp.tobytes(),
            exp_bitstream=exp_bitstream,
            exp_num_symbols=exp.size,
            exp_num_valid_bits=exp_num_valid_bits,
            codebook=codebook,
            orig_shape=orig_shape,
        )

    # =========================================================
    # BF16 bit ops
    # =========================================================

    def _bf16_tensor_to_uint16_numpy(self, x: torch.Tensor) -> np.ndarray:
        """
        Convert a BF16 tensor to flat np.uint16 raw BF16 bit patterns.
        """
        bits_t = x.detach().contiguous().view(torch.uint16).cpu()
        bits_np = bits_t.numpy().reshape(-1).astype(np.uint16, copy=False)
        return bits_np

    def _uint16_numpy_to_bf16_tensor(
        self,
        bits: np.ndarray,
        shape: Tuple[int, ...],
        device: Optional[torch.device | str] = None,
    ) -> torch.Tensor:
        bits = np.asarray(bits, dtype=np.uint16).reshape(-1)
        t = torch.from_numpy(bits.copy()).view(torch.bfloat16).reshape(shape)
        if device is not None:
            t = t.to(device)
        return t

    def _split_bf16_bits(self, bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        BF16 layout:
          bit 15      : sign
          bits 14..7  : exponent (8 bits)
          bits 6..0   : mantissa (7 bits)

        Returns:
          non_exp: uint8, storing [sign|mantissa]
          exp:     uint8, storing exponent
        """
        bits = bits.astype(np.uint16, copy=False)

        sign = (bits >> 15).astype(np.uint8)          # 1 bit
        exp = ((bits >> 7) & 0xFF).astype(np.uint8)   # 8 bits
        mant = (bits & 0x7F).astype(np.uint8)         # 7 bits

        non_exp = ((sign << 7) | mant).astype(np.uint8)
        return non_exp, exp

    def _merge_bf16_bits(self, non_exp: np.ndarray, exp: np.ndarray) -> np.ndarray:
        non_exp = np.asarray(non_exp, dtype=np.uint8)
        exp = np.asarray(exp, dtype=np.uint8)

        sign = (non_exp >> 7).astype(np.uint16)
        mant = (non_exp & 0x7F).astype(np.uint16)
        exp16 = exp.astype(np.uint16)

        bits = ((sign << 15) | (exp16 << 7) | mant).astype(np.uint16)
        return bits

    # =========================================================
    # Huffman coding
    # =========================================================

    def _build_huffman_codebook(self, symbols: np.ndarray) -> HuffmanCodebook:
        """
        symbols: flattened exponent stream, dtype uint8
        """
        freq = np.bincount(symbols, minlength=256)
        items = [(int(f), sym) for sym, f in enumerate(freq) if f > 0]

        if len(items) == 1:
            sym = items[0][1]
            encode = {sym: "0"}
            decode = {"0": sym}
            return HuffmanCodebook(encode_table=encode, decode_table=decode)

        heap = []
        counter = 0
        for f, sym in items:
            heapq.heappush(heap, (f, counter, _HuffNode(freq=f, symbol=sym)))
            counter += 1

        while len(heap) > 1:
            f1, _, n1 = heapq.heappop(heap)
            f2, _, n2 = heapq.heappop(heap)
            parent = _HuffNode(freq=f1 + f2, left=n1, right=n2)
            heapq.heappush(heap, (f1 + f2, counter, parent))
            counter += 1

        root = heap[0][2]
        encode: Dict[int, str] = {}

        def dfs(node: _HuffNode, prefix: str):
            if node.symbol is not None:
                encode[node.symbol] = prefix if prefix else "0"
                return
            dfs(node.left, prefix + "0")
            dfs(node.right, prefix + "1")

        dfs(root, "")
        decode = {v: k for k, v in encode.items()}
        return HuffmanCodebook(encode_table=encode, decode_table=decode)

    def _huffman_encode_symbols(
        self,
        symbols: np.ndarray,
        codebook: HuffmanCodebook,
    ) -> Tuple[bytes, int]:
        bitstring = "".join(codebook.encode_table[int(s)] for s in symbols.tolist())
        return self._pack_bitstring_to_bytes(bitstring)

    def _huffman_decode_symbols(
        self,
        bitstream: bytes,
        num_valid_bits: int,
        codebook: HuffmanCodebook,
        expected_num_symbols: int,
    ) -> np.ndarray:
        bitstring = self._unpack_bytes_to_bitstring(bitstream, num_valid_bits)

        out = []
        cur = ""
        decode_table = codebook.decode_table

        for ch in bitstring:
            cur += ch
            if cur in decode_table:
                out.append(decode_table[cur])
                cur = ""
                if len(out) == expected_num_symbols:
                    break

        if len(out) != expected_num_symbols:
            raise ValueError(
                f"Huffman decode failed: expected {expected_num_symbols} symbols, "
                f"got {len(out)}"
            )

        return np.asarray(out, dtype=np.uint8)

    def _pack_bitstring_to_bytes(self, bitstring: str) -> Tuple[bytes, int]:
        num_valid_bits = len(bitstring)
        if num_valid_bits == 0:
            return b"", 0

        pad = (-num_valid_bits) % 8
        if pad:
            bitstring += "0" * pad

        out = bytearray()
        for i in range(0, len(bitstring), 8):
            out.append(int(bitstring[i:i + 8], 2))
        return bytes(out), num_valid_bits

    def _unpack_bytes_to_bitstring(self, data: bytes, num_valid_bits: int) -> str:
        if len(data) == 0:
            return ""
        bitstring = "".join(f"{b:08b}" for b in data)
        return bitstring[:num_valid_bits]
