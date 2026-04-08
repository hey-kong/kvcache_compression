from __future__ import annotations

import heapq
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from ._C import (
        decompress_layer_parallel_cpp,
        huffman_decode_symbols_cpp,
        huffman_encode_symbols_cpp,
    )

    _HAS_HUFFMAN_EXT = True
except ImportError:
    huffman_encode_symbols_cpp = None
    huffman_decode_symbols_cpp = None
    decompress_layer_parallel_cpp = None
    _HAS_HUFFMAN_EXT = False


@dataclass
class HuffmanCodebook:
    encode_table: Dict[int, str]
    decode_table: Dict[str, int]


@dataclass
class CompressedChunk:
    non_exp_bytes: bytes
    exp_bitstream: bytes
    exp_num_symbols: int
    exp_num_valid_bits: int
    codebook: HuffmanCodebook
    orig_shape: Tuple[int, ...]


@dataclass
class LayerCompressionStats:
    layer_idx: int
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    compression_time_ms: float
    decompression_time_ms: Optional[float] = None


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
    """CPU-only lossless compressor for BF16 KV cache."""

    def __init__(self):
        self._encode_cache: Dict[int, Dict[int, Tuple[int, int]]] = {}
        self._decode_trie_cache: Dict[int, Tuple[List[int], List[int], List[int]]] = {}

    @staticmethod
    def _resolve_num_workers(num_workers: Optional[int]) -> int:
        if num_workers is None:
            return max(1, min(32, (os.cpu_count() or 1)))
        if num_workers <= 0:
            raise ValueError(f"num_workers must be > 0, got {num_workers}")
        return num_workers

    @staticmethod
    def _assert_cpu_device(device: Optional[torch.device | str]) -> None:
        if device is None:
            return
        d = torch.device(device)
        if d.type != "cpu":
            raise ValueError(
                f"HuffmanCompressor is CPU-only; expected CPU device, got {device}"
            )

    @staticmethod
    def _to_cpu_bf16(x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            raise TypeError(f"HuffmanCompressor only supports torch.bfloat16, got {x.dtype}")
        return x.detach().to(device="cpu", copy=True).contiguous()

    @torch.inference_mode()
    def compress(self, kv_cache: torch.Tensor) -> List[List[CompressedChunk]]:
        self._validate_kv_cache(kv_cache)
        kv_cache_cpu = self._to_cpu_bf16(kv_cache)

        _, num_layers, _, _, _ = kv_cache_cpu.shape
        compressed_kv_cache: List[List[CompressedChunk]] = [[], []]
        for kv_idx in range(2):
            for layer_idx in range(num_layers):
                compressed_kv_cache[kv_idx].append(self._compress_chunk(kv_cache_cpu[kv_idx, layer_idx]))
        return compressed_kv_cache

    @torch.inference_mode()
    def compress_with_stats(
        self,
        kv_cache: torch.Tensor,
        print_stats: bool = True,
    ) -> Tuple[List[List[CompressedChunk]], List[LayerCompressionStats]]:
        """Compress KV cache and return per-layer compression stats.

        Compression ratio is defined as: compressed_size / original_size.
        """
        self._validate_kv_cache(kv_cache)
        kv_cache_cpu = self._to_cpu_bf16(kv_cache)

        _, num_layers, _, _, _ = kv_cache_cpu.shape
        compressed_kv_cache: List[List[CompressedChunk]] = [[], []]
        layer_stats: List[LayerCompressionStats] = []

        for layer_idx in range(num_layers):
            t0 = time.perf_counter()
            layer_original_bytes = 0
            layer_compressed_bytes = 0

            for kv_idx in range(2):
                layer_tensor = kv_cache_cpu[kv_idx, layer_idx]
                chunk = self._compress_chunk(layer_tensor)
                compressed_kv_cache[kv_idx].append(chunk)

                original_size = layer_tensor.numel() * layer_tensor.element_size()
                compressed_size = (
                    len(chunk.non_exp_bytes)
                    + len(chunk.exp_bitstream)
                    + self._estimate_codebook_size_bytes(chunk.codebook)
                )
                layer_original_bytes += original_size
                layer_compressed_bytes += compressed_size

            compress_ms = (time.perf_counter() - t0) * 1000.0
            ratio = layer_compressed_bytes / max(1, layer_original_bytes)
            layer_stats.append(
                LayerCompressionStats(
                    layer_idx=layer_idx,
                    original_size_bytes=layer_original_bytes,
                    compressed_size_bytes=layer_compressed_bytes,
                    compression_ratio=ratio,
                    compression_time_ms=compress_ms,
                )
            )

        if print_stats:
            self.print_layer_stats(layer_stats)

        return compressed_kv_cache, layer_stats

    @torch.inference_mode()
    def decompress(
        self,
        compressed_kv_caches: List[List[List[CompressedChunk]]],
        device: Optional[torch.device | str] = None,
    ) -> torch.Tensor:
        """Single-thread decompression for a batch of compressed KV caches."""
        self._assert_cpu_device(device)
        self._validate_chunk_batch(compressed_kv_caches)

        num_blocks = len(compressed_kv_caches)
        num_layers = len(compressed_kv_caches[0][0])
        block_size, num_kv_heads, head_size = compressed_kv_caches[0][0][0].orig_shape

        decompressed_kv_caches = torch.empty(
            (num_blocks, 2, num_layers, block_size, num_kv_heads, head_size),
            dtype=torch.bfloat16,
            device="cpu",
        )

        for block_idx in range(num_blocks):
            for kv_idx in range(2):
                for layer_idx in range(num_layers):
                    chunk = compressed_kv_caches[block_idx][kv_idx][layer_idx]
                    if not _HAS_HUFFMAN_EXT:
                        raise RuntimeError(
                            "Single-thread decompress requires Huffman C++ extension "
                            "(huffman_decode_symbols_cpp)"
                        )
                    non_exp = np.frombuffer(chunk.non_exp_bytes, dtype=np.uint8)
                    left, right, symbol = self._get_decode_trie(chunk.codebook)
                    exp = np.asarray(
                        huffman_decode_symbols_cpp(
                            chunk.exp_bitstream,
                            int(chunk.exp_num_valid_bits),
                            left,
                            right,
                            symbol,
                            int(chunk.exp_num_symbols),
                        ),
                        dtype=np.uint8,
                    )
                    if non_exp.size != exp.size:
                        raise RuntimeError("Size mismatch during decompression")
                    bits = (
                        (non_exp.astype(np.uint16) >> 7) << 15
                        | (exp.astype(np.uint16) << 7)
                        | (non_exp.astype(np.uint16) & 0x7F)
                    ).astype(np.uint16, copy=False)
                    decompressed_kv_caches[block_idx, kv_idx, layer_idx] = torch.from_numpy(bits.copy()).view(
                        torch.bfloat16
                    ).reshape(chunk.orig_shape)

        return decompressed_kv_caches

    @torch.inference_mode()
    def decompress_parallel(
        self,
        compressed_kv_caches: List[List[List[CompressedChunk]]],
        device: Optional[torch.device | str] = None,
        num_workers: Optional[int] = None,
    ) -> torch.Tensor:
        """Multi-thread decompression for a batch of compressed KV caches."""
        self._assert_cpu_device(device)
        self._validate_chunk_batch(compressed_kv_caches)

        num_blocks = len(compressed_kv_caches)
        num_layers = len(compressed_kv_caches[0][0])
        block_size, num_kv_heads, head_size = compressed_kv_caches[0][0][0].orig_shape

        decompressed_kv_caches = torch.empty(
            (num_blocks, 2, num_layers, block_size, num_kv_heads, head_size),
            dtype=torch.bfloat16,
            device="cpu",
        )

        workers = num_blocks if num_workers is None else self._resolve_num_workers(num_workers)

        for layer_idx in range(num_layers):
            for kv_idx in range(2):
                layer_chunks = [compressed_kv_caches[block_idx][kv_idx][layer_idx] for block_idx in range(num_blocks)]

                non_exp_bytes_list = [chunk.non_exp_bytes for chunk in layer_chunks]
                bitstream_list = [chunk.exp_bitstream for chunk in layer_chunks]
                num_valid_bits_list = [int(chunk.exp_num_valid_bits) for chunk in layer_chunks]
                expected_num_symbols_list = [int(chunk.exp_num_symbols) for chunk in layer_chunks]
                left_nodes_list: List[List[int]] = []
                right_nodes_list: List[List[int]] = []
                symbol_nodes_list: List[List[int]] = []
                for chunk in layer_chunks:
                    left, right, symbol = self._get_decode_trie(chunk.codebook)
                    left_nodes_list.append(left)
                    right_nodes_list.append(right)
                    symbol_nodes_list.append(symbol)

                layer_bits_list = decompress_layer_parallel_cpp(
                    non_exp_bytes_list,
                    bitstream_list,
                    num_valid_bits_list,
                    expected_num_symbols_list,
                    left_nodes_list,
                    right_nodes_list,
                    symbol_nodes_list,
                    int(workers),
                )

                for block_idx, bits in enumerate(layer_bits_list):
                    shape = layer_chunks[block_idx].orig_shape
                    decompressed_kv_caches[block_idx, kv_idx, layer_idx] = torch.from_numpy(bits.copy()).view(torch.bfloat16).reshape(shape)

            # layer_cpu = decompressed_kv_caches[:, :, layer_idx]
            # layer_gpu = layer_cpu.to("cuda", non_blocking=True)

        return decompressed_kv_caches

    @torch.inference_mode()
    def decompress_parallel_with_stats(
        self,
        compressed_kv_caches: List[List[List[CompressedChunk]]],
        device: Optional[torch.device | str] = None,
        num_workers: Optional[int] = None,
        print_stats: bool = True,
    ) -> Tuple[torch.Tensor, List[LayerCompressionStats]]:
        """Parallel decompress with per-layer decompression timing.

        For each layer, decompression time includes both K/V tensors across all blocks.
        """
        self._assert_cpu_device(device)
        self._validate_chunk_batch(compressed_kv_caches)

        num_blocks = len(compressed_kv_caches)
        num_layers = len(compressed_kv_caches[0][0])
        block_size, num_kv_heads, head_size = compressed_kv_caches[0][0][0].orig_shape

        decompressed_kv_caches = torch.empty(
            (num_blocks, 2, num_layers, block_size, num_kv_heads, head_size),
            dtype=torch.bfloat16,
            device="cpu",
        )

        workers = num_blocks if num_workers is None else self._resolve_num_workers(num_workers)
        layer_stats: List[LayerCompressionStats] = []

        for layer_idx in range(num_layers):
            t0 = time.perf_counter()
            layer_original_bytes = 0
            layer_compressed_bytes = 0

            for kv_idx in range(2):
                layer_chunks = [compressed_kv_caches[block_idx][kv_idx][layer_idx] for block_idx in range(num_blocks)]

                non_exp_bytes_list = [chunk.non_exp_bytes for chunk in layer_chunks]
                bitstream_list = [chunk.exp_bitstream for chunk in layer_chunks]
                num_valid_bits_list = [int(chunk.exp_num_valid_bits) for chunk in layer_chunks]
                expected_num_symbols_list = [int(chunk.exp_num_symbols) for chunk in layer_chunks]
                left_nodes_list: List[List[int]] = []
                right_nodes_list: List[List[int]] = []
                symbol_nodes_list: List[List[int]] = []
                for chunk in layer_chunks:
                    left, right, symbol = self._get_decode_trie(chunk.codebook)
                    left_nodes_list.append(left)
                    right_nodes_list.append(right)
                    symbol_nodes_list.append(symbol)
                    layer_compressed_bytes += (
                        len(chunk.non_exp_bytes)
                        + len(chunk.exp_bitstream)
                        + self._estimate_codebook_size_bytes(chunk.codebook)
                    )
                    layer_original_bytes += int(np.prod(chunk.orig_shape)) * 2

                layer_bits_list = decompress_layer_parallel_cpp(
                    non_exp_bytes_list,
                    bitstream_list,
                    num_valid_bits_list,
                    expected_num_symbols_list,
                    left_nodes_list,
                    right_nodes_list,
                    symbol_nodes_list,
                    int(workers),
                )

                for block_idx, bits in enumerate(layer_bits_list):
                    shape = layer_chunks[block_idx].orig_shape
                    decompressed_kv_caches[block_idx, kv_idx, layer_idx] = torch.from_numpy(bits.copy()).view(torch.bfloat16).reshape(shape)

            decompress_ms = (time.perf_counter() - t0) * 1000.0
            ratio = layer_compressed_bytes / max(1, layer_original_bytes)
            layer_stats.append(
                LayerCompressionStats(
                    layer_idx=layer_idx,
                    original_size_bytes=layer_original_bytes,
                    compressed_size_bytes=layer_compressed_bytes,
                    compression_ratio=ratio,
                    compression_time_ms=0.0,
                    decompression_time_ms=decompress_ms,
                )
            )

        if print_stats:
            self.print_layer_stats(layer_stats)

        return decompressed_kv_caches, layer_stats

    @staticmethod
    def print_layer_stats(layer_stats: List[LayerCompressionStats]) -> None:
        """Pretty-print per-layer compression ratio and timing."""
        print(
            f"{'layer':>5} | {'orig(KB)':>10} | {'comp(KB)':>10} | {'ratio':>8} | {'comp(ms)':>9} | {'decomp(ms)':>11}"
        )
        print("-" * 70)
        for s in layer_stats:
            decomp = "-" if s.decompression_time_ms is None else f"{s.decompression_time_ms:11.3f}"
            print(
                f"{s.layer_idx:5d} | "
                f"{s.original_size_bytes / 1024:10.2f} | "
                f"{s.compressed_size_bytes / 1024:10.2f} | "
                f"{s.compression_ratio:8.4f} | "
                f"{s.compression_time_ms:9.3f} | "
                f"{decomp}"
            )

    def compressed_size_bytes(
        self,
        compressed_kv_cache: List[List[CompressedChunk]],
    ) -> Tuple[int, int, int]:
        """Return total size plus K/V payload sizes.

        Returns:
            (total_size_bytes, k_size_bytes, v_size_bytes)
            - total_size_bytes includes payload + codebook bytes
            - k_size_bytes / v_size_bytes include payload bytes only
              (non_exp_bytes + exp_bitstream), excluding codebooks
        """
        total_size_bytes = 0
        k_size_bytes = 0
        v_size_bytes = 0

        for kv_idx, kv_chunks in enumerate(compressed_kv_cache):
            kv_payload_size = 0
            for c in kv_chunks:
                payload_size = len(c.non_exp_bytes) + len(c.exp_bitstream)
                kv_payload_size += payload_size
                total_size_bytes += payload_size
                total_size_bytes += self._estimate_codebook_size_bytes(c.codebook)

            if kv_idx == 0:
                k_size_bytes = kv_payload_size
            elif kv_idx == 1:
                v_size_bytes = kv_payload_size

        return total_size_bytes, k_size_bytes, v_size_bytes

    def _estimate_codebook_size_bytes(self, codebook: HuffmanCodebook) -> int:
        """Estimate codebook bytes using an explicit compact serialization layout."""
        # uint16 entry_count
        size = 2
        for _, code in codebook.encode_table.items():
            code_len = len(code)
            if code_len > 255:
                # With <=256 symbols this should not happen, but keep a safe fallback.
                size += 1 + 2 + (code_len + 7) // 8
            else:
                # uint8 symbol + uint8 code_len + packed code bits
                size += 1 + 1 + (code_len + 7) // 8
        return size

    def original_size_bytes(self, kv_cache: torch.Tensor) -> int:
        return kv_cache.numel() * kv_cache.element_size()

    def _validate_kv_cache(self, kv_cache: torch.Tensor):
        if kv_cache.dtype != torch.bfloat16:
            raise TypeError(
                f"HuffmanCompressor only supports torch.bfloat16, but got {kv_cache.dtype}"
            )
        if kv_cache.ndim != 5:
            raise ValueError(f"Expected kv_cache.ndim == 5, got shape={tuple(kv_cache.shape)}")
        if kv_cache.shape[0] != 2:
            raise ValueError(f"Expected kv_cache.shape[0] == 2 for (K,V), got {kv_cache.shape[0]}")

    def _validate_chunk_batch(self, compressed_kv_caches: List[List[List[CompressedChunk]]]) -> None:
        if len(compressed_kv_caches) == 0:
            raise ValueError("compressed_kv_caches must be non-empty")

        ref_layers = None
        ref_shape = None
        for block_idx, block_chunks in enumerate(compressed_kv_caches):
            if len(block_chunks) != 2:
                raise ValueError(f"Block {block_idx} should have 2 entries for K/V")
            if len(block_chunks[0]) == 0:
                raise ValueError(f"Block {block_idx} has empty K chunks")
            if len(block_chunks[1]) != len(block_chunks[0]):
                raise ValueError(f"Block {block_idx} K/V layer count mismatch")

            cur_layers = len(block_chunks[0])
            if ref_layers is None:
                ref_layers = cur_layers
            elif cur_layers != ref_layers:
                raise ValueError(
                    f"Layer count mismatch across blocks: {cur_layers} vs {ref_layers}"
                )

            for kv_idx in range(2):
                for layer_idx, chunk in enumerate(block_chunks[kv_idx]):
                    if ref_shape is None:
                        ref_shape = chunk.orig_shape
                    elif chunk.orig_shape != ref_shape:
                        raise ValueError(
                            "Chunk shape mismatch across batch: "
                            f"{chunk.orig_shape} vs {ref_shape}"
                        )

    def _compress_chunk(self, x: torch.Tensor) -> CompressedChunk:
        if x.dtype != torch.bfloat16:
            raise TypeError(f"_compress_chunk expects BF16 tensor, got {x.dtype}")
        if x.device.type != "cpu":
            raise ValueError(f"_compress_chunk expects CPU tensor, got {x.device}")

        bits = self._bf16_tensor_to_uint16_numpy(x)
        non_exp, exp = self._split_bf16_bits(bits)
        # Build one Huffman codebook per compressed chunk (not per full kv_cache block).
        # Here, a chunk is one tensor slice: [kv_idx, layer_idx, :, :, :].
        codebook = self._build_huffman_codebook(exp)
        exp_bitstream, exp_num_valid_bits = self._huffman_encode_symbols(exp, codebook)
        return CompressedChunk(
            non_exp_bytes=non_exp.tobytes(),
            exp_bitstream=exp_bitstream,
            exp_num_symbols=exp.size,
            exp_num_valid_bits=exp_num_valid_bits,
            codebook=codebook,
            orig_shape=tuple(x.shape),
        )

    def _bf16_tensor_to_uint16_numpy(self, x: torch.Tensor) -> np.ndarray:
        bits_t = x.detach().contiguous().view(torch.uint16).cpu()
        return bits_t.numpy().reshape(-1).astype(np.uint16, copy=False)

    def _split_bf16_bits(self, bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        bits = bits.astype(np.uint16, copy=False)
        sign = (bits >> 15).astype(np.uint8)
        exp = ((bits >> 7) & 0xFF).astype(np.uint8)
        mant = (bits & 0x7F).astype(np.uint8)
        return ((sign << 7) | mant).astype(np.uint8), exp

    def _build_huffman_codebook(self, symbols: np.ndarray) -> HuffmanCodebook:
        freq = np.bincount(symbols, minlength=256)
        items = [(int(f), sym) for sym, f in enumerate(freq) if f > 0]

        if len(items) == 1:
            sym = items[0][1]
            return HuffmanCodebook(encode_table={sym: "0"}, decode_table={"0": sym})

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

        encode: Dict[int, str] = {}

        def dfs(node: _HuffNode, prefix: str):
            if node.symbol is not None:
                encode[node.symbol] = prefix if prefix else "0"
                return
            dfs(node.left, prefix + "0")
            dfs(node.right, prefix + "1")

        dfs(heap[0][2], "")
        return HuffmanCodebook(encode_table=encode, decode_table={v: k for k, v in encode.items()})

    def _huffman_encode_symbols(
        self,
        symbols: np.ndarray,
        codebook: HuffmanCodebook,
    ) -> Tuple[bytes, int]:
        if _HAS_HUFFMAN_EXT:
            codes = [""] * 256
            for sym, bits in codebook.encode_table.items():
                codes[int(sym)] = bits
            encoded, num_valid_bits = huffman_encode_symbols_cpp(
                np.asarray(symbols, dtype=np.uint8),
                codes,
            )
            return encoded, int(num_valid_bits)

        encode_compiled = self._get_compiled_encode_table(codebook)
        out = bytearray()
        bit_buffer = 0
        bit_count = 0
        num_valid_bits = 0

        for s in symbols.tolist():
            code, code_len = encode_compiled[int(s)]
            bit_buffer = (bit_buffer << code_len) | code
            bit_count += code_len
            num_valid_bits += code_len

            while bit_count >= 8:
                shift = bit_count - 8
                out.append((bit_buffer >> shift) & 0xFF)
                bit_buffer &= (1 << shift) - 1
                bit_count -= 8

        if bit_count > 0:
            out.append((bit_buffer << (8 - bit_count)) & 0xFF)

        return bytes(out), num_valid_bits

    def _get_compiled_encode_table(self, codebook: HuffmanCodebook) -> Dict[int, Tuple[int, int]]:
        key = id(codebook)
        cached = self._encode_cache.get(key)
        if cached is not None:
            return cached

        compiled = {
            sym: (int(bits, 2), len(bits))
            for sym, bits in codebook.encode_table.items()
        }
        self._encode_cache[key] = compiled
        return compiled

    def _get_decode_trie(self, codebook: HuffmanCodebook) -> Tuple[List[int], List[int], List[int]]:
        key = id(codebook)
        cached = self._decode_trie_cache.get(key)
        if cached is not None:
            return cached

        left = [-1]
        right = [-1]
        symbol = [-1]

        for bits, sym in codebook.decode_table.items():
            node = 0
            for ch in bits:
                if ch == "0":
                    nxt = left[node]
                    if nxt == -1:
                        nxt = len(left)
                        left[node] = nxt
                        left.append(-1)
                        right.append(-1)
                        symbol.append(-1)
                    node = nxt
                else:
                    nxt = right[node]
                    if nxt == -1:
                        nxt = len(left)
                        right[node] = nxt
                        left.append(-1)
                        right.append(-1)
                        symbol.append(-1)
                    node = nxt
            symbol[node] = int(sym)

        trie = (left, right, symbol)
        self._decode_trie_cache[key] = trie
        return trie
