from __future__ import annotations

import heapq
import os
from concurrent.futures import ThreadPoolExecutor
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
        out: List[List[CompressedChunk]] = [[], []]
        for kv_idx in range(2):
            for layer_idx in range(num_layers):
                out[kv_idx].append(self._compress_chunk(kv_cache_cpu[kv_idx, layer_idx]))
        return out

    @torch.inference_mode()
    def batch_compress(
        self,
        kv_caches: torch.Tensor,
        num_workers: Optional[int] = None,
    ) -> List[List[List[CompressedChunk]]]:
        if kv_caches.ndim != 6:
            raise ValueError(
                "Expected kv_caches shape "
                "(num_blocks, 2, num_layers, block_size, num_kv_heads, head_size), "
                f"got {tuple(kv_caches.shape)}"
            )
        if kv_caches.shape[1] != 2:
            raise ValueError(f"Expected kv_caches.shape[1] == 2, got {kv_caches.shape[1]}")

        kv_caches_cpu = self._to_cpu_bf16(kv_caches)
        num_blocks, _, num_layers, _, _, _ = kv_caches_cpu.shape

        out: List[List[List[CompressedChunk]]] = [
            [[None for _ in range(num_layers)] for _ in range(2)]
            for _ in range(num_blocks)
        ]

        workers = num_blocks if num_workers is None else self._resolve_num_workers(num_workers)

        def work_block(block_idx: int) -> Tuple[int, List[List[CompressedChunk]]]:
            block_out: List[List[CompressedChunk]] = [[], []]
            for layer_idx in range(num_layers):
                for kv_idx in range(2):
                    block_out[kv_idx].append(
                        self._compress_chunk(kv_caches_cpu[block_idx, kv_idx, layer_idx])
                    )
            return block_idx, block_out

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for block_idx, block_out in ex.map(work_block, range(num_blocks)):
                out[block_idx] = block_out

        return out

    @torch.inference_mode()
    def decompress(
        self,
        chunk: CompressedChunk,
        device: Optional[torch.device | str] = None,
    ) -> torch.Tensor:
        self._assert_cpu_device(device)
        return self._decompress_chunk(chunk)

    @torch.inference_mode()
    def batch_decompress(
        self,
        chunks_batch: List[List[List[CompressedChunk]]],
        device: Optional[torch.device | str] = None,
        num_workers: Optional[int] = None,
    ) -> torch.Tensor:
        self._assert_cpu_device(device)
        self._validate_chunk_batch(chunks_batch)

        num_blocks = len(chunks_batch)
        num_layers = len(chunks_batch[0][0])
        block_size, num_kv_heads, head_size = chunks_batch[0][0][0].orig_shape

        out = torch.empty(
            (num_blocks, 2, num_layers, block_size, num_kv_heads, head_size),
            dtype=torch.bfloat16,
            device="cpu",
        )

        workers = num_blocks if num_workers is None else self._resolve_num_workers(num_workers)

        if _HAS_HUFFMAN_EXT:
            for layer_idx in range(num_layers):
                for kv_idx in range(2):
                    layer_chunks = [chunks_batch[block_idx][kv_idx][layer_idx] for block_idx in range(num_blocks)]

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
                        out[block_idx, kv_idx, layer_idx] = torch.from_numpy(bits.copy()).view(torch.bfloat16).reshape(shape)
        else:
            def work_block_layer(block_idx: int, layer_idx: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
                k = self._decompress_chunk(chunks_batch[block_idx][0][layer_idx])
                v = self._decompress_chunk(chunks_batch[block_idx][1][layer_idx])
                return block_idx, k, v

            with ThreadPoolExecutor(max_workers=workers) as ex:
                for layer_idx in range(num_layers):
                    for block_idx, k, v in ex.map(
                        lambda b: work_block_layer(b, layer_idx), range(num_blocks)
                    ):
                        out[block_idx, 0, layer_idx] = k
                        out[block_idx, 1, layer_idx] = v

        return out

    @torch.inference_mode()
    def reconstruct_kv_cache(
        self,
        chunks: List[List[CompressedChunk]],
        device: Optional[torch.device | str] = None,
    ) -> torch.Tensor:
        self._assert_cpu_device(device)
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

        block_size, num_kv_heads, head_size = chunks[0][0].orig_shape
        out = torch.empty(
            (2, num_layers, block_size, num_kv_heads, head_size),
            dtype=torch.bfloat16,
            device="cpu",
        )

        for kv_idx in range(2):
            for layer_idx in range(num_layers):
                out[kv_idx, layer_idx] = self._decompress_chunk(chunks[kv_idx][layer_idx])

        return out

    def compressed_size_bytes(self, chunks: List[List[CompressedChunk]]) -> int:
        total = 0
        for kv_chunks in chunks:
            for c in kv_chunks:
                total += len(c.non_exp_bytes)
                total += len(c.exp_bitstream)
                for _, code in c.codebook.encode_table.items():
                    total += 1 + 2 + (len(code) + 7) // 8
        return total

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

    def _validate_chunk_batch(self, chunks_batch: List[List[List[CompressedChunk]]]) -> None:
        if len(chunks_batch) == 0:
            raise ValueError("chunks_batch must be non-empty")

        ref_layers = None
        ref_shape = None
        for block_idx, block_chunks in enumerate(chunks_batch):
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

    def _decompress_chunk(self, chunk: CompressedChunk) -> torch.Tensor:
        non_exp = np.frombuffer(chunk.non_exp_bytes, dtype=np.uint8).copy()
        exp = self._huffman_decode_symbols(
            bitstream=chunk.exp_bitstream,
            num_valid_bits=chunk.exp_num_valid_bits,
            codebook=chunk.codebook,
            expected_num_symbols=chunk.exp_num_symbols,
        )
        if non_exp.size != exp.size:
            raise ValueError(
                f"Size mismatch during decompression: non_exp.size={non_exp.size}, exp.size={exp.size}"
            )

        bits = self._merge_bf16_bits(non_exp, exp)
        return self._uint16_numpy_to_bf16_tensor(bits, chunk.orig_shape)

    def _bf16_tensor_to_uint16_numpy(self, x: torch.Tensor) -> np.ndarray:
        bits_t = x.detach().contiguous().view(torch.uint16).cpu()
        return bits_t.numpy().reshape(-1).astype(np.uint16, copy=False)

    def _uint16_numpy_to_bf16_tensor(self, bits: np.ndarray, shape: Tuple[int, ...]) -> torch.Tensor:
        bits = np.asarray(bits, dtype=np.uint16).reshape(-1)
        return torch.from_numpy(bits.copy()).view(torch.bfloat16).reshape(shape)

    def _split_bf16_bits(self, bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        bits = bits.astype(np.uint16, copy=False)
        sign = (bits >> 15).astype(np.uint8)
        exp = ((bits >> 7) & 0xFF).astype(np.uint8)
        mant = (bits & 0x7F).astype(np.uint8)
        return ((sign << 7) | mant).astype(np.uint8), exp

    def _merge_bf16_bits(self, non_exp: np.ndarray, exp: np.ndarray) -> np.ndarray:
        non_exp = np.asarray(non_exp, dtype=np.uint8)
        exp = np.asarray(exp, dtype=np.uint8)
        sign = (non_exp >> 7).astype(np.uint16)
        mant = (non_exp & 0x7F).astype(np.uint16)
        exp16 = exp.astype(np.uint16)
        return ((sign << 15) | (exp16 << 7) | mant).astype(np.uint16)

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

    def _huffman_decode_symbols(
        self,
        bitstream: bytes,
        num_valid_bits: int,
        codebook: HuffmanCodebook,
        expected_num_symbols: int,
    ) -> np.ndarray:
        left, right, symbol = self._get_decode_trie(codebook)
        if _HAS_HUFFMAN_EXT:
            return huffman_decode_symbols_cpp(
                bitstream,
                int(num_valid_bits),
                left,
                right,
                symbol,
                int(expected_num_symbols),
            )

        node = 0
        out = np.empty(expected_num_symbols, dtype=np.uint8)
        out_idx = 0
        bits_read = 0

        for byte in bitstream:
            for bit_idx in range(7, -1, -1):
                if bits_read == num_valid_bits:
                    break
                node = right[node] if ((byte >> bit_idx) & 1) else left[node]
                if node < 0:
                    raise ValueError("Invalid Huffman bitstream for this codebook")
                sym = symbol[node]
                if sym >= 0:
                    if out_idx >= expected_num_symbols:
                        raise ValueError("Decoded more symbols than expected")
                    out[out_idx] = sym
                    out_idx += 1
                    node = 0
                bits_read += 1
            if bits_read == num_valid_bits:
                break

        if out_idx != expected_num_symbols:
            raise ValueError(
                f"Huffman decode failed: expected {expected_num_symbols} symbols, got {out_idx}"
            )
        return out

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
