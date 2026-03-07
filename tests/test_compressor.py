import pytest
import torch

from compressor import QuantizedCompressor


def _make_kv_cache(dtype=torch.float16):
    torch.manual_seed(0)
    # (2, num_layers, block_size, num_kv_heads, head_size)
    return torch.randn(2, 3, 8, 4, 15, device="cuda", dtype=dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for int4_ext")
def test_compress_kv_cache_int4_roundtrip():
    compressor = QuantizedCompressor(bits=4)
    kv_cache = _make_kv_cache()

    ck, cv, meta = compressor.compress(kv_cache, mode="kv")

    assert ck.dtype == torch.uint8
    assert cv.dtype == torch.uint8
    assert meta.k is not None and meta.v is not None
    assert meta.k.last_dim == kv_cache.shape[-1]
    assert meta.v.last_dim == kv_cache.shape[-1]

    restored = compressor.decompress(torch.stack([ck, cv], dim=0), meta)

    assert restored.shape == kv_cache.shape

    # 4-bit quantization has visible error; keep tolerance practical.
    assert torch.allclose(restored[0], kv_cache[0], atol=0.35, rtol=0.35)
    assert torch.allclose(restored[1], kv_cache[1], atol=0.35, rtol=0.35)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for int4_ext")
def test_compress_mode_k_only_keeps_v():
    compressor = QuantizedCompressor(bits=4)
    kv_cache = _make_kv_cache()

    ck, cv, meta = compressor.compress(kv_cache, mode="k")

    assert ck.dtype == torch.uint8
    assert torch.equal(cv, kv_cache[1])
    assert meta.k is not None and meta.v is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for int4_ext")
def test_compress_mode_v_only_keeps_k():
    compressor = QuantizedCompressor(bits=4)
    kv_cache = _make_kv_cache()

    ck, cv, meta = compressor.compress(kv_cache, mode="v")

    assert torch.equal(ck, kv_cache[0])
    assert cv.dtype == torch.uint8
    assert meta.k is None and meta.v is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for int4_ext")
def test_decompress_restores_kv_layout_for_partial_mode():
    compressor = QuantizedCompressor(bits=4)
    kv_cache = _make_kv_cache()

    ck, cv, meta = compressor.compress(kv_cache, mode="k")
    restored = compressor.decompress(torch.stack([ck, cv], dim=0), meta)

    assert restored.shape == kv_cache.shape
    # K is quantized/dequantized, V should stay exact in mode="k"
    assert torch.allclose(restored[0], kv_cache[0], atol=0.35, rtol=0.35)
    assert torch.equal(restored[1], kv_cache[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for int4_ext")
def test_layerwise_quantization_scales_differ_by_layer():
    compressor = QuantizedCompressor(bits=4)

    kv_cache = torch.randn(2, 3, 8, 4, 16, device="cuda", dtype=torch.float16)
    # Enforce very different per-layer ranges so per-layer scales should differ clearly.
    kv_cache[:, 0] *= 0.1
    kv_cache[:, 1] *= 1.0
    kv_cache[:, 2] *= 10.0

    _, _, meta = compressor.compress(kv_cache, mode="kv")

    assert meta.k is not None and meta.v is not None
    assert meta.k.scale.shape == (3, 1, 1, 1)
    assert meta.v.scale.shape == (3, 1, 1, 1)

    # layer-2 scale should be much larger than layer-0 due to input scaling above.
    assert meta.k.scale[2].item() > meta.k.scale[0].item()
    assert meta.v.scale[2].item() > meta.v.scale[0].item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for int4_ext")
def test_batch_decompress_restores_batched_layout():
    compressor = QuantizedCompressor(bits=4)

    kv_blocks = torch.stack([_make_kv_cache(), _make_kv_cache()], dim=0)

    packed_blocks = []
    metas = []
    for i in range(kv_blocks.size(0)):
        ck, cv, meta = compressor.compress(kv_blocks[i], mode="kv")
        packed_blocks.append(torch.stack([ck, cv], dim=0))
        metas.append(meta)

    packed_batch = torch.stack(packed_blocks, dim=0)
    restored = compressor.batch_decompress(packed_batch, metas)

    assert restored.shape == kv_blocks.shape
    assert torch.allclose(restored[:, 0], kv_blocks[:, 0], atol=0.35, rtol=0.35)
    assert torch.allclose(restored[:, 1], kv_blocks[:, 1], atol=0.35, rtol=0.35)


def test_decompress_meta_type_validation():
    compressor = QuantizedCompressor(bits=4)
    kv_cache = torch.randn(2, 3, 8, 4, 15)
    with pytest.raises(TypeError, match="KVQuantMeta"):
        compressor.decompress(kv_cache, meta=None)


def test_batch_decompress_meta_type_validation():
    compressor = QuantizedCompressor(bits=4)
    kv_caches = torch.randn(2, 2, 3, 8, 4, 15)
    with pytest.raises(TypeError, match="KVQuantMeta"):
        compressor.batch_decompress(kv_caches, metas=[None, None])
