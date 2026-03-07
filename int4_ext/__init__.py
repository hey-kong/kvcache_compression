"""Python wrapper for CUDA int4 extension."""

try:
    from ._C import quantize_pack_int4, unpack_int4
except ImportError as exc:
    raise ImportError(
        "Failed to import int4_ext CUDA extension (_C). "
        "Please run `pip install -e .` from repository root "
        "and ensure CUDA/PyTorch build environment is available."
    ) from exc

__all__ = ["quantize_pack_int4", "unpack_int4"]
