#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> quantize_pack_int4_cuda(torch::Tensor x);
torch::Tensor unpack_int4_cuda(torch::Tensor packed, int64_t orig_D);

std::vector<torch::Tensor> quantize_pack_int4(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.is_floating_point(), "x must be a floating-point tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(x.dim() >= 2, "x.dim() must be >= 2");
  TORCH_CHECK(x.size(0) > 0, "x.size(0) must be > 0");
  TORCH_CHECK(x.size(-1) > 0, "x.size(-1) must be > 0");

  return quantize_pack_int4_cuda(x);
}

torch::Tensor unpack_int4(torch::Tensor packed, int64_t orig_D) {
  TORCH_CHECK(packed.is_cuda(), "packed must be a CUDA tensor");
  TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");
  TORCH_CHECK(packed.is_contiguous(), "packed must be contiguous");
  TORCH_CHECK(orig_D > 0, "orig_D must be > 0");
  return unpack_int4_cuda(packed, orig_D);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize_pack_int4", &quantize_pack_int4, "Quantize to int4 and pack uint8 (CUDA)");
  m.def("unpack_int4", &unpack_int4, "Unpack int4 (packed uint8) -> int8 (CUDA)");
}
