#include <torch/extension.h>
#include <vector>

torch::Tensor pack_int4_cuda(torch::Tensor q);
torch::Tensor unpack_int4_cuda(torch::Tensor packed, int64_t orig_D);

torch::Tensor pack_int4(torch::Tensor q) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(q.dtype() == torch::kInt8, "q must be int8");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(q.size(-1) > 0, "q.size(-1) must be > 0");
  return pack_int4_cuda(q);
}

torch::Tensor unpack_int4(torch::Tensor packed, int64_t orig_D) {
  TORCH_CHECK(packed.is_cuda(), "packed must be a CUDA tensor");
  TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");
  TORCH_CHECK(packed.is_contiguous(), "packed must be contiguous");
  TORCH_CHECK(orig_D > 0, "orig_D must be > 0");
  return unpack_int4_cuda(packed, orig_D);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_int4", &pack_int4, "Pack int8 [-8,7] -> packed uint8 (CUDA)");
  m.def("unpack_int4", &unpack_int4, "Unpack int4 (packed uint8) -> int8 (CUDA)");
}
