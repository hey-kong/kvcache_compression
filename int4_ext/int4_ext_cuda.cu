#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>

static inline __device__ int8_t sign_extend_4bit(uint8_t v) {
    return (int8_t)((int8_t)(v << 4) >> 4);
}

__global__ void unpack_int4_kernel_2d(
    const uint8_t* __restrict__ packed,
    int8_t* __restrict__ out,
    int64_t rows,
    int64_t orig_D,
    int64_t packed_D) {

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_elems = rows * orig_D;
    if (idx >= n_elems) return;

    int64_t row = idx / orig_D;
    int64_t d   = idx - row * orig_D;

    int64_t p = d >> 1;
    uint8_t byte = packed[row * packed_D + p];
    uint8_t nibble = (d & 1) ? (byte >> 4) : (byte & 0x0F);
    out[row * orig_D + d] = sign_extend_4bit(nibble);
}

at::Tensor unpack_int4_cuda(at::Tensor packed, int64_t orig_D) {
    c10::cuda::CUDAGuard device_guard(packed.device());

    auto sizes = packed.sizes();
    int64_t packed_D = sizes.back();

    int64_t rows = 1;
    for (int i = 0; i < packed.dim() - 1; ++i) rows *= sizes[i];

    std::vector<int64_t> out_sizes(sizes.begin(), sizes.end());
    out_sizes.back() = orig_D;

    auto out = torch::empty(out_sizes, packed.options().dtype(torch::kInt8));

    auto packed_2d = packed.view({rows, packed_D});
    auto out_2d    = out.view({rows, orig_D});

    int threads = 256;
    int64_t n = rows * orig_D;
    int blocks = (int)((n + threads - 1) / threads);

    auto stream = c10::cuda::getCurrentCUDAStream(packed.device().index());

    unpack_int4_kernel_2d<<<blocks, threads, 0, stream.stream()>>>(
        packed_2d.data_ptr<uint8_t>(),
        out_2d.data_ptr<int8_t>(),
        rows, orig_D, packed_D
    );

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "unpack_int4_kernel launch failed: ", cudaGetErrorString(err));
    }
    return out;
}

// ---------------- pack ----------------

__global__ void pack_int4_kernel_2d(
    const int8_t* __restrict__ q,
    uint8_t* __restrict__ packed,
    int64_t rows,
    int64_t orig_D,
    int64_t packed_D) {

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = rows * packed_D;
    if (idx >= n) return;

    int64_t row = idx / packed_D;
    int64_t p   = idx - row * packed_D;

    int64_t d0 = p * 2;
    int64_t d1 = d0 + 1;

    int8_t v0 = q[row * orig_D + d0];
    v0 = (v0 < -8) ? -8 : (v0 > 7 ? 7 : v0);
    uint8_t low = ((uint8_t)v0) & 0x0F;

    uint8_t high = 0;
    if (d1 < orig_D) {
        int8_t v1 = q[row * orig_D + d1];
        v1 = (v1 < -8) ? -8 : (v1 > 7 ? 7 : v1);
        high = (((uint8_t)v1) & 0x0F) << 4;
    }

    packed[row * packed_D + p] = (uint8_t)(low | high);
}

at::Tensor pack_int4_cuda(at::Tensor q, int64_t orig_D) {
    c10::cuda::CUDAGuard device_guard(q.device());

    auto sizes = q.sizes();
    int64_t rows = 1;
    for (int i = 0; i < q.dim() - 1; ++i) rows *= sizes[i];

    int64_t packed_D = (orig_D + 1) / 2;

    std::vector<int64_t> out_sizes(sizes.begin(), sizes.end());
    out_sizes.back() = packed_D;

    auto packed = torch::empty(out_sizes, q.options().dtype(torch::kUInt8));

    auto q_2d = q.view({rows, orig_D});
    auto p_2d = packed.view({rows, packed_D});

    int threads = 256;
    int64_t n = rows * packed_D;
    int blocks = (int)((n + threads - 1) / threads);

    auto stream = c10::cuda::getCurrentCUDAStream(q.device().index());

    pack_int4_kernel_2d<<<blocks, threads, 0, stream.stream()>>>(
        q_2d.data_ptr<int8_t>(),
        p_2d.data_ptr<uint8_t>(),
        rows, orig_D, packed_D
    );

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "pack_int4_kernel launch failed: ", cudaGetErrorString(err));
    }
    return packed;
}
