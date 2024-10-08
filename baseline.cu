#include "baseline.hpp"

#include <math.h>
#include <stdio.h>

#include "matrix_util.hpp"

namespace baseline {

#if 0

// Thread (x,y) computes output (row = x, col = y)
__global__ void mul_qk_transpose_kernel(int N, int d, const __nv_bfloat16* q, const __nv_bfloat16* k, __nv_bfloat16* s)
{
    const uint32_t r = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t c = blockDim.y * blockIdx.y + threadIdx.y;
    if (r < N && c < N) {
        float accum = 0.0f;
        for (uint32_t i = 0; i < d; ++i) {
            // Dot product of q's row r and k's row c.
            accum += float(q[i*N + r]) * float(k[i*N + c]);
        }
        s[c*N + r] = __nv_bfloat16(accum);  // N×N matrix out
    }
}

// Thread x computes row x
// Very inefficient access pattern for column-major matrix
__global__ void row_softmax_kernel(int N, const __nv_bfloat16* s, __nv_bfloat16* p)
{
    const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x < N) {
        float max_accum = 0.0f;
        for (uint32_t i = 0; i < N; ++i) {
            max_accum = fmaxf(max_accum, s[i*N + x]);
        }

        float exp_accum = 0.0f;
        for (uint32_t i = 0; i < N; ++i) {
            const float exp_value = __expf(float(s[i*N + x]) - max_accum);
            p[i*N + x] = exp_value;  // Save for later
            exp_accum += exp_value;
        }
        const float rcp = 1.0f / exp_accum;
        for (uint32_t i = 0; i < N; ++i) {
            p[i*N + x] *= rcp;  // N×N matrix out
        }
    }
}

// Thread (x,y) computes output (row = x, col = y)
// x = 0, 1, 2, ... , N - 1
// y = 0, 1, 2, ... , d - 1
__global__ void mul_pv_kernel(int N, int d, const __nv_bfloat16* p, const __nv_bfloat16* v, __nv_bfloat16* o)
{
    const uint32_t r = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t c = blockDim.y * blockIdx.y + threadIdx.y;
    if (r < N && c < d) {
        float accum = 0.0f;
        for (uint32_t i = 0; i < N; ++i) {
            // Dot product of p's row r and v's column c
            accum += float(p[i*N + r]) * float(v[c*N + i]);
        }
        o[c*N + r] = __nv_bfloat16(accum);  // N×d matrix out
    }
}

#endif

// Thread (x,y) computes output (row = x, col = y)
__global__ void mul_qk_transpose_kernel(int N, int d, const __nv_bfloat16* q, const __nv_bfloat16* k, __nv_bfloat16* s)
{
    const uint32_t r = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t c = blockDim.y * blockIdx.y + threadIdx.y;
    if (r < N && c < N) {
        float accum = 0.0f;
        for (uint32_t i = 0; i < d; ++i) {
            // Dot product of q's row r and k's row c.
            accum += float(q[i*N + r]) * float(k[i*N + c]);
        }
        s[c*N + r] = __nv_bfloat16(accum);  // N×N matrix out
    }
}

// Thread x computes row x
__global__ void row_softmax_kernel(int N, const __nv_bfloat16* s, __nv_bfloat16* p)
{
    const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x < N) {
        float max_accum = 0.0f;
        for (uint32_t i = 0; i < N; ++i) {
            max_accum = fmaxf(max_accum, s[i + x*N]);
        }

        float exp_accum = 0.0f;
        for (uint32_t i = 0; i < N; ++i) {
            const float exp_value = __expf(float(s[i + x*N]) - max_accum);
            p[i + x*N] = exp_value;  // Save for later
            exp_accum += exp_value;
        }
        const float rcp = 1.0f / exp_accum;
        for (uint32_t i = 0; i < N; ++i) {
            p[i + x*N] *= rcp;  // N×N matrix out
        }
    }
}

// Thread (x,y) computes output (row = x, col = y)
// x = 0, 1, 2, ... , N - 1
// y = 0, 1, 2, ... , d - 1
__global__ void mul_pv_kernel(int N, int d, const __nv_bfloat16* p, const __nv_bfloat16* v, __nv_bfloat16* o)
{
    const uint32_t r = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t c = blockDim.y * blockIdx.y + threadIdx.y;
    if (r < N && c < d) {
        float accum = 0.0f;
        for (uint32_t i = 0; i < N; ++i) {
            // Dot product of p's row r and v's column c
            accum += float(p[i + r*N]) * float(v[c + i*d]);
        }
        o[c + r*d] = __nv_bfloat16(accum);  // N×d matrix out
    }
}

}  // end namespace baseline

void launch_baseline_kernel(const HostAttnArgs& args, __nv_bfloat16* o)
{
    // (inefficient) scratch memory.
    auto s = make_unique_cuda_array<__nv_bfloat16>(args.N, args.N);
    auto p = make_unique_cuda_array<__nv_bfloat16>(args.N, args.N);

    {
        dim3 block{16, 16, 1};
        dim3 grid{(args.N + 15u) / 16u, (args.N + 15u) / 16u, 1};
        baseline::mul_qk_transpose_kernel<<<grid, block, 0, args.stream>>>(
                args.N, args.d, args.q.get(), args.k.get(), s.get());
    }
    {
        dim3 block{256, 1, 1};
        dim3 grid{(args.N + 255u) / 256u, 1, 1};
        baseline::row_softmax_kernel<<<grid, block, 0, args.stream>>>(args.N, s.get(), p.get());
    }
    {
        dim3 block{16, 16, 1};
        dim3 grid{(args.N + 15u) / 16u, (args.d + 15u) / 16u, 1u};
        baseline::mul_pv_kernel<<<grid, block, 0, args.stream>>>(args.N, args.d, p.get(), args.v.get(), o);
    }
}
