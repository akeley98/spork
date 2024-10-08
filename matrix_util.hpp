#pragma once
#include <memory>
#include <stdint.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

struct CudaDeleter
{
    void operator() (const void* victim)
    {
        cudaFree(const_cast<void*>(victim));
    }
};

template <typename T>
using unique_cuda_array = std::unique_ptr<T[], CudaDeleter>;

template <typename T, typename Int_t>
unique_cuda_array<T> make_unique_cuda_array(Int_t w, Int_t h = 1)
{
    void* ptr = nullptr;
    cudaMalloc(&ptr, size_t(w) * size_t(h) * sizeof(T));
    if (ptr == 0 && w != 0 && h != 0) {
        throw std::bad_alloc{};
    }
    return unique_cuda_array<T>(static_cast<T*>(ptr));
}

void launch_random_bf16_kernel(cudaStream_t stream, size_t w, size_t h, __nv_bfloat16* array, int seed);

template <typename Int_t>
unique_cuda_array<__nv_bfloat16> make_randomized_unique_cuda_array(cudaStream_t stream, Int_t w, Int_t h, int seed)
{
    auto array = make_unique_cuda_array<__nv_bfloat16>(w, h);
    launch_random_bf16_kernel(stream, size_t(w), size_t(h), array.get(), seed);
    return array;
}

struct HostAttnArgs
{
    cudaStream_t stream;
    int N, d;
    unique_cuda_array<__nv_bfloat16> q, k, v;
};

inline HostAttnArgs make_attn_args(cudaStream_t stream, int N, int d, int seed)
{
    HostAttnArgs args;
    args.stream = stream;
    args.N = N;
    args.d = d;
    args.q = make_randomized_unique_cuda_array(stream, N, d, seed);
    args.k = make_randomized_unique_cuda_array(stream, N, d, seed ^ 888);
    args.v = make_randomized_unique_cuda_array(stream, N, d, seed ^ 1776);
    return args;
}

struct DeviceAttnArgs
{
    int N, d;
    const __nv_bfloat16 *__restrict__ q;
    const __nv_bfloat16 *__restrict__ k;
    const __nv_bfloat16 *__restrict__ v;

    DeviceAttnArgs(const HostAttnArgs& a) : N(a.N), d(a.d), q(a.q.get()), k(a.k.get()), v(a.v.get()) {}
};

#if 0
// Compare two cuda device matrices for exact equality.
// *d_out must be 0 initially, and is overwritten if a problem is found.
void check_equal_cuda_arrays(cudaStream_t stream, size_t w, size_t h,
                             const __nv_bfloat16* a,
                             const __nv_bfloat16* b,
                             uint64_t* d_out);
#endif
