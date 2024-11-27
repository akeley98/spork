#include "gemm_test.h"

#include <stdlib.h>
#include <stdio.h>

#include "gemm_sm80.h"
#include "exo_par.h"

namespace {

// Copied pseudo random number generation code.
// http://www.jcgt.org/published/0009/03/02/
// Hash Functions for GPU Rendering, Mark Jarzynski, Marc Olano, NVIDIA
__device__ uint64_t pcg3d(uint32_t x, uint32_t y, uint32_t z)
{
  x = x*1664525u + 1013904223u;
  y = y*1664525u + 1013904223u;
  z = z*1664525u + 1013904223u;

  x += y*z;
  y += z*x;
  z += x*y;

  x ^= x >> 16u;
  y ^= y >> 16u;
  z ^= z >> 16u;

  x += y*z;
  y += z*x;
  z += x*y;

  return x ^ uint64_t(y) << 12u ^ uint64_t(z) << 24u;
}

__global__ void device_init_test_data(float* d_tensor, uint32_t rows, uint32_t cols, TestDataCode code)
{
    uint32_t tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    for (uint32_t y = tid_y; y < rows; y += blockDim.y * gridDim.y) {
        for (uint32_t x = tid_x; x < cols; x += blockDim.x * gridDim.x) {
            float value;
            switch (code) {
              case TestDataCode::identity:
                value = x == y ? 1.0f : 0.0f;
                break;
              case TestDataCode::tiled_numbers:
                value = (x % 64) + 100.0f * (y % 64);
                break;
              case TestDataCode::random:
              default:
                value = float(pcg3d(x, y, 19980724) % 20010106u);
                break;
            }
            d_tensor[y * cols + x] = value;
        }
    }
}

__global__ void device_compare_tensor_test_init_bitfield(unsigned long long* d_bitfield)
{
    *d_bitfield = UINT64_MAX;
}

// Requires that *d_bitfield is initialized to UINT64_MAX.
// Compare the two equal-sized matrices and, if any comparison failures, put the coordinates of the wrong value
// into *d_bitfield, packed as (row << 32 | col).
__global__ void device_compare_tensor_test(const float* d_a, const float* d_b, uint32_t rows, uint32_t cols,
                                           unsigned long long* d_bitfield)
{
    uint32_t tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    for (uint32_t y = tid_y; y < rows; y += blockDim.y * gridDim.y) {
        for (uint32_t x = tid_x; x < cols; x += blockDim.x * gridDim.x) {
            float a = d_a[y * cols + x];
            float b = d_b[y * cols + x];
            bool correct = a * b >= 0.0f;  // Sign error, or inf/nan if wrong
            if (correct) {
                a = fabsf(a);
                b = fabsf(b);
                const float m = fminf(a, b);
                const float M = fmaxf(a, b);
                correct = M == 0 || M / m < (1.0f + 1/32.0f);
            }
            if (!correct) {
                unsigned long long packed_coords = uint64_t(y) << 32u | x;
                atomicMin(d_bitfield, packed_coords);
            }
        }
    }
}

// Print info on wrong value from function above.
__global__ void device_compare_tensor_test_print(TestParams test_params,
                                                 const float* d_a, const float* d_b, uint32_t rows, uint32_t cols,
                                                 unsigned long long* d_bitfield)
{
    uint32_t y = uint32_t(*d_bitfield >> 32u);
    uint32_t x = uint32_t(*d_bitfield);

    if (x < cols && y < rows) {
        const float a = d_a[y * cols + x];
        const float b = d_b[y * cols + x];
        printf("TestParams{%u,%u,%u, %i,%i} [%u,%u] %g != %g\n", test_params.M, test_params.N, test_params.K,
               static_cast<int>(test_params.test_data_code_A), static_cast<int>(test_params.test_data_code_B),
               y, x, a, b);
    }
}

void launch_device_compare_tensor(TestParams test_params,
                                  const float* d_a, const float* d_b, uint32_t rows, uint32_t cols,
                                  unsigned long long* d_bitfield, cudaStream_t stream)
{
    dim3 grid{(cols + 15u) / 16u, (rows + 15u) / 16u, 1};
    dim3 block{16, 16, 1};
    device_compare_tensor_test_init_bitfield<<<1, 1, 0, stream>>>(d_bitfield);
    device_compare_tensor_test<<<grid, block, 0, stream>>>(d_a, d_b, rows, cols, d_bitfield);
    device_compare_tensor_test_print<<<1, 1, 0, stream>>>(test_params, d_a, d_b, rows, cols, d_bitfield);
}

}  // end namespace

void gemm_test(TestParams params, cudaStream_t stream)
{
    unsigned long long* d_bitfield = nullptr;
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c_sm80 = nullptr;
    float* d_c_sm90_warmup = nullptr;
    float* d_c_sm90_tested = nullptr;

    cudaMallocAsync(&d_bitfield, sizeof(unsigned long long), stream);
    cudaMallocAsync(&d_a, params.M * params.K * sizeof(float), stream);
    cudaMallocAsync(&d_b, params.N * params.K * sizeof(float), stream);
    cudaMallocAsync(&d_c_sm80, params.M * params.N * sizeof(float), stream);
    cudaMallocAsync(&d_c_sm90_warmup, params.M * params.N * sizeof(float), stream);
    cudaMallocAsync(&d_c_sm90_tested, params.M * params.N * sizeof(float), stream);

    if (!d_bitfield || !d_a || !d_b || !d_c_sm80 || !d_c_sm90_warmup || !d_c_sm90_tested) {
        fprintf(stderr, "Out of GPU memory\n");
        exit(1);
    }

    // Initialize "random" test data
    {
        dim3 grid_a{(params.K + 15u) / 16u, (params.M + 15u) / 16u, 1};
        dim3 grid_b{(params.N + 15u) / 16u, (params.K + 15u) / 16u, 1};
        dim3 block{16, 16, 1};
        device_init_test_data<<<grid_a, block, 0, stream>>>(d_a, params.M, params.K, params.test_data_code_A);
        device_init_test_data<<<grid_b, block, 0, stream>>>(d_b, params.K, params.N, params.test_data_code_B);
    }

    // Initialize SM80 data
    {
        GPU_Tensors t{params.M, params.N, params.K, d_a, d_b, d_c_sm80};
        matmul_sm80(t, stream);
    }

    // XXX
    {
        gpu_gemm(nullptr, params.M, params.N, params.K, d_a, d_b, d_c_sm90_warmup);
    }
    launch_device_compare_tensor(params, d_c_sm80, d_c_sm90_warmup, params.M, params.N, d_bitfield, stream);

    cudaFreeAsync(d_a, stream);
    cudaFreeAsync(d_b, stream);
    cudaFreeAsync(d_c_sm80, stream);
    cudaFreeAsync(d_c_sm90_warmup, stream);
    cudaFreeAsync(d_c_sm90_tested, stream);
}
