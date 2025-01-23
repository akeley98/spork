#include "gemm_test.h"

#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

#include "cute_gemm.h"
#include "cutlass_gemm.h"
#include "gemm_sm80.h"
#include "gemm_sm90.h"
#include "xgemm.h"

#include "cutlass/arch/synclog.hpp"

#ifndef CUBLAS_TEST_ENABLED
#define CUBLAS_TEST_ENABLED 1
#endif

#if CUBLAS_TEST_ENABLED
#include <cublas_v2.h>
#endif

#ifndef CUTLASS_TEST_ENABLED
#define CUTLASS_TEST_ENABLED 1
#endif

namespace gemm_test_impl {

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

template <typename T>
__global__ void device_init_test_data(T* d_tensor, uint32_t rows, uint32_t cols,
                                      TestDataCode code, bool transpose_rule)
{
    uint32_t tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    for (uint32_t r = tid_y; r < rows; r += blockDim.y * gridDim.y) {
        for (uint32_t c = tid_x; c < cols; c += blockDim.x * gridDim.x) {
            T value;
            const uint32_t x = transpose_rule ? r : c;
            const uint32_t y = transpose_rule ? c : r;
            switch (code) {
              case TestDataCode::identity:
                value = x == y ? T(1) : T(0);
                break;
              case TestDataCode::tiled_numbers:
                value = T((x % 64) + 100 * (y % 64));
                break;
              case TestDataCode::random:
              default:
                {
                    const auto randbits = pcg3d(x, y, 20010106);
                    if (randbits % 100'000u == 0) {
                        // 1 in 100'000 chance of a "big" value (1000).
                        // This greatly reduces the chance that a genuine bug is mistaken for fp error.
                        value = T(1000);
                    }
                    else if (randbits % 4u != 0u) {
                        value = T(0);  // 75% chance of a 0
                    }
                    else {
                        // 25% chance of random value [0, 1], biased towards small numbers.
                        value = T((pcg3d(x, y, 19980724) % 1'000'000) * 1e-6f);
                        value = (value * value) * (value * value);
                    }
                }
                break;
            }
            d_tensor[r * cols + c] = value;
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
template <typename TA, typename TB>
__global__ void device_compare_tensor_test(const TA* d_a, const TB* d_b, uint32_t rows, uint32_t cols,
                                           unsigned long long* d_bitfield)
{
    uint32_t tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    for (uint32_t y = tid_y; y < rows; y += blockDim.y * gridDim.y) {
        for (uint32_t x = tid_x; x < cols; x += blockDim.x * gridDim.x) {
            float a = static_cast<float>(d_a[y * cols + x]);
            float b = static_cast<float>(d_b[y * cols + x]);
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
template <typename TA, typename TB>
__global__ void device_compare_tensor_test_print(TestParams test_params,
                                                 const TA* d_a, const TB* d_b, uint32_t rows, uint32_t cols,
                                                 unsigned long long* d_bitfield)
{
    uint32_t y = uint32_t(*d_bitfield >> 32u);
    uint32_t x = uint32_t(*d_bitfield);

    if (x < cols && y < rows) {
        const float a = static_cast<float>(d_a[y * cols + x]);
        const float b = static_cast<float>(d_b[y * cols + x]);
        printf("TestParams{%u,%u,%u, %i,%i} [%u,%u] %g != %g\n", test_params.M, test_params.N, test_params.K,
               static_cast<int>(test_params.test_data_code_A), static_cast<int>(test_params.test_data_code_B),
               y, x, a, b);
        uint32_t y_min = y < 2 ? 0u : y - 2;
        uint32_t y_max = y + 2 >= rows ? rows - 1u : y + 2;
        uint32_t x_min = x < 2 ? 0u : x - 2;
        uint32_t x_max = x + 2 >= cols ? cols - 1u : x + 2;

        for (uint32_t cy = y_min; cy <= y_max; cy++) {
            for (uint32_t cx = x_min; cx <= x_max; cx++) {
                if (cy == y && cx == x) {
                    printf("\x1b[1m");
                }
                printf("[%6g, %5g]  ", static_cast<float>(d_a[cy*cols + cx]), static_cast<float>(d_b[cy*cols + cx]));
                if (cy == y && cx == x) {
                    printf("\x1b[0m");
                }
            }
            printf("\n");
        }
    }
}

template <typename TA, typename TB>
void launch_device_compare_tensor(TestParams test_params,
                                  const TA* d_a, const TB* d_b, uint32_t rows, uint32_t cols,
                                  unsigned long long* d_bitfield, cudaStream_t stream)
{
    dim3 grid{(cols + 15u) / 16u, (rows + 15u) / 16u, 1};
    dim3 block{16, 16, 1};
    device_compare_tensor_test_init_bitfield<<<1, 1, 0, stream>>>(d_bitfield);
    device_compare_tensor_test<<<grid, block, 0, stream>>>(d_a, d_b, rows, cols, d_bitfield);
    device_compare_tensor_test_print<<<1, 1, 0, stream>>>(test_params, d_a, d_b, rows, cols, d_bitfield);
}

enum class AlgorithmCode
{
    mine_output_stationary = 0,
    cublas = 1,
    cutlass = 2,
    mine_split_k_inner = 3,
    mine_split_k_outer = 4,
    mine_stream_k = 5,
};

inline const char* algorithm_name(AlgorithmCode code)
{
    switch (code) {
      case AlgorithmCode::mine_output_stationary:
        return "mine (output stationary)";
      case AlgorithmCode::cublas:
        return "cublas";
      case AlgorithmCode::cutlass:
        return "cutlass";
      case AlgorithmCode::mine_split_k_inner:
        return "mine (split k inner)";
      case AlgorithmCode::mine_split_k_outer:
        return "mine (split k outer)";
      case AlgorithmCode::mine_stream_k:
        return "mine (stream k)";
    }
    return "unknown";
}

#define CUBLAS_CHECK(x) if (auto _cublas_status = x; _cublas_status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "%s:%i cublas status %i\n", __FILE__, __LINE__, (int)_cublas_status); }

void gemm_test(TestParams params, cudaStream_t stream)
{
    StreamWorkspace stream_ws{stream};

#if CUBLAS_TEST_ENABLED
    cublasHandle_t cublasH;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));
    CUBLAS_CHECK(cublasSetMathMode(cublasH, CUBLAS_TF32_TENSOR_OP_MATH));
#endif

    unsigned long long* d_bitfield = nullptr;
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_bT = nullptr;
    float* d_c_baseline = nullptr;
    float* d_c_warmup = nullptr;
    float* d_c_tested = nullptr;
    using cute::half_t;
    half_t* d_a16 = nullptr;
    half_t* d_bT16 = nullptr;
    half_t* d_c16 = nullptr;

    cudaMallocAsync(&d_bitfield, sizeof(unsigned long long), stream);

    cudaMallocAsync(&d_a,    sizeof(float) * params.M * params.K, stream);
    cudaMallocAsync(&d_a16,  sizeof(float) * params.M * params.K, stream);

    cudaMallocAsync(&d_b,    sizeof(float) * params.N * params.K, stream);
    cudaMallocAsync(&d_bT,   sizeof(float) * params.N * params.K, stream);
    cudaMallocAsync(&d_bT16, sizeof(half_t) * params.N * params.K, stream);

    cudaMallocAsync(&d_c_baseline, sizeof(float) * params.M * params.N, stream);
    cudaMallocAsync(&d_c_warmup,   sizeof(float) * params.M * params.N, stream);
    cudaMallocAsync(&d_c_tested,   sizeof(float) * params.M * params.N, stream);
    cudaMallocAsync(&d_c16,        sizeof(half_t) * params.M * params.N, stream);

    if (!d_bitfield || !d_a || !d_b || !d_c_baseline || !d_c_warmup || !d_c_tested) {
        const cudaError_t err = cudaGetLastError();
        fprintf(stderr, "Error on cudaMallocAsync: %i (%s)\n", (int)err, cudaGetErrorString(err));
        exit(1);
    }

    // Initialize "random" test data
    {
        dim3 grid_a{(params.K + 15u) / 16u, (params.M + 15u) / 16u, 1};
        dim3 grid_b{(params.N + 15u) / 16u, (params.K + 15u) / 16u, 1};
        dim3 block{16, 16, 1};
        device_init_test_data<<<grid_a, block, 0, stream>>>(d_a, params.M, params.K, params.test_data_code_A, false);
        device_init_test_data<<<grid_a, block, 0, stream>>>(d_a16, params.M, params.K, params.test_data_code_A, false);
        device_init_test_data<<<grid_b, block, 0, stream>>>(d_b, params.K, params.N, params.test_data_code_B, false);
        device_init_test_data<<<grid_b, block, 0, stream>>>(d_bT, params.N, params.K, params.test_data_code_B, true);
        device_init_test_data<<<grid_b, block, 0, stream>>>(d_bT16, params.N, params.K, params.test_data_code_B, true);
    }

    auto fill_garbage = [params, stream] (auto* d_c)
    {
        cudaMemsetAsync(d_c, 0x7D, sizeof(*d_c) * params.M * params.N, stream);
    };

    auto run_cublas = [&] (float* d_c)
    {
        assert(CUBLAS_TEST_ENABLED);
#if CUBLAS_TEST_ENABLED
        // Need to do A,B swap trick to deal with Fortran-inherited column majorness
        cublasOperation_t transa = CUBLAS_OP_T;
        cublasOperation_t transb = CUBLAS_OP_N;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(cublasH, transa, transb, int(params.N), int(params.M), int(params.K),
                                 &alpha, d_bT, int(params.K), d_a, int(params.K), &beta, d_c, int(params.N)));
#endif
    };

    // Initialize baseline data
    {
        GPU_Tensors t{params.M, params.N, params.K, d_a, d_b, d_c_baseline, 0, 0, 0};
        fill_garbage(t.c);
        matmul_sm80(t, stream);
    }

    // cute test (need to do A,B swap trick to deal with Fortran-inherited column majorness)
    if (0) {
        const int ldA = int(params.K);
        const int ldB = int(params.K);
        const int ldC = int(params.N);
        cute_gemm::f16('T', 'N', int(params.N), int(params.M), int(params.K),
                       d_bT16, ldB, d_a16, ldA, d_c16, ldC, stream);
        launch_device_compare_tensor(params, d_c_baseline, d_c16, params.M, params.N, d_bitfield, stream);
    }

    // Exo test
    if (0) {
        assert(stream == 0);
        exo_cuda_gemm(nullptr, int(params.M), int(params.N), int(params.K), d_a, d_b, d_c_warmup);
        launch_device_compare_tensor(params, d_c_baseline, d_c_warmup, params.M, params.N, d_bitfield, stream);
    }

    // Initialize SM90 data
    auto sm90_test = [&] (gemm_sm90_k_mode k_mode)
    {
        GPU_Tensors t{params.M, params.N, params.K, d_a, d_bT, d_c_warmup, 0, 1, 0};
        fill_garbage(t.c);
        matmul_sm90(t, k_mode, stream);
        launch_device_compare_tensor(params, d_c_baseline, d_c_warmup, params.M, params.N, d_bitfield, stream);
    };
    if (params.test_k_modes == TestKModes::all || params.test_k_modes == TestKModes::output_stationary) {
        sm90_test(gemm_sm90_k_mode::output_stationary);
    }
    if (params.test_k_modes == TestKModes::all || params.test_k_modes == TestKModes::split_k_outer) {
        sm90_test(gemm_sm90_k_mode::split_k_outer);
    }
    if (params.test_k_modes == TestKModes::all || params.test_k_modes == TestKModes::split_k_inner) {
        sm90_test(gemm_sm90_k_mode::split_k_inner);
    }
    if (params.test_k_modes == TestKModes::all || params.test_k_modes == TestKModes::stream_k) {
        sm90_test(gemm_sm90_k_mode::stream_k);
    }

    // Test loop
    auto run_tests = [&] (AlgorithmCode algo, uint32_t test_count)
    {
        std::vector<float> test_times(test_count);
        std::vector<cudaEvent_t> test_events(test_count + 1);
        auto new_event = [stream]
        {
            cudaEvent_t event{};
            if (const cudaError_t err = cudaEventCreate(&event)) {
                fprintf(stderr, "cudaError_t %i: %s\n", (int)err, cudaGetErrorString(err));
                exit(1);
            }
            cudaEventRecord(event, stream);
            return event;
        };
        for (uint32_t test_i = 0; test_i < test_count; ++test_i) {
            if (test_i == 0) {
                test_events[0] = new_event();
            }
            if (algo == AlgorithmCode::cublas) {
                run_cublas(d_c_tested);
            }
            else {
                GPU_Tensors t{params.M, params.N, params.K, d_a, d_bT, d_c_tested, 0, 1, 0};
                if (algo == AlgorithmCode::mine_output_stationary) {
                    matmul_sm90(t, gemm_sm90_k_mode::output_stationary, stream);
                }
                else if (algo == AlgorithmCode::mine_split_k_inner) {
                    matmul_sm90(t, gemm_sm90_k_mode::split_k_inner, stream);
                }
                else if (algo == AlgorithmCode::mine_split_k_outer) {
                    matmul_sm90(t, gemm_sm90_k_mode::split_k_outer, stream);
                }
                else if (algo == AlgorithmCode::mine_stream_k) {
                    matmul_sm90(t, gemm_sm90_k_mode::stream_k, stream);
                }
                else if (algo == AlgorithmCode::cutlass) {
                    matmul_cutlass(t, stream_ws);
                }
                else {
                    assert(0);
                }
            }
            test_events[test_i + 1] = new_event();
        }
        cudaStreamSynchronize(stream);
        for (uint32_t test_i = 0; test_i < test_count; ++test_i) {
            cudaEventElapsedTime(&test_times[test_i], test_events[test_i], test_events[test_i + 1]);
            cudaEventDestroy(test_events[test_i]);
        }
        cudaEventDestroy(test_events[test_count]);
        std::sort(&test_times[0], &test_times[test_count]);
        double samples_ms = 0.0;
        const int sample_count = (test_count + 1) / 2;
        for (int i = 0; i < sample_count; ++i) {
            samples_ms += test_times[test_count/4 + i];  // We time based on the IQR.
        }
        const double flops = sample_count * double(params.M) * params.N * params.K * 2.0 * 1000.0 / samples_ms;
        const bool bold = double(params.M) * params.N * params.K >= 1e11
                            && params.test_data_code_A == TestDataCode::random
                            && params.test_data_code_B == TestDataCode::random;
        int color_code = 0;
        if (bold) {
            color_code = 31 + int(algo);
        }
        printf("TestParams{%u,%u,%u, %i,%i} %.3g ms %.3g \x1b[%im\x1b[%imTFLOPS\x1b[0m (%s)\n",
               params.M, params.N, params.K,
               static_cast<int>(params.test_data_code_A), static_cast<int>(params.test_data_code_B),
               samples_ms / sample_count, flops * 1e-12, bold, color_code, algorithm_name(algo));
        launch_device_compare_tensor(params, d_c_baseline, d_c_tested, params.M, params.N, d_bitfield, stream);
    };
    const int mine_test_count = 48;
    if (params.test_k_modes == TestKModes::all || params.test_k_modes == TestKModes::output_stationary) {
        run_tests(AlgorithmCode::mine_output_stationary, mine_test_count);
    }
    if (params.test_k_modes == TestKModes::all || params.test_k_modes == TestKModes::split_k_outer) {
        run_tests(AlgorithmCode::mine_split_k_outer, mine_test_count);
    }
    if (params.test_k_modes == TestKModes::all || params.test_k_modes == TestKModes::split_k_inner) {
        run_tests(AlgorithmCode::mine_split_k_inner, mine_test_count);
    }
    if (params.test_k_modes == TestKModes::all || params.test_k_modes == TestKModes::stream_k) {
        run_tests(AlgorithmCode::mine_stream_k, mine_test_count);
    }
#if CUBLAS_TEST_ENABLED
    run_tests(AlgorithmCode::cublas, 16);
#endif
#if CUTLASS_TEST_ENABLED
    run_tests(AlgorithmCode::cutlass, 1);
#endif
    printf("\n");

    cudaFreeAsync(d_a, stream);
    cudaFreeAsync(d_a16, stream);
    cudaFreeAsync(d_b, stream);
    cudaFreeAsync(d_bT, stream);
    cudaFreeAsync(d_bT16, stream);
    cudaFreeAsync(d_c_baseline, stream);
    cudaFreeAsync(d_c_warmup, stream);
    cudaFreeAsync(d_c_tested, stream);
    cudaFreeAsync(d_c16, stream);

    cudaStreamSynchronize(stream);
    if (const cudaError_t err = cudaGetLastError()) {
        fprintf(stderr, "cudaError_t %i: %s\n", (int)err, cudaGetErrorString(err));
        exit(1);
    }

#if CUBLAS_TEST_ENABLED
    cublasDestroy(cublasH);
#endif
}

__global__ void device_synclog_print()
{
    cutlass::arch::synclog_print();
}

}  // end namespace

void gemm_test(TestParams params, cudaStream_t stream)
{
    gemm_test_impl::gemm_test(params, stream);
}

void cutlass_synclog_setup()
{
    cutlass::arch::synclog_setup();
}

void cutlass_synclog_print()
{
    gemm_test_impl::device_synclog_print<<<1, 32, 0, cudaStreamLegacy>>>();
    cudaStreamSynchronize(cudaStreamLegacy);
}
