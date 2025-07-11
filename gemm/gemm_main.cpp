#include "gemm_test.h"
#include "gemm_sm80.h"

#include <cuda_runtime.h>
#include <stdio.h>

int main()
{
    cudaSetDevice(0);

    int cuda_cc_major{}, cuda_cc_minor{};
    cudaDeviceGetAttribute(&cuda_cc_major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&cuda_cc_minor, cudaDevAttrComputeCapabilityMinor, 0);
    const bool is_h100 = cuda_cc_major == 9 && cuda_cc_minor == 0;
    fprintf(stderr, "is_h100: %i\n", is_h100);

    auto sized_tests = [is_h100] (uint32_t M, uint32_t N, uint32_t K, int data_style)
    {
        TestParams params{};
        params.M = M;
        params.N = N;
        params.K = K;

        params.algorithm_code_bits = 0;
        params.algorithm_code_bits |= algorithm_code_bit(AlgorithmCode::cublas);
        if (is_h100) {
            params.algorithm_code_bits |= algorithm_code_bit(AlgorithmCode::cutlass);
            params.algorithm_code_bits |= algorithm_code_bit(AlgorithmCode::mine_output_stationary);
            params.algorithm_code_bits |= algorithm_code_bit(AlgorithmCode::mine_split_k_inner);
            params.algorithm_code_bits |= algorithm_code_bit(AlgorithmCode::mine_split_k_outer);
            // params.algorithm_code_bits |= algorithm_code_bit(AlgorithmCode::mine_stream_k_early_tma);
            params.algorithm_code_bits |= algorithm_code_bit(AlgorithmCode::mine_stream_k_late_tma);
        }
        {
            GPU_Tensors t{};
            t.M = M;
            t.N = N;
            t.K = K;
            if (matmul_sm80_supports(t)) {
                params.algorithm_code_bits |= algorithm_code_bit(AlgorithmCode::mine_sm_80);
            }
        }
        // TODO lift Exo restrictions
        if (M % 768 == 0 && N % 768 == 0 && K % 32 == 0) {
            params.algorithm_code_bits |= algorithm_code_bit(AlgorithmCode::exo_sm_80_fence);
            params.algorithm_code_bits |= algorithm_code_bit(AlgorithmCode::exo_sm_80_mbarrier);
            if (is_h100) {
                params.algorithm_code_bits |= algorithm_code_bit(AlgorithmCode::exo_sm_90_n256);
                params.algorithm_code_bits |= algorithm_code_bit(AlgorithmCode::exo_sm_90_n128);
                params.algorithm_code_bits |= algorithm_code_bit(AlgorithmCode::edited_exo_sm_90);
            }
        }

        if (data_style == 0 || data_style == 1) {
            params.test_data_code_A = TestDataCode::tiled_numbers;
            params.test_data_code_B = TestDataCode::identity;
            gemm_test(params, {});
        }
        if (data_style == 0 || data_style == 2) {
            params.test_data_code_A = TestDataCode::identity;
            params.test_data_code_B = TestDataCode::tiled_numbers;
            gemm_test(params, {});
        }
        if (data_style == 0 || data_style == 3) {
            params.test_data_code_A = TestDataCode::random;
            params.test_data_code_B = TestDataCode::random;
            gemm_test(params, {});
        }
    };

    sized_tests(768, 768, 32, 2);
    sized_tests(7680, 1536, 4096, 0);
    sized_tests(15360, 1536, 8192, 3);
    sized_tests(6 * 256, 9 * 256, 65536, 3);
    if (is_h100) {
        sized_tests(160, 480, 816, 0);
        sized_tests(32768, 65536, 48, 3);
        sized_tests(32768, 40016, 816, 0);
        sized_tests(40016, 32768, 6400, 3);
        sized_tests(12288, 24576, 1536, 3);
        sized_tests(6 * 256, 3 * 256, 65536, 3);
        sized_tests(6 * 256, 10 * 256, 65536, 3);
        sized_tests(6 * 256, 11 * 256, 65536, 3);  // 66 2-SM clusters on H100 SXM5
        sized_tests(6 * 256, 12 * 256, 65536, 3);
        sized_tests(3 * 256, 11 * 256, 65536, 3);
        sized_tests(9 * 256, 11 * 256, 65536, 3);
        sized_tests(4 * 256, 11 * 256, 16384 * 3, 3);
    }
    else {
        sized_tests(7680, 3072, 16384, 3);
        sized_tests(768, 3072, 3072, 3);
    }

    cudaDeviceSynchronize();
}
