#include "gemm_test.h"

#include <cuda_runtime.h>
#include <stdio.h>

int main()
{
    cudaSetDevice(0);
    cutlass_synclog_setup();

    int cuda_cc_major{}, cuda_cc_minor{};
    cudaDeviceGetAttribute(&cuda_cc_major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&cuda_cc_minor, cudaDevAttrComputeCapabilityMinor, 0);
    const bool is_h100 = cuda_cc_major == 9 && cuda_cc_minor == 0;
    fprintf(stderr, "is_h100: %i\n", is_h100);

    auto sized_tests = [is_h100] (uint32_t M, uint32_t N, uint32_t K, bool nontrivial_only)
    {
        TestParams params{};
        params.M = M;
        params.N = N;
        params.K = K;

        params.algorithm_code_bits = 0;
        params.algorithm_code_bits |= algorithm_code_bit(AlgorithmCode::cublas);
        params.algorithm_code_bits |= algorithm_code_bit(AlgorithmCode::exo);
        if (is_h100) {
            params.algorithm_code_bits |= algorithm_code_bit(AlgorithmCode::cutlass);
            params.algorithm_code_bits |= algorithm_code_bit(AlgorithmCode::mine_output_stationary);
            params.algorithm_code_bits |= algorithm_code_bit(AlgorithmCode::mine_split_k_inner);
            params.algorithm_code_bits |= algorithm_code_bit(AlgorithmCode::mine_split_k_outer);
            // params.algorithm_code_bits |= algorithm_code_bit(AlgorithmCode::mine_stream_k_early_tma);
            params.algorithm_code_bits |= algorithm_code_bit(AlgorithmCode::mine_stream_k_late_tma);
        }

        if (!nontrivial_only) {
            params.test_data_code_A = TestDataCode::tiled_numbers;
            params.test_data_code_B = TestDataCode::identity;
            gemm_test(params, {});
        }
        if (!nontrivial_only) {
            params.test_data_code_A = TestDataCode::identity;
            params.test_data_code_B = TestDataCode::tiled_numbers;
            gemm_test(params, {});
        }
        if (true) {
            params.test_data_code_A = TestDataCode::random;
            params.test_data_code_B = TestDataCode::random;
            gemm_test(params, {});
        }
    };

    sized_tests(160, 480, 816, false);
    sized_tests(720, 1440, 65536, true);
    sized_tests(1440, 1440, 65536, true);
    if (is_h100) {
        sized_tests(32768, 65536, 48, true);
        sized_tests(32768, 40016, 816, false);
        sized_tests(40016, 32768, 6400, true);
        sized_tests(12288, 24576, 1536, true);
        sized_tests(6 * 256, 3 * 256, 65536, true);
        sized_tests(6 * 256, 10 * 256, 65536, true);
        sized_tests(6 * 256, 11 * 256, 65536, true);  // 66 2-SM clusters on H100 SXM5
        sized_tests(6 * 256, 12 * 256, 65536, true);
        sized_tests(3 * 256, 11 * 256, 65536, true);
        sized_tests(9 * 256, 11 * 256, 65536, true);
        sized_tests(4 * 256, 11 * 256, 16384 * 3, true);
    }

    cudaDeviceSynchronize();
    cutlass_synclog_print();
}
