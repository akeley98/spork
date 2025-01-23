#include "gemm_test.h"

int main()
{
    cutlass_synclog_setup();

    auto sized_tests = [] (uint32_t M, uint32_t N, uint32_t K, bool nontrivial_only)
    {
        TestParams params{};
        params.M = M;
        params.N = N;
        params.K = K;
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
    sized_tests(32768, 65536, 48, true);
    sized_tests(32768, 40016, 816, false);
    sized_tests(40016, 32768, 6400, true);
    sized_tests(12288, 24576, 1536, true);
    sized_tests(720, 1440, 65536, true);
    sized_tests(1440, 1440, 65536, true);
    sized_tests(6 * 256, 3 * 256, 65536, true);
    sized_tests(6 * 256, 10 * 256, 65536, true);
    sized_tests(6 * 256, 11 * 256, 65536, true);  // 66 2-SM clusters on H100 SXM5
    sized_tests(6 * 256, 12 * 256, 65536, true);
    sized_tests(3 * 256, 11 * 256, 65536, true);
    sized_tests(9 * 256, 11 * 256, 65536, true);
    sized_tests(4 * 256, 11 * 256, 16384 * 3, true);

    cudaDeviceSynchronize();
    cutlass_synclog_print();
}
