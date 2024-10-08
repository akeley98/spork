#include "attn_main.hpp"

#include <memory>
#include <stdio.h>

#include "matrix_util.hpp"
#include "baseline.hpp"
#include "tk_4090_fwd.hpp"

int main()
{
    cudaStream_t stream{0};
    HostAttnArgs args = make_attn_args(stream, testbed_N, testbed_d, 1337);
    auto tk_o = make_unique_cuda_array<__nv_bfloat16>(testbed_N, testbed_d);
    auto baseline_o = make_unique_cuda_array<__nv_bfloat16>(testbed_N, testbed_d);
    launch_tk_4090_fwd_kernel(args, tk_o.get());
    launch_baseline_kernel(args, baseline_o.get());

    const size_t element_count = args.N * args.d;
    std::unique_ptr<__nv_bfloat16[]> p_host_output(new __nv_bfloat16[element_count]);
    cudaMemcpy(p_host_output.get(), tk_o.get(), element_count * sizeof p_host_output[0], cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i) {
        printf("%i, 72: %g\n", i, __bfloat162float(p_host_output[i + 72*args.d]));
    }
    printf("\n");
    cudaMemcpy(p_host_output.get(), baseline_o.get(), element_count * sizeof p_host_output[0], cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i) {
        printf("%i, 72: %g\n", i, __bfloat162float(p_host_output[i + 72*args.d]));
    }
}
