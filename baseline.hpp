#include <cuda_bf16.h>

struct HostAttnArgs;

void launch_baseline_kernel(const HostAttnArgs& args, __nv_bfloat16* o);
