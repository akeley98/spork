#pragma once
#include <cuda_bf16.h>

struct HostAttnArgs;
void launch_tk_4090_fwd_kernel(const HostAttnArgs& args, __nv_bfloat16* o);

extern "C" void htod_f32_tk_4090_fwd(int N, int d, const float* q, const float* k, const float* v, float* o);
