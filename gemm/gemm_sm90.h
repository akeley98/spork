#pragma once

#include "gpu_tensor.h"

enum class gemm_sm90_k_mode
{
    output_stationary,
    split_k_outer,      // Split-k, with k being the slow dimension for scheduling
    split_k_inner,      // Split-k, with k being the fast dimension for scheduling [less efficient L2?]
    stream_k,
};

void matmul_sm90(GPU_Tensors t, gemm_sm90_k_mode k_mode, cudaStream_t stream);

