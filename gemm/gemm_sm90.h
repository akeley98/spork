#pragma once

#include "gpu_tensor.h"

enum class gemm_sm90_k_mode
{
    output_stationary,
    split_k,
    stream_k,
};

void matmul_sm90(GPU_Tensors t, gemm_sm90_k_mode k_mode, cudaStream_t stream);

