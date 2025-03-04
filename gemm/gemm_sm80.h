#pragma once

#include "gpu_tensor.h"

void matmul_sm80(GPU_Tensors t, cudaStream_t stream);
bool matmul_sm80_supports(GPU_Tensors t);
