#pragma once

#include "gpu_tensor.h"

void matmul_cutlass(GPU_Tensors t, cudaStream_t stream);
