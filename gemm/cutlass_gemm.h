#pragma once

#include "gpu_tensor.h"

void matmul_cutlass(GPU_Tensors t, StreamWorkspace& stream_ws);
bool cutlass_enabled();
