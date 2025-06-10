from __future__ import annotations
from exo import *
from exo.platforms.x86 import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *

# TeX: version async_block 1
# TeX: begin async_block[0]
@proc
def async_blocks():
    # TeX: remark! *
    # CPU code here
    gmem: f32[1024] @ CudaGmemLinear  # CPU calls cudaMallocAsync
    grid_const: f32 @ CudaGridConstant
    grid_const = 42
    # Launch CUDA kernel with blockDim=256; gridDim={device-specific}
    # Exo implicitly transfers CPU-defined values (gmem, grid_const) as kernel launch parameters.
    # TeX: color line *
    #    gggggggggggggggggg
    with CudaDeviceFunction(blockDim=256):
        # TeX: remark! *
        # In-order CUDA code here
        # TeX: end async_block[0]
        # TeX: summary
        # (omitted parallel-for loops)
        for t in cuda_tasks(0, 1):
            smem: f32[64] @ CudaSmemLinear
            for tid in cuda_threads(0, 2):
                # TeX: begin async_block[0]
                smem[0] = grid_const
                # TeX: color line *
                #    ggggggggg
                with CudaAsync(Sm80_cp_async):
                    # TeX: remark! *
                    # Async CUDA code here
                    Sm80_cp_async_f32(smem[4:8], gmem[4:8], size=4)  # smem[4:8] = gmem[4:8]
                    # TeX: end async_block[0]

# TeX: version intro_tasks_threads 1
# TeX: begin intro_tasks_threads[0]
@proc
def intro_tasks_threads(N: size, X: f32[N] @ CudaGmemLinear, Y: f32[N] @ CudaGmemLinear):
    assert N % 128 == 0
    # TeX: color line *
    #                                yyy
    with CudaDeviceFunction(blockDim=128):
        # Hello world: parallel X += Y vec add.
        # TeX: color line *
        #           gggggggggg
        for task in cuda_tasks(0, N / 128):
            # TeX: color remark! *
            #                              yyy
            # Collective unit here: CTA of 128 threads
            # TeX: color line *
            #          bbbbbbbbbbbb         bbbbbbbbbbbbbbbb
            for tid in cuda_threads(0, 128, unit=cuda_thread):
                # TeX: color remark! *
                #                       bbbbbbbbbbbbb
                # Collective unit here: 1 CUDA thread
                X[128 * task + tid] += Y[128 * task + tid]
            # TeX: color line *
            #                               ggggggggggg  bbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            # Underlying loop mode objects: CudaTasks(), CudaThreads(unit=cuda_thread)
# TeX: end intro_tasks_threads[0]
