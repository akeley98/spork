from __future__ import annotations
from exo import *
from exo.platforms.x86 import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *

xyzzy = 1

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
        for t in cuda_tasks(0, xyzzy):  # See ``Loop Mode''
            smem: f32[64] @ CudaSmemLinear  # Shared memory allocation
            for tid in cuda_threads(0, 256):  # See ``Loop Mode''
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

# TeX: version cuda_threads_cxx 1
@proc
def cuda_threads_cxx():
    with CudaDeviceFunction(blockDim=256):
        for task in cuda_tasks(0, xyzzy):
            # TeX: begin cuda_threads_cxx[0]
            # Exo
            # TeX: color line *
            #                       g          vv
            # Break threads into an 8 $\times$ 16 iteration space
            # TeX: color line *
            #   g                    g       bbbbbbbbbbbbbbbb
            for m in cuda_threads(0, 8, unit=16 * cuda_thread):
                # TeX: color line *
                #   v                    vv
                for n in cuda_threads(0, 16, unit=cuda_thread):
                    # TeX: end cuda_threads_cxx[0]
                    pass
"""
            # TeX: begin cuda_threads_cxx[0]
            # C++
            # TeX: color line *
            #       ggggggggggg                  bb                 g
            if (int exo_16thr_m = (threadIdx.x / 16); exo_16thr_m < 8) {
              # TeX: color line *
              #       vvvvvvvvvv
              if (int exo_1thr_n = (threadIdx.x % 16); 1) {
            # TeX: end cuda_threads_cxx[0]
"""
