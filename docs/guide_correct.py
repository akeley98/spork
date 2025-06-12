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

# TeX: version my_warp_config 1
# TeX: begin my_warp_config[0]
my_warp_config = [
    CudaWarpConfig("producer", 1, setmaxnreg_dec=40),  # 1 warp; reduce to 40 regs
    CudaWarpConfig("unused", 3, setmaxnreg_dec=40),  # 3 warps; reduce to 40 regs
    CudaWarpConfig("consumer", 8, setmaxnreg_inc=232),  # 8 warps; increase to 232 regs
]  # Total: 12 warps: blockDim=384

@proc
def warp_config_example():
    with CudaDeviceFunction(warp_config=my_warp_config):
        for task in cuda_tasks(0, xyzzy):
            smem: f32[3, 128, 256] @ CudaSmemLinear  # Common code executed by all warps
            with CudaWarps(name="producer"):
                # The string ``producer'' has no meaning to Exo,
                # but it's less confusing if you put producer code here.
                # TeX: end my_warp_config[0]
                for tid in cuda_threads(0, 1):
                    smem[0,0,0] = 0  # compiler warning fix
            # TeX: begin my_warp_config[0]
            with CudaWarps(name="consumer"):
                # The string ``consumer'' has no meaning to Exo,
                # but it's less confusing if you put consumer code here.
                # TeX: end my_warp_config[0]
                pass

# TeX: version warpgroup_CudaWarps 1
# TeX: begin warpgroup_CudaWarps[0]
@proc
def warpgroup_CudaWarps():
    with CudaDeviceFunction(blockDim=256):
        for task in cuda_tasks(0, xyzzy):
            for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                # Collective unit here is 1 warpgroup (4 warps)
                with CudaWarps(3, 4):
                    # Collective unit here is 1 warp
                    # TeX: end warpgroup_CudaWarps[0]
                    pass

@proc
def simple_dist():
    # TeX: version simple_dist 2
    # TeX: begin simple_dist
    with CudaDeviceFunction(blockDim=512):
        for task in cuda_tasks(0, xyzzy):
            # TeX: color remark simple_dist[0]
            # rrrrrrrrrrrrrrrrrrrrr  yyyyyyyyyyyyyyyyyyyyyyy
            # Distributed (16 x 32); non-distributed (8 x 4)
            # TeX: color line *
            #         rrrrrr  yyyy                                      yyyyy
            vals: f32[16, 32, 8, 4] @ CudaRmem  # Each thread allocates 8 x 4 registers
            # TeX: remark simple_dist[1]
            # Tile here: (512,); $t_a = 512$; $t_n = 1$ (native unit cuda_thread for CudaRmem).
            # TeX: color line *
            #   g                                                   gggggggggggggggggg
            for m in cuda_threads(0, 16, unit=32 * cuda_thread):  # $m: 512\mapsto 32$
                # TeX: remark simple_dist[1]
                # Tile here: (32,)
                # TeX: color line *
                #   v                                               vvvvvvvvvvvvvvvv
                for n in cuda_threads(0, 32, unit=cuda_thread):   # $n: 32\mapsto 1$
                    # TeX: remark simple_dist[1]
                    # Tile here: (1,)
                    # TeX: color line simple_dist
                    #    g  v  yyyy
                    vals[m, n, 0, 0] = 0
                    # TeX: color line simple_dist
                    #    g  v  yyyy
                    vals[m, n, 0, 1] = 0
                    # TeX: remark simple_dist[0]
                    # ...
                    # TeX: color remark simple_dist[1]
                    #               ggggggggggggggggggg  vvvvvvvvvvvvvvvvv
                    # Tiling chain: $m: 512 \mapsto 32$, $n: 32 \mapsto 1$
                    # TeX: color remark simple_dist[1]
                    #                       yyyyyyyyyyyyyyy
                    # Remaining indices are non-distributed
    # TeX: end simple_dist

"""
# TeX: version simple_dist_cxx 1
# TeX: begin simple_dist_cxx[0]
# TeX: color line *
#          yyyyy                                rrrrrrr
float vals[8 * 4];  # Distributed across CTA of 16 x 32 threads
# TeX: color line *
#                        ggggggggggg                               ggggggggggggggggggg
if ([[maybe_unused]] int exo_32thr_m = (threadIdx.x / 32); 1) {  # $m: 512 \mapsto 32$
# TeX: color line *
#                          vvvvvvvvvv                               vvvvvvvvvvvvvvvvv
  if ([[maybe_unused]] int exo_1thr_n = (threadIdx.x % 32); 1) {  # $n: 32 \mapsto 1$
    vals[0] = 0.0f;
    vals[1] = 0.0f;  # [m, n] (distributed indices) removed
    # TeX: end simple_dist_cxx[0]
  }
}
"""
