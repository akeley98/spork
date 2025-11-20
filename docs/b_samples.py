from __future__ import annotations
from exo import *
from exo.platforms.cuda import *

M = 13
N = 37

# TeX: version CudaDeviceFunction 1
@proc
def cuda_device_function_example():
    # TeX: begin CudaDeviceFunction[0]
    # Each cluster has 2 CTAs, 384 threads (12 warps).
    # The first 4 warps have 40 registers per thread, the last 8 have 232.
    with CudaDeviceFunction(clusterDim=2, warp_config=[
            CudaWarpConfig("producer", 1, setmaxnreg_dec=40),  # 1 producer warp
            CudaWarpConfig("unused", 3, setmaxnreg_dec=40),    # 3 unused warps
            CudaWarpConfig("consumer", 8, setmaxnreg_inc=232), # 8 consumer warpgroups
            # Note, the names don't have any built-in meaning to Exo.
    ]):
        for m in cuda_tasks(0, M):
            for n in cuda_tasks(0, N):
                # Device task. cuda_threads loops may appear here.
                with CudaWarps(name="producer"):
                    # TeX: color line *
                   #....
                    pass
                with CudaWarps(name="consumer"):
                    # TeX: color line *
                   #....
                    pass
                # TeX: end CudaDeviceFunction[0]

if False:
    # TeX: version CudaDeviceFunction_scheduling 1
    # TeX: begin CudaDeviceFunction_scheduling[0]
    # TeX: color line *
    #                                                        ............
    wrap_with_context(proc, block_cursor, CudaDeviceFunction(blockDim=128))
    # TeX: color line *
    #                                               ....
    wrap_with_context(proc, block_cursor, CudaWarps(0, 1))
    set_loop_mode(proc, loop_cursor, CudaTasks)
    # TeX: end CudaDeviceFunction_scheduling[0]
