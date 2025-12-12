from __future__ import annotations
from exo import *
from exo.platforms.cuda import *

M = 13
N = 37

# TeX: version loop_modes 1
if False:
    # TeX: begin loop_modes[0]
    for _ in seq(_, _, pragma_unroll=_): _
    for _ in par(_, _): _
    for _ in cuda_tasks(_, _): _
    for _ in cuda_threads(_, _, unit=_): _
    # TeX: end loop_modes[0]

# TeX: version OverviewThreads 1
# TeX: begin OverviewThreads[0]
@proc
def overview_threads_example(num_tasks: size):
    # CPU scope here.
    for i in seq(0, 100):
    # TeX: color line *
      # ....
        pass  # Still a typical CPU loop, 100 iterations
    with CudaDeviceFunction(clusterDim=1, blockDim=256):
        # CUDA scope inside this CudaDeviceFunction block.
        for task_id in cuda_tasks(0, num_tasks):
            # CTA-scope here, also cluster-scope because clusterDim=1.
            for w in cuda_threads(0, 8, unit=cuda_warp):
                # Warp-scope here.
                # blockDim=256 threads subdivided into 8 separate warps indexed by w.
                for t in cuda_threads(0, 32, unit=cuda_thread):
                    # Thread-scope here.
                    for s in seq(0, 100):
                      # TeX: color line *
                      # ....
                        pass  # Each thread does something 100 times.
@proc
def overview_distributed_memory_example(num_tasks: size):
    with CudaDeviceFunction(clusterDim=1, blockDim=256):
        for task_id in cuda_tasks(0, num_tasks):
            tile: f32[8, 32, 8, 8] @ CudaRmem  # Registers must be sharded per-thread.
            for w in cuda_threads(0, 8, unit=cuda_warp):  # Thread pitch = 32
                for t in cuda_threads(0, 32, unit=cuda_thread):  # Thread pitch = 1
                    for y in seq(0, 8):
                        for x in seq(0, 8):
                            # Deduction: tile[w, t, _, _] is stored on thread $32w + 1t$ of CTA.
                            tile[w, t, y, x] = 0
# TeX: end OverviewThreads[0]

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
    set_loop_mode(proc, loop_cursor, CudaTasks)  # sugar for CudaTasks()
    # TeX: end CudaDeviceFunction_scheduling[0]


# TeX: version OverviewSyncExample 1
# TeX: begin OverviewSyncExample[0]
@proc
def overview_sync_example(num_tasks: size):
    with CudaDeviceFunction(clusterDim=2, blockDim=384):
        for task_id in cuda_tasks(0, num_tasks):
            # Cluster scope.
            mbar: barrier[2] @ CudaMbarrier  # mbarrier variable.
            for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                # CTA scope.
                # cuda_in_order is a SyncTL,
                # indicates only non-async instrs' effects are synchronized.
                Fence(cuda_in_order, cuda_in_order)  # __syncthreads-equivalent
                for w in cuda_threads(0, 12, unit=cuda_warp):
                    # Warp scope.
                    Fence(cuda_in_order, cuda_in_order)  # __syncwarp-equivalent
                # Distributed memory: each element of mbar[...] allocated into its own CTA.
                # Example of Arrive/Await using mbarrier mechanism
                # (because mbar was annotated with @CudaMbarrier).
                Arrive(cuda_in_order) >> mbar[cta]
                # ...
                Await(mbar[cta], cuda_in_order, ~0)
                # Synchronization chapter explains ~0 (n).
                # TeX: end OverviewSyncExample[0]


# TeX: version for_CollTiling_figure 1
@proc
def coll_tiling_example(num_tasks: size):
    # TeX: begin for_CollTiling_figure[0]
    with CudaDeviceFunction(clusterDim=8, warp_config=[  # blockDim = $384$ = $32\times(1+3+8)$
            CudaWarpConfig("producer", 1, setmaxnreg_dec=40),
            CudaWarpConfig("unused", 3, setmaxnreg_dec=40),
            # TeX: color line *
            #               rrrrrrrr
            CudaWarpConfig("consumer", 8, setmaxnreg_inc=232), # prefix = 4 warps (128 threads)
    ]):
        for task_id in cuda_tasks(0, num_tasks):
            # TeX: color line *
            #   vvvvv                            vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            for n_cta in cuda_threads(0, 2, unit=4 * cuda_cta_in_cluster_strided(2)):
                # n_cta=0 implies CTA 0, 2, 4, 6 used
                # n_cta=1 implies CTA 1, 3, 5, 7 used
                # TeX: color line *
                #   ggggg                            ggggggggggggggggggg
                for m_cta in cuda_threads(0, 4, unit=cuda_cta_in_cluster):
                    # TeX: color line *
                    #                    rrrrrrrr            bbbbbbbbbbbbbbbbbbbbbbbbbbbb
                    with CudaWarps(name="consumer"):  # iter=CudaWarps_consumer_None_None
                        # TeX: color line *
                        #   bb                            bbbbbbbbbbbbbb
                        for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                            # TeX: color line *
                            #   b                              bbbbbbbbbbb
                            for t in cuda_threads(0, 128, unit=cuda_thread):
                                # Illustrating collective tiling $\omega: \Omega$ annotating this body statement:
                                # TeX: color line *
                                #   .
                                for s in seq(0, 6):
                                    # TeX: end for_CollTiling_figure[0]
                                    pass
