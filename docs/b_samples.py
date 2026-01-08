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
# TeX: end OverviewThreads[0]

# TeX: version OverviewDistributedMemory 1
@proc
def overview_distributed_memory_example(num_tasks: size):
    # TeX: begin OverviewDistributedMemory[0]
    with CudaDeviceFunction(clusterDim=1, blockDim=256):
        for task_id in cuda_tasks(0, num_tasks):
            # TeX: color line *
            #                b  bb
            tile: f32[8, 32, 8, 16] @ CudaRmem  # Registers must be sharded per-thread.
            for w in cuda_threads(0, 8, unit=cuda_warp):  # Thread pitch = 32
                for t in cuda_threads(0, 32, unit=cuda_thread):  # Thread pitch = 1
                    for y in seq(0, 8):
                        for x in seq(0, 16):
                            # Deduction: tile[w, t, _, _] is stored on thread $32w + 1t$ of CTA.
                            # TeX: color line *
                            #                        b          bb
                            # Each register holds an 8 $\times$ 16 tile.
                            tile[w, t, y, x] = 0
# TeX: end OverviewDistributedMemory[0]

if False:
    # TeX: version why_dist 1
    # TeX: begin why_dist[0]
    # TeX: remark! *
    # Hypothetical (broken) alternative: registers implicitly duplicated per-thread.
    # TeX: color line *
    #                                                rrrrrrrrrrrrrrrrrrr
    x: f32 @ CudaRmem  # implicitly one x per thread (not valid Exo-GPU)
    for tid in cuda_threads(0, 2, unit=cuda_thread):
        x = a[tid]  # x = a[0] in thread 0; x = a[1] in thread 1
    for tid in cuda_threads(0, 2, unit=cuda_thread):
        # TeX: color line *
        #             ggggggggggg
        b[tid] = x  # b[0] = a[0]; b[1] = a[1]

    # TeX: remark! *
    # Same program interpreted sequentially (gives different result).
    x: f32 @ CudaRmem  # Shared state for both tid-loop iterations
    for tid in cuda_threads(0, 2, unit=cuda_thread):
        x = a[tid]  # x = a[1], overwrites x = a[0]
    for tid in cuda_threads(0, 2, unit=cuda_thread):
        # TeX: color line *
        #             rrrrrrrrrrr
        b[tid] = x  # b[0] = a[1]; b[1] = a[1]

    # TeX: remark! *
    # Correct Exo-GPU
    x[2]: f32 @ CudaRmem  # We explicitly model there are 2 x's, distributed into 2 threads
    for tid in cuda_threads(0, 2, unit=cuda_thread):
        x[tid] = a[tid]  # x[0] = a[0]; x[1] = a[1]
    for tid in cuda_threads(0, 2, unit=cuda_thread):
        b[tid] = x[tid]  # b[0] = a[0]; b[1] = a[1]
    # TeX: end why_dist[0]


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
@proc
def overview_sync_example(num_tasks: size):
    # TeX: begin OverviewSyncExample[0]
    with CudaDeviceFunction(clusterDim=2, blockDim=384):
        for task_id in cuda_tasks(0, num_tasks):
            # Cluster scope.
            # TeX: color line *
            #                                                                                  .
            # Distributed memory (Section $\ref{sec:DistributedMemory}$): each element of mbar[.] allocated into its own CTA.
            mbar: barrier[2] @ CudaMbarrier
            for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                # CTA scope.
                # cuda_in_order is a SyncTL,
                # indicates only non-async instrs' effects are synchronized.
                Fence(cuda_in_order, cuda_in_order)  # __syncthreads-equivalent
                for w in cuda_threads(0, 12, unit=cuda_warp):
                    # Warp scope.
                    Fence(cuda_in_order, cuda_in_order)  # __syncwarp-equivalent
                # Example of Arrive/Await using mbarrier mechanism
                # (because mbar was annotated with @CudaMbarrier, Section $\ref{sec:MbarrierUsage}$).
                Arrive(cuda_in_order) >> mbar[cta]
                # ...
                Await(mbar[cta], cuda_in_order, ~0)
                # Section $\ref{sec:ArriveAwaitPairing}$ explains n=~0.
                # TeX: end OverviewSyncExample[0]


# TeX: version OverviewCollTiling 1
@proc
def overview_collective_tiling_example(num_tasks: size):
    # TeX: begin OverviewCollTiling[0]
    with CudaDeviceFunction(clusterDim=2, blockDim=384):  # 2 CTAs per cluster (def $\ref{sec:gCluster}$)
        for task_id in cuda_tasks(0, num_tasks):
            # Cluster scope.
            # 384 threads per CTA (def $\ref{sec:gCta}$). Local thread indices (def $\ref{sec:gLocalThreadIndex}$) of threads in
            # CTA 0 of the cluster are $[0, 383]_\mathbb{N}$, CTA 1 are $[384, 765]_\mathbb{N}$.
            for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                # CTA scope.
                # Local thread indices used to execute are given by $[384\sigma(\texttt{cta})$, $384\sigma(\texttt{cta}) + 383]_\mathbb{N}$
                # where $\sigma(y)$ gives the value of a variable $y$ from the control environment (def $\ref{sec:gControlEnv}$).
                # TeX: color line *
                #     ............................
                Fence(cuda_in_order, cuda_in_order)  # __syncthreads-equivalent, syncs across 384 threads named above.
                for w in cuda_threads(0, 12, unit=cuda_warp):
                    # Warp scope. Local thread indices used to execute are given by
                    # $[384\sigma(\texttt{cta}) + 32\sigma(\texttt{w})$, $384\sigma(\texttt{cta}) + 32\sigma(\texttt{w}) + 31]_\mathbb{N}$
                    # TeX: color line *
                    #     ............................
                    Fence(cuda_in_order, cuda_in_order)  # __syncwarp-equivalent, syncs across 32 threads named above.
# TeX: end OverviewCollTiling[0]


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


if False:
    @proc
    def bad_multicast_example():
        with CudaDeviceFunction(clusterDim=4, blockDim=32):
            for task_id in cuda_tasks(0, 1):
                # TeX: version bad_multicast_example 1
                # TeX: begin bad_multicast_example[0]
                z: barrier[2, 2] @ CudaMbarrier
                for m in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster):
                    # TeX: color line *
                    #  ......
                    if m == 0:  # Valid if statement (m is not multicast)
                        # TeX: color line *
                        #   v
                        for n in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                            # TeX: color line *
                            #  ......
                            if n == 0:  # Invalid if statement underneath n loop
                                # TeX: color line *
                                #                             v          v
                                Arrive(cuda_in_order) >> z[m, n] >> z[m, :]  # n is multicast
                                # TeX: end bad_multicast_example[0]
                                Await(z[m, n], cuda_in_order, ~0)
    # TeX: version multicast_flags_example 1
    # TeX: begin multicast_flags_example[0]
    Arrive(cuda_in_order) >> z[m, n] >> z[m, :]   # (False, False), (False, True)
    Arrive(cuda_in_order) >> z[:, n] >> z[m, :]   # (True, False), (False, True)
    Arrive(cuda_in_order) >> z[m, n, k]           # (False, False, False),
    # TeX: end multicast_flags_example[0]


@proc
def mbarrier_2_cycle(num_tasks: size):
    with CudaDeviceFunction(warp_config=[  # blockDim = $384$ = $32\times(1+3+8)$
            CudaWarpConfig("producer", 1, setmaxnreg_dec=40),
            CudaWarpConfig("unused", 3, setmaxnreg_dec=40),
            # TeX: color line *
            #               rrrrrrrr
            CudaWarpConfig("consumer", 8, setmaxnreg_inc=232), # prefix = 4 warps (128 threads)
    ]):
        for task_id in cuda_tasks(0, num_tasks):
            # TeX: version mbarrier_2_cycle 1
            # TeX: begin mbarrier_2_cycle[0]
        # TeX: color line *
        #   bb                                                          rr
            z0: barrier @ CudaMbarrier          # Implicitly guarded-by z1
        # TeX: color line *
        #   rr          bb                                              bb
            z1: barrier(z0) @ CudaMbarrier      # Explicitly guarded-by z0
            with CudaWarps(name="producer"):
                # TeX: color line *
                #     rr  .............
                Await(z1, cuda_in_order, ~4)
                # ...instr calls with trailing barrier expressions involving z0 may appear here
                # TeX: color line *
                #      .............     bb
                Arrive(cuda_in_order) >> z0
            with CudaWarps(name="consumer"):
                # TeX: color line *
                #     bb  .............
                Await(z0, cuda_in_order, ~0)
                # ...instr calls with trailing barrier expressions involving z1 may appear here
                # TeX: color line *
                #      .............     rr
                Arrive(cuda_in_order) >> z1
            # TeX: end mbarrier_2_cycle[0]


if False:
  # TeX: version multicast_pseudocode 1
  # TeX: begin multicast_pseudocode[0]
  for cta in cuda_threads(0, ncta, unit=cuda_cta_in_cluster):
    for i0 in seq(0, size0):
      for i1 in seq(0, size1):
        # ...
        # TeX: color line *
        #                 ..                 ..
        smem[cta, i0, i1, i2] = gmem[i0, i1, i2]
  # TeX: end multicast_pseudocode[0]

  # TeX: version multicast_tma_excerpt 1
  # TeX: begin multicast_tma_excerpt[0]
  # Defined outside @proc (i.e. this is executed Python code)
  # divided by ncta_M because ncta=ncta_M below.
  smem_box_B = (1, smem_N // ncta_M, 1, smem_K)
  smem_K = 32
  # Defined inside @proc (i.e. this is parsed Exo code) at CPU scope (def $\ref{sec:gCpuScope}$)
  # TeX: color line *
  #                                         yyy
  B_tensorMap = B[:,:,:,:] @ Sm90_tensorMap(128, *smem_box_B)
  # Defined inside @proc at CUDA scope (def $\ref{sec:gCudaScope}$)
  # TeX: color line *
  #            gggggg  vvvvvv        rrrrrr  bbbbbb                      yyy
  B_smem : f32[ncta_M, ncta_N, RING, smem_N, smem_K] @ Sm90_SmemSwizzled(128)
  # length-2 barrier guard cycle {raw, war} (def $\ref{sec:gBarrierGuardCycle}$)
  raw : barrier[ncta_M, ncta_N] @ CudaMbarrier
  war : barrier(raw)[ncta_M, ncta_N] @ CudaMbarrier
  # TeX: color line *
  #                   ..........
  with CudaWarps(name="producer"):  # Not shown: referenced warp variable (def $\ref{sec:gWarpVariable}$) is 1 warp
    for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
      for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
        Await(war[cta_m,cta_n], cuda_temporal, ~(RING-1))
    # ...
    for cta_n in cuda_threads(0, ncta_N,
    # TeX: color line *
    #        gggggg                               vvvvvv
        unit=ncta_M * cuda_cta_in_cluster_strided(ncta_N)
    ):
      # TeX: color line *
      #                                                                           yyyyyyyyyyy
      # Rightmost extent is smem_K=32, times 4 bytes per element (f32) $\implies$ swizzle=128
      Sm90_multicast_copy_tensor_to_smem_swizzled_2f32(
        # TeX: color line *
        #      g                     r b                       gggggg  rrrrrr  bbbbbb
        B_smem[:,cta_n,iter_k % RING,:,:],  # Window extents:  ncta_M, smem_N, smem_K
        B_tensorMap[                                     # smem_box coordinates required:
          batch,                                                            # 1 (point expr)
          (ncta_N*task_n+cta_n) * smem_N: (ncta_N*task_n+cta_n+1) * smem_N, # smem_N // ncta_M
          task_k,                                                           # 1 (point expr)
          iter_k * smem_K: iter_k * smem_K + smem_K],                       # smem_K
        # TeX: color line *
        #    gggggg             vvvvvv        rrrrrr        bbbbbb
        ncta=ncta_M, cta_stride=ncta_N, size0=smem_N, size1=smem_K, smem_box=smem_box_B
      ) >> raw[:,cta_n]
      for cta_m in cuda_threads(0, ncta_M, unit=cuda_cta_in_cluster):
        # Await/TMA/Arrive structure satisfies guarding requirement (Section $\ref{sec:BarrierGuarding}$)
        Arrive(cuda_temporal) >> raw[cta_m,:] >> raw[:,cta_n]
    # TeX: end multicast_tma_excerpt[0]
