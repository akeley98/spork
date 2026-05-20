from __future__ import annotations

from exo import *
from exo.platforms.cuda import *

@proc
def overview_threads_example(Y_TASKS: size, X_TASKS: size):
# TeX: version OverviewThreads 1
# TeX: begin OverviewThreads[0]
    # CPU scope here.
    for i in seq(0, 100):
    # TeX: color line *
      # ....
        pass  # Still a typical CPU loop, 100 iterations
    with CudaDeviceFunction(clusterDim=1, blockDim=256):
        # CUDA scope inside this CudaDeviceFunction block.
        # The task_y/task_x loop nest spawns Y_TASKS × X_TASKS tasks,
        # each assigned to a cluster for execution in an unspecified mapping.
        for task_y in cuda_tasks(0, Y_TASKS):
            for task_x in cuda_tasks(0, X_TASKS):
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

# Not a proc since this doesn't compile at all
def mma_example():
    # TeX: version mma_a 6
    # TeX: begin mma_a[0]
    word[2 @ (16 * bit)]
    # TeX: end mma_a[0]

    # TeX: begin mma_a[1]
    # TeX: color line *
    #   gggggggggg
    row[4 @ thread, 2 @ (16 * bit)]
    # TeX: end mma_a[1]

    # TeX: begin mma_a[2]
    matrix[
        # TeX: color line *
      # gggggggggggggggg
        8 @ (4 * thread),               # M dimension
        4 @ thread, 2 @ (16 * bit),     # K dimension
    ]
    # TeX: end mma_a[2]

    # TeX: begin mma_a[3]
    # TeX: begin mma_a[4]
    matrices[
        # TeX: color line mma_a[3]
      # ggggggggggggggg
        2 @ register_id, 8 @ (4 * thread),                  # M dimension
        # TeX: color line mma_a[3]
      # ggggggggggggggggggggg
        2 @ (2 * register_id), 4 @ thread, 2 @ (16 * bit),  # K dimension
    ]
    # TeX: end mma_a[3]
    # Free variables highlighted in yellow
    matrices[m_reg, m_thr, k_reg, k_thr, k_packed] = (
        # TeX: color line *
        #yyyyyyyyyyyyy                      yyyyyyyyyyy
        (base_register + m_reg + k_reg*2) * register_id +
        # TeX: color line *
        #yyyyyyyyyyy                      yyyyyyyyy
        (base_thread + m_thr*4 + k_thr) * thread_id +
        # TeX: color line *
        #               yyy
        k_packed * 16 * bit)
    # TeX: end mma_a[4]

    # TeX: begin mma_a[5]
    # NB here the fast dimension is the right-most, which is opposite of CuTe
    matrices[
        [2 @ register_id, 8 @ (4 * thread)],                 # M dimension
        [2 @ (2 * register_id), 4 @ thread, 2 @ (16 * bit)]  # K dimension
    ]
    # Not 1000% sure I did the math right:
    matrices[m, k] = (
        (base_register + DivAndMod(m, 8, 2) + DivAndMod(k, 8, 2) * 2) * register_id +
        (base_thread + DivAndMod(m, 1, 8) * 4 + DivAndMod(k, 2, 4)) * thread_id +
        DivAndMod(k, 1, 2) * 16 * bit)
    # TeX: end mma_a[5]


def annotation_example():
    # TeX: version ann 4
    # TeX: begin ann[0]
    # Control variable from unseen cuda_tasks loop $\mathrm{(Section~\ref{sec:SeqPar})}$
    m_offset: index
    # Unseen input parameters to the proc
    gmem_a: f32[M, K]  # Strides not given (compiler should auto-fill "obvious" strides)
    a_scale: f32

    smem_a: f32[RING, M_TILE, K_TILE]
    # Each thread has its own window references, so "stride" 1 thread between array entries.
    # No real syntax for this yet (array of window references).
    # TeX: begin ann[1]
    gmem_ref[K_TILE @ thread]: GmemWindowExpr
    smem_ref[K_TILE @ thread]: SmemWindowExpr

    for k_thr in cuda_threads(0, K_TILE, unit=cuda_thread):
        # Initialize per-thread GMEM pointers. These will be incremented each k iteration.
        # TeX: color line ann[1]
      # ggggggggggggggg
        gmem_ref[k_thr] = gmem_a[m_offset:, k_thr:]
    # TeX: end ann[0] ann[1]
    # TeX: begin ann
    for k_iter in seq(0, K_ITERS):
        for k_thr in cuda_threads(0, K_TILE, unit=cuda_thread):
            # Each thread will be assigned to fill one column of one ring buffer entry of smem_a.
          # TeX: color line ann[1]
          # rrrrrrrrrrrrrrr
            smem_ref[k_thr] = smem_a[k_iter % RING, 0:, k_thr]
            for m in seq(0, M_TILE):
                # TeX: end ann
                """
                annotation: smem_ref[k_thr] = smem_a[k_iter % RING, 0:, k_thr]
                deduced:    smem_ref[k_thr][m] = smem_a[k_iter % RING, m, k_thr]
                annotation: gmem_ref[k_thr] = gmem_a[m_offset:, k_iter*K_TILE + k_thr:]
                deduced:    gmem_ref[k_thr][m, 0] = gmem_a[m_offset + m, k_iter*K_TILE + k_thr]
                read gmem_ref[k_thr][m, 0]: gmem_a(m_offset + m, k_iter*K_TILE + k_thr)
                read a_scale: a_scale
                write: gmem_a(m_offset + m, k_iter*K_TILE + k_thr) * a_scale
                """
                # TeX: begin ann
                # TeX: color remark ann[0]
              # bbbbbbbbbbbbbbbbbbbbbbbbbbbbb
                # This stmt will be annotated
              # TeX: color remark ann[1]
            # bbbbbbbbbbbbbbbrrrrrrrrrrrrrrrbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
              # annotation:  smem_ref[k_thr] = smem_a[k_iter % RING, 0:, k_thr]
                # TeX: color remark ann[1]
            # bbbbbbbbbbbbbbbrrrrrrrrrrrrrrrbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
              # auto-deduce: smem_ref[k_thr][m] = smem_a[k_iter % RING, m, k_thr]
              # TeX: color remark ann[1]
            # bbbbbbbbbbbbbbbgggggggggggggggbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
              # annotation:  gmem_ref[k_thr] = gmem_a[m_offset:, k_iter*K_TILE + k_thr:]
                # TeX: color remark ann[1]
            # bbbbbbbbbbbbbbbgggggggggggggggbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
              # auto-deduce: gmem_ref[k_thr][m, 0] = gmem_a[m_offset + m, k_iter*K_TILE + k_thr]
              # TeX: color remark ann[2]
            # bbbbbbbgggggggggggggggbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
              # read gmem_ref[k_thr][m, 0] = gmem_a(m_offset + m, k_iter*K_TILE + k_thr)
              # TeX: color remark ann[2]
            # bbbbbbbbbbbbbbbbbbbbbbbb
              # read a_scale = a_scale
              # TeX: color remark ann[3]
            # bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
              # write: gmem_a(m_offset + m, k_iter*K_TILE + k_thr) * a_scale
                # TeX: color line ann[1] ann[3]
              # rrrrrrrrrrrrrrr
                smem_ref[k_thr][m] = (
                    # TeX: color line ann[1] ann[2]
                  # ggggggggggggggg
                    gmem_ref[k_thr][m, 0]
                    * a_scale)
            # Increment gmem_ref by K_TILE on the K dimension.
            # The first [] selects a window reference from the array, and the second [] dereferences it.
            # TeX: color line ann[1]
          # ggggggggggggggg
            gmem_ref[k_thr] = gmem_ref[k_thr][0:, K_TILE:]
    # TeX: end ann

"""
# TeX: version mod_heuristic 1
# TeX: begin mod_heuristic[0]
# LHS known from window reference annotation; RHS known from write annotation
smem_a[k_iter % RING, m, k_thr] = gmem_a(m_offset + m,
                                         k_iter*K_TILE + k_thr) * a_scale
# Want to rewrite as some function f
smem_a[_0, _1, _2] = f(_0, _1, _2)
# Trivially,
m = _1, k_thr = _2
# For k_iter, use the identity
k_iter = (k_iter // RING) * RING + k_iter % RING  # // is floor division
# Therefore,
k_iter = (k_iter // RING) * RING + _0
smem_a[_0, _1, _2] = gmem_a(m_offset + _1,
                            (k_iter // RING * RING + _0)*K_TILE + _2) * a_scale
# All variables other than the following would get substituted with concrete values:
_0, _1, _2, a_scale
# If there were another tensor coordinate _3 indexed with
k_iter // RING
# as would occur with tiling, then the dependence on the
# concrete value of k_iter would be eliminated, as
k_iter = _3 * RING + _0
# TeX: end mod_heuristic[0]
"""


"""
# TeX: version cutlass_pseudocode 1
RING = get_pipeline_depth()  # Some tuning constant
M_cta, N_cta, K_cta = cta_tile_size() # Some tuning constants
# Create ring buffers. $k^{th}$ tile goes to ring buffer slot k % RING
A_smem: f16[RING, M_cta, K_cta]  # Row major
B_smem: f16[RING, N_cta, K_cta]  # Column major
# Register tile, split into 2 logical halves.
# This is used to hide latency of SMEM to RMEM loads.
# TeX: color line *
#              .
A_rmem: f16[2, x]  # Figure out exact size yourself (warp tiles within CTA tiles)
# TeX: color line *
#              .
B_rmem: f16[2, x]  # NB the 2 halves don't have to be explicit, just shown here for clarity.
# For the real code, guard cp.async calls so they don't read out of bounds.
# TeX: begin cutlass_pseudocode[0]
for k in seq(0, RING - 1):  # SMEM is like a skewed sliding window
    cp.async tile k of A and B into SMEM  # Pre-populate first few sliding window entries
    cp.async.commit_group  # Not sure of any scheduling op that allows this skewed stage_mem
cp.async.wait_group RING-2
__syncthreads()
# TeX: color line *
#                                                                                    rrrr
Ld low half of tile 0 of A, B from SMEM into A_rmem[0,:], B_rmem[0,:]  # Consumed at (s0)
C = 0  # MMA accumulators
# The $k^{th}$ iteration of the loop accumulates the $k^{th}$ tiles of A and B into C,
# and starts GMEM $\to$ SMEM, SMEM $\to$ RMEM loads needed for future iterations.
for k in seq(0, K / K_cta):
# TeX: color line *
#                                                                                         bbbb
    Ld high half of tile k of A, B from SMEM into A_rmem[1,:], B_rmem[1,:]  # Consumed at (s1)
# TeX: color line *
#                                     rrrr
    C += A_rmem[0,:] @ B_rmem[0,:]  # (s0), using mma.sync

    cp.async tile (k + RING - 1) of A and B into SMEM
    cp.async.commit_group
    cp.async.wait_group RING-2
    __syncthreads()

# TeX: color line *
#                 rrrr
    # Consumed at (s0) of the NEXT iteration
    Ld low half of tile (k+1) of A, B from SMEM into A_rmem[0,:], B_rmem[0,:]
# TeX: color line *
#                                     bbbb
    C += A_rmem[1,:] @ B_rmem[1,:]  # (s1), using mma.sync
# TeX: end cutlass_pseudocode[0]
cp.async.wait_group 0
__syncthreads()
Write out C
"""
