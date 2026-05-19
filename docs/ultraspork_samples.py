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
