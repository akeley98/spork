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

