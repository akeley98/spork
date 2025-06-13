# Snippets of code for spork_guide.pdf that are not expected to compile.

# TeX: version loop_mode_syntax 1
# TeX: begin loop_mode_syntax[0]
# TeX: color line *
#           gggggggggggggg         ggggggggggggggggggggggggggggg
for iter in loop-mode-name(lo, hi, optional-keyword=optional-arg, ...):
# TeX: end loop_mode_syntax[0]

# TeX: version cuda_threads_syntax 3
# TeX: begin cuda_threads_syntax[0]
for iter in cuda_threads(0, hi, unit=<collective-unit>):
    ...body
# TeX: end cuda_threads_syntax[0]

# TeX: version why_dist 1
# TeX: begin why_dist[0]
# TeX: remark! *
# Interpreted with M-semantics (multithreaded)
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
# Interpreted with S-semantics (single-threaded)
# TeX: color line *
#                                                rrrrrrrrrrrrrrrrrrr
x: f32 @ CudaRmem  # implicitly one x per thread (not valid Exo-GPU)
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


@proc
def mbarrier_teams():
    with CudaDeviceFunction(blockDim=512):
        for task in cuda_tasks(0, xyzzy):
            # TeX: version mbarrier_teams 1
            # TeX: begin mbarrier_teams[0]
            foo: f32[2, 128] @ CudaSmemLinear
            my_mbarrier: barrier @ CudaMbarrier
            # Split into two teams of 256 threads each
            for team in cuda_threads(0, 2, unit=256 * cuda_thread):
                with CudaWarps(0, 4):
                    for tid in cuda_threads(0, 128, unit=cuda_thread):
                        foo[team, tid] = 137
                    # Threads [0, 127] arrive on my_mbarrier[0]; threads [256, 383] arrive on my_mbarrier[1]
                    Arrive(cuda_classic, my_mbarrier[team], 1)
            for team in cuda_threads(0, 2, unit=256 * cuda_thread):
                with CudaWarps(4, 8):
                    # Threads [128, 255] wait for my_mbarrier[0]; threads [384, 511] wait for my_mbarrier[1]
                    Await(my_mbarrier[team], cuda_classic, ~0)
                    for tid in cuda_threads(0, 128, unit=cuda_thread):
                        bar: f32 @ CudaRmem
                        bar = foo[team, tid]
            # TeX: end mbarrier_teams[0]
