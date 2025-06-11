from exo import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *

xyzzy = 1
cuda_unit = cuda_thread
lo, hi = 0, 1

@proc
def loop_example():
    with CudaDeviceFunction(blockDim=32):
        for task in cuda_tasks(0, xyzzy):
            # TeX: version cuda_threads 1 
            # TeX: begin cuda_threads[0]
            # Parent collective tiling here (domain $D^P$, box $B^P$); assume domain completion so $D = D^P$
            for iter in cuda_threads(lo, hi, unit=cuda_unit):
                # New collective tiling here (domain $D$, tile $T$, box $B$, tile $O$)
                # TeX: end cuda_threads[0]
                pass

@proc
def tmp_weird():
    with CudaDeviceFunction(clusterDim=4, blockDim=256):
        for task in cuda_tasks(0, xyzzy):
            for w in cuda_threads(0, 8, unit=cuda_warp):
                for cta in cuda_threads(0, 4, unit=cuda_cta_in_cluster):
                    pass

@proc
def simple_example():
    # TeX: version simple 2
    # TeX: begin simple
    with CudaDeviceFunction(blockDim=256):
        for task in cuda_tasks(0, xyzzy):
            # TeX: remark simple[1]
            # Tile here: (256,)
            # TeX: color remark simple[0]
            # rrrrrrrrrrrrrrrrrrrrr  yyyyyyyyyyyyyyyyyyyyyyy
            # Distributed (16 x 16); non-distributed (8 x 4)
            # TeX: color line simple[0]
            #         rrrrrr  yyyy
            vals: f32[16, 16, 8, 4] @ CudaRmem
            # TeX: color remark simple[1]
            # gggggggggggggggggg  rrrrrrrrrrrrrrrr
            # m: $256\mapsto 16$, threadIdx.x / 16
            # TeX: color line *
            #   g
            for m in cuda_threads(0, 16, unit=16 * cuda_thread):
                # TeX: remark simple[1]
                # Tile here: (16,)
                # TeX: color remark simple[1]
                # vvvvvvvvvvvvvvvv  rrrrrrrrrrrrrrrr
                # n: $16\mapsto 1$, threadIdx.x % 16
                # TeX: color line *
                #   v
                for n in cuda_threads(0, 16, unit=cuda_thread):
                    # TeX: remark simple[1]
                    # Tile here: (1,)
                    # TeX: color line simple[1]
                    #    g  v
                    vals[m, n, 0, 0] = 0
                    # TeX: color line simple[1]
                    #    g  v
                    vals[m, n, 0, 1] = 0
                    # TeX: color remark simple[1]
                    #           rrrrrrrrrrrrrrrr ggg  rrrrrrrrrrrrrrrr vvv
                    # Deduced:  threadIdx.x / 16 (m), threadIdx.x % 16 (n)
                    # TeX: remark simple[0]
                    # ...
    # TeX: end simple
"""
# TeX: version simple_cxx 1
# TeX: begin simple_cxx
# TeX: color line *
#            yyyyy
  float vals[8 * 4];
  # TeX: color line *
  #       ggggggggggg   rrrrrrrrrrrrrrrr
  if (int exo_16thr_m = threadIdx.x / 16; 1) {
  # TeX: color line *
  #         vvvvvvvvvv   rrrrrrrrrrrrrrrr
    if (int exo_1thr_n = threadIdx.x % 16; 1) {
      vals[0] = 0.0f;
      vals[1] = 0.0f;
# TeX: end simple_cxx

"""

if False:
    @proc
    def seq_fail():
        # TeX: version seq_fail 1
        # TeX: begin seq_fail
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, xyzzy):
                vals: f32[16, 16, 16, 4] @ CudaRmem
                # TeX: color line *
                #   g
                for m in cuda_threads(0, 16, unit=16 * cuda_thread):
                    # TeX: color line *
                    #   v
                    for n in cuda_threads(0, 16, unit=cuda_thread):
                        # TeX: color line *
                        #   b
                        for s in seq(0, 16):
                            # Expecting tiling chain 256 -> ... -> 1
                            # TeX: color line *
          #                                    b                            gggggggggg
          # Failure: non-cuda_threads variable s consumed when we only have m: 256->16
                            # TeX: color line *
                            #    g  b  v
                            vals[m, s, n, 0] = 0
                            # Remedy: reorder s and n
                            # TeX: color line *
                            #    g  v  b            gggggggggg  vvvvvvvv
                            vals[m, n, s, 0] = 0  # m: 256->16, n: 16->1
        # TeX: end seq_fail

if False:
    @proc
    def mismatched():
        # TeX: version mismatched 1
        # TeX: begin mismatched[0]
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, xyzzy):
                vals: f32[16, 4] @ CudaRmem
                # TeX: color line *
                #   g
                for m in cuda_threads(0, 16, unit=4 * cuda_thread):# m = threadIdx.x / 4
                    # TeX: color line *
                    #   v
                    for n in cuda_threads(0, 4, unit=cuda_thread):# n = threadIdx.x % 4
                        # TeX: color line *
                        #    g  v            gggggg  vvvvvvv
                        vals[m, n] = 0  # m: 256->4, n: 4->1
                # TeX: color line *
                #            rrrrrrrrrrrrrrr ggg  rrrrrrrrrrrrrrr vvv
                # Deduction: threadIdx.x / 4 (m), threadIdx.x % 4 (n)
                with CudaWarps(1, 3):
                    for m in cuda_threads(0, 16, unit=4 * cuda_thread):# m = (threadIdx.x - 32) / 4
                        for n in cuda_threads(0, 4, unit=cuda_thread): # n = threadIdx.x % 4
                            # TeX: color line *
                            #    g  v            gggggg  vvvvvvv
                            vals[m, n] = 0  # m: 256->4, n: 4->1
                    # TeX: color line *
                    #                       rrrrrrrrrrrrrrrrrrrrrr ggg  rrrrrrrrrrrrrrr vvv
                    # Mismatched deduction: (threadIdx.x - 32) / 4 (m), threadIdx.x % 4 (n)
                # TeX: color line *
                #   y
                for t in cuda_threads(0, 16, unit=cuda_thread):# t = threadIdx.x
                    # TeX: color line *
                    #   b
                    for s in seq(0, 4):
                        # TeX: color line *
                        #    y  b         yyyyyyyyy   b
                        vals[t, s] = 0  # t: 256->1;  s not distributed
                # TeX: color line *
                #                       rrrrrrrrrrr  y
                # Mismatched deduction: threadIdx.x (t)  [1 dims != 2 dims]
        # TeX: end mismatched[0]

@proc
def matched():
    # TeX: version matched 1
    # TeX: begin matched[0]
    with CudaDeviceFunction(blockDim=128):
        for task in cuda_tasks(0, xyzzy):
            vals: f32[16, 8] @ CudaRmem
            # TeX: color line *
            #   g
            for m in cuda_threads(0, 16, unit=8 * cuda_thread):# m = threadIdx.x / 8
                # TeX: color line *
                #   v
                for n in cuda_threads(0, 8, unit=cuda_thread):# n = threadIdx.x % 8
                    # TeX: color line *
                    #    g  v         ggggggggg  vvvvvvv
                    vals[m, n] = 0  # m: 128->8, n: 8->1
            # TeX: color line *
            #            rrrrrrrrrrrrrrr ggg  rrrrrrrrrrrrrrr vvv
            # Deduction: threadIdx.x / 8 (m), threadIdx.x % 8 (n)
            #
            # TeX: color remark! matched[0]
            #                                                            rrrrrrrrrrrrr
            # The names of the variables do not matter; only the deduced CollIndexExpr
            for a in cuda_threads(0, 16, unit=8 * cuda_thread):# a = threadIdx.x / 8
                for b in cuda_threads(0, 8, unit=cuda_thread):# b = threadIdx.x % 8
                    vals[a, b] = 0
            # TeX: color line *
            #            rrrrrrrrrrrrrrr      rrrrrrrrrrrrrrr
            # Deduction: threadIdx.x / 8 (a), threadIdx.x % 8 (b)
            #
            # TeX: remark! matched[0]
            # We can also transpose the loops and have it still work.
            # TeX: color remark matched[0]
            #                                                               rrrrrrrrrrrrr
            # The tiling chain is different, but it works since the deduced CollIndexExpr
            # tuple matches. This example requires a custom collective unit (every 8th thread).
            # n = threadIdx.x % 8
            # TeX: color line *
            #   v
            for n in cuda_threads(0, 8, unit=CollUnit((8,), (1,), "one_thread_per_8", None)):
                                           # CollUnit(domain, box, __repr__, scaled_dim_idx)
                # TeX: color line *
                #   g
                for m in cuda_threads(0, 16, unit=cuda_thread):  # m = threadIdx.x / 8
                    # TeX: color line *
                    #    g  v         vvvvvvvvvv  ggggggg
                    vals[m, n] = 0  # n: 128->16, m:16->1 (not same order as indices)
            # TeX: color line *
            #            rrrrrrrrrrrrrrr ggg  rrrrrrrrrrrrrrr vvv
            # Deduction: threadIdx.x / 8 (m), threadIdx.x % 8 (n)
    # TeX: end matched[0]

if False:
    @proc
    def broken_chain():
        # TeX: version broken_chain 3
        # TeX: begin broken_chain[0]
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, xyzzy):
                # TeX: color line *
                #                                 rrrrrrrrrrrrrrrrrr
                vals: f32[16, 8, 2] @ CudaRmem  # t_a = 256; t_n = 1
                # TeX: color line *
                #   y                                                   yyyyyyyyyyyyy
                for b in cuda_threads(0, 2, unit=128 * cuda_thread):  # b: 256->128
                    # TeX: color line *
                    #   g                                                  ggggggggggg
                    for m in cuda_threads(0, 16, unit=8 * cuda_thread):  # m: 128->8
                        # TeX: color line *
                        #   v                                             vvvvvvvvv
                        for n in cuda_threads(0, 8, unit=cuda_thread):  # n: 8->1
                            # TeX: color line *
                            #   b
                            for s in seq(0, 2):
                                # TeX: color line *
                # ggggggggg     vvvvvvv                                   rrrrrr
                # m: 128->8 and n: 8->1 is insufficient to reach the goal 256->1
                # TeX: color line *
                #                                                  b
                # Distributed memory analysis fails upon consuming s (non-cuda_threads iter)
                                # TeX: color line *
                                #    g  v  b
                                vals[m, n, s] = 0
                # TeX: color line *
                #                                                                                      y
                # This rule enforces that two different thread collectives (in this case, differing by b value)
                # can't access the same distributed shard.
        # TeX: end broken_chain[0]

# TeX: version warp_example 1

@proc
def warp_example():
    # TeX: begin warp_example[0]
    with CudaDeviceFunction(blockDim=256):
        for task in cuda_tasks(0, xyzzy):
            # TeX: color line *
            #      rrrr                                  rrrrrrrrrrrrrrrrrrr
            D: f32[2, 4, 6, 16, 8] @ Sm80_RmemMatrixD  # t_a = 256, t_n = 32
            # TeX: color line *
            #            rrrrrrrrrrrrrrrrr  rrrrrrrrrrrrrrrrrrrrrrrr
            # Deduction: threadIdx.x / 128, threadIdx.x % 128 / 32
            # TeX: color line *
            #   gg
            for mw in cuda_threads(0, 2, unit=128 * cuda_thread):# mw = threadIdx.x / 128
                # TeX: color line *
                #   vv
                for nw in cuda_threads(0, 4, unit=32 * cuda_thread):
                    # nw = threadIdx.x % 128 / 32
                    # TeX: color line *
                    #   b
                    for s in seq(0, 6):
                        # TeX: color line *
                        #                      gg  vv  b           gggggggggggg  vvvvvvvvvv
                        Sm80_mma_zero_d_tf32(D[mw, nw, s, :, :]) # mw: 256->128, nw:128->32
                # TeX: color line *
                #                                   b                                        rrrrrrrr
                # Indexing by seq-for iter variable s is OK as we already reached the target t_n = 32
# TeX: end warp_example[0]

# TeX: version chain 2

@proc
def chain_0():
    # TeX: begin chain[0]
    with CudaDeviceFunction(blockDim=256):
        for task in cuda_tasks(0, xyzzy):
            # TeX: color line *
            #         rrrrr                   rrrrrrrrrrrrrrrrrr
            vals: f32[16, 8, 2] @ CudaRmem  # t_a = 256; t_n = 1
            # TeX: color line *
            #            rrrrrrrrrrrrrrrrrrrrrr  rrrrrrrrrrrrrrr
            # Deduction: (threadIdx.x - 64) / 8, threadIdx.x % 8
            # tile = (256,), box = (256,), offset = (0,)
            with CudaWarps(2, 6):  # Offset by 2*32 = 64 threads; box = (6-2)*32 = 128 threads
                # TeX: color line *
                #        gggggg
                # tile = (256,), box = (128,), offset=(64,)
                # TeX: color line *
                #   g                                                  ggggggggg
                for m in cuda_threads(0, 16, unit=8 * cuda_thread):  # m: 256->8
                    # TeX: color line *
                    #        gggg
                    # tile = (8,), box = (8,), offset = (0,)
                    # TeX: color line *
                    #   v                                             vvvvvvv
                    for n in cuda_threads(0, 8, unit=cuda_thread):  # n: 8->1
                        # tile = (1,), box = (1,), offset = (0,)
                        # TeX: color line *
                        #   b
                        for s in seq(0, 2):
                            # TeX: color line *
                            #    g  v  b         ggggggggg  vvvvvvv
                            vals[m, n, s] = 0  # m: 256->8, n: 8->1
# TeX: end chain[0]

@proc
def chain_1():
    # TeX: begin chain[1]
    with CudaDeviceFunction(blockDim=256):
        for task in cuda_tasks(0, xyzzy):
            # TeX: color line *
            #         rrrrrrrr                rrrrrrrrrrrrrrrrrr
            vals: f32[16, 8, 2] @ CudaRmem  # t_a = 256; t_n = 1
            # TeX: color line *
            #            rrrrrrrrrrrrrrrrrrrrr  rrrrrrrrrrrrrrr  rrrrrrrrrrrrrrrrr
            # Deduction: threadIdx.x % 128 / 8, threadIdx.x % 8, threadIdx.x / 128
            # TeX: color line *
            #   y                                                   yyyyyyyyyyyyy
            for b in cuda_threads(0, 2, unit=128 * cuda_thread):  # b: 256->128
                # TeX: color line *
                #   g                                                  ggggggggggg
                for m in cuda_threads(0, 16, unit=8 * cuda_thread):  # m: 128->8
                    # TeX: color line *
                    #   v                                             vvvvvvvvv
                    for n in cuda_threads(0, 8, unit=cuda_thread):  # n: 8->1
                        # TeX: color line *
                        #    g  v  y         yyyyyyyyyyy  ggggggggg  vvvvvvv
                        vals[m, n, b] = 0  # b: 256->128, m: 128->8, n: 8->1
# TeX: end chain[1]

# TeX: version repeated 2

if False:
    @proc
    def repeated_index():
        # TeX: begin repeated[0]
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, xyzzy):
                # TeX: color line *
                #                                   rrrrrrrrrrrrrrrrrr
                vals: f32[16, 16, 16] @ CudaRmem  # t_a = 256, t_n = 1
                # TeX: color line *
                #   g
                for m in cuda_threads(0, 16, unit=16 * cuda_thread):
                    # TeX: color line *
                    #   v
                    for n in cuda_threads(0, 16, unit=cuda_thread):
                        # TeX: color line *
                        #    g  g  v         gggggggggg  gggggggggg
                        vals[m, m, n] = 0  # m: 256->16, m: 256->16
                # TeX: color line *
                #                                       ggggggggg                            rrrrrrr
                # Fail: we encounter another index with t_0 = 256 before we reach the target t_n = 1
# TeX: end repeated[0]

@proc
def repeated_index():
    # TeX: begin repeated[1]
    with CudaDeviceFunction(blockDim=256):
        for task in cuda_tasks(0, xyzzy):
            # TeX: color line *
            #         rrrrrr  yy                rrrrrrrrrrrrrrrrrr
            vals: f32[16, 16, 16] @ CudaRmem  # t_a = 256, t_n = 1
            # TeX: color line *
            #            rrrrrrrrrrrrrrrr  rrrrrrrrrrrrrrrr
            # Deduction: threadIdx.x % 16, threadIdx.x / 16
            # TeX: color line *
            #   g
            for m in cuda_threads(0, 16, unit=16 * cuda_thread):# m = threadIdx. / 16
                # TeX: color line *
                #   v
                for n in cuda_threads(0, 16, unit=cuda_thread):# n = threadIdx.x % 16
                    # TeX: color line *
                    #    v  g            gggggggggg  vvvvvvvv
                    vals[n, m, m] = 0  # m: 256->16, n: 16->1
            # TeX: color line *
            #                                                                             rrrrrrr
            # Second m not deduced as distributed idx since we already reached the target t_n = 1
# TeX: end repeated[1]

"""
# TeX: version repeated_cxx 1
# TeX: begin repeated_cxx[0]
# TeX: color line *
#            yy
  float vals[16];
  # TeX: color line *
  #       ggggggggggg   rrrrrrrrrrrrrrrr
  if (int exo_16thr_m = threadIdx.x / 16; 1) {
    # TeX: color line *
    #       vvvvvvvvvv   rrrrrrrrrrrrrrrr
    if (int exo_1thr_n = threadIdx.x % 16; 1) {
      # TeX: color line *
      #    ggggggggggg
      vals[exo_16thr_m] = 0.0f;
      # TeX: end repeated_cxx[0]
"""


# TeX: version many_tensors 1

@proc
def many_tensors():
    # TeX: begin many_tensors[0]
    with CudaDeviceFunction(warp_config = [CudaWarpConfig("first_half", 8),
                                           CudaWarpConfig("second_half", 8)]):
        for task in cuda_tasks(0, xyzzy):
            # TeX: color line *
            #                                         rrrrrrrrrrrrrrrr  rrrrrrrrrrrrrrrr
            val1: f32[16, 16] @ CudaRmem # Deduction: threadIdx.x / 16, threadIdx.x % 16
            # TeX: color line *
            #                                         rrrrrrrrrrrrrrrr  rrrrrrrrrrrrrrrr
            val2: f32[16, 16] @ CudaRmem # Deduction: threadIdx.x / 16, threadIdx.x % 16
            # TeX: color line *
            #                                         rrrrrrrrrrrrrrrrrrrrrrrr  rrrrrrrrrrrrrrrr
            val3: f32[16, 16] @ CudaRmem # Deduction: (threadIdx.x - 256) / 16, threadIdx.x % 16

            with CudaWarps(name="first_half"):
                # TeX: color line *
                #   g
                for m in cuda_threads(0, 16, unit=16 * cuda_thread):  # m = threadIdx.x / 16
                    # TeX: color line *
                    #   v
                    for n in cuda_threads(0, 16, unit=cuda_thread):   # n = threadIdx.x % 16
                        # TeX: color line *
                        #    g  v
                        val1[m, n] = 0
                        # TeX: color line *
                        #    g  v
                        val2[m, n] = 0
            with CudaWarps(name="second_half"):
                # m = (threadIdx.x - 256) / 16
                # TeX: color line *
                #   g
                for m in cuda_threads(0, 16, unit=16 * cuda_thread):
                    # TeX: color line *
                    #   v
                    for n in cuda_threads(0, 16, unit=cuda_thread):   # n = threadIdx.x % 16
                        # TeX: color line *
                        #    g  v
                        val3[m, n] = 0
# TeX: end many_tensors[0]


@proc
def two_ctas():
    with CudaDeviceFunction(blockDim=256, clusterDim=2):
        for task in cuda_tasks(0, xyzzy):
            vals: f32[2, 100] @ CudaSmemLinear
            for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                for tid in cuda_threads(0, 1):
                    vals[cta, 5] = 0
