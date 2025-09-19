from __future__ import annotations

from exo import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.stdlib.scheduling import *

if True:
# TeX: version cpu 5
# TeX: begin cpu[:4]
# TeX: color line cpu[0]
# bbbbb
  @proc
  # TeX: color line cpu[1]
  #                     bbbb     bbbb     bbbb
  def nyc25_gemm_cpu(M: size, N: size, K: size,
  # TeX: color line cpu[1]
  #                     bbbbbbbbb
  # TeX: color line cpu[2]
  #                               bbbbbb
                     A: f32[M, K] @ DRAM,
  # TeX: color line cpu[1]
  #                     bbbbbbbbb
  # TeX: color line cpu[2]
  #                               bbbbbb
                     B: f32[K, N] @ DRAM,
  # TeX: color line cpu[1]
  #                     bbbbbbbbb
  # TeX: color line cpu[2]
  #                               bbbbbb
                     C: f32[M, N] @ DRAM):
    # TeX: begin cpu[4]
    # TeX: color line cpu[3:]
    #   g    ggg
    for m in seq(0, M):
      # TeX: color line cpu[3:]
      #   v    vvv
      for n in seq(0, N):
        # TeX: color line cpu[1]
        #      bbb
        # TeX: color line cpu[2]
        #          bbbbbb
        accum: f32 @ DRAM
        accum = 0
        # TeX: color line cpu[3]
        #   b    bbb
        # TeX: color line cpu[4]
        #   b
        for k in seq(0, K):
          # TeX: color line cpu[4]
          #          g            v
          accum += A[m, k] * B[k, n]
        # TeX: color line cpu[4]
        # g  v
        C[m, n] = accum
        # TeX: end cpu[4]
# TeX: end cpu[:4]

# TeX: version reorder_loops 2
# TeX: version m_divide_loop 3
# TeX: begin m_divide_loop reorder_loops[0]
M1 = 192
M0 = 12
# TeX: end m_divide_loop
# TeX: version n_divide_loop 1
# TeX: begin n_divide_loop
N1 = 256
N0 = 16
# TeX: end n_divide_loop reorder_loops[0]
# TeX: version smem_broken 3
# TeX: begin smem_broken[0]
K0 = 16
# TeX: end smem_broken[0]

# TeX: summary
# @proc body excerpt below

@proc
def nyc25_gemm_m_divide_loop(M: size, N: size, K: size, A: f32[M, K], B: f32[K, N], C: f32[M, N]):
  # TeX: begin m_divide_loop
  # TeX: color line m_divide_loop[2]
# gggggggggggggggggg
  assert M % M1 == 0
  # TeX: color line m_divide_loop[0]
  #   gg
  for m2 in seq(0, M / M1):
    # TeX: color line m_divide_loop[0]
    #   gg
    for m1 in seq(0, M1 / M0):
      # TeX: color line m_divide_loop[0]
      #   gg
      for m0 in seq(0, M0):
        for n in seq(0, N):
          accum: f32 @ DRAM
          accum = 0
          for k in seq(0, K):
            accum += (
                # TeX: color line m_divide_loop[1]
                # gggggggggggggggggggggg
                A[m2 * M1 + m1 * M0 + m0, k]
              * B[k, n]
            )
          # TeX: color line m_divide_loop[1]
          # gggggggggggggggggggggg
          C[m2 * M1 + m1 * M0 + m0, n] = accum
  # TeX: end m_divide_loop


@proc
def nyc25_gemm_n_divide_loop(M: size, N: size, K: size, A: f32[M, K], B: f32[K, N], C: f32[M, N]):
  # TeX: begin n_divide_loop
  assert M % M1 == 0
  # TeX: color line *
# vvvvvvvvvvvvvvvvvv
  assert N % N1 == 0
  for m2 in seq(0, M / M1):
    for m1 in seq(0, M1 / M0):
      for m0 in seq(0, M0):
        # TeX: color line *
        #   vv
        for n2 in seq(0, N / N1):
          # TeX: color line *
          #   vv
          for n1 in seq(0, N1 / N0):
            # TeX: color line *
            #   vv
            for n0 in seq(0, N0):
              accum: f32 @ DRAM
              accum = 0
              for k in seq(0, K):
                accum += (
                    A[m2 * M1 + m1 * M0 + m0, k]
                    # TeX: color line *
                #        vvvvvvvvvvvvvvvvvvvvvv
                  * B[k, n2 * N1 + n1 * N0 + n0]
                )
              # TeX: color line *
              #                         vvvvvvvvvvvvvvvvvvvvvv
              C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] = accum
# TeX: end n_divide_loop


@proc
def nyc25_gemm_reorder_loop(M: size, N: size, K: size, A: f32[M, K] @ DRAM, B: f32[K, N] @ DRAM, C: f32[M, N] @ DRAM):
  assert M % M1 == 0
  assert N % N1 == 0
  # TeX: begin reorder_loops
  # TeX: color line *
  #   gg
  for m2 in seq(0, M / M1):
    # TeX: color line *
    #   vv
    for n2 in seq(0, N / N1): # Outer loops over ``large'' M1 $\times$ N1 tiles of C
      # TeX: color line *
      #   gg
      for m1 in seq(0, M1 / M0):
        # TeX: color line *
        #   vv
        for n1 in seq(0, N1 / N0): # Middle loops over ``small'' M0 $\times$ N0 tiles of C
          # TeX: color line *
          #   gg
          for m0 in seq(0, M0):
            # TeX: color line *
            #   vv
            for n0 in seq(0, N0): # Inner loops over C elements to fill one-by-one
              accum: f32 @ DRAM
              accum = 0
              # TeX: color line *
              #   b
              for k in seq(0, K):  # Compute dot product A[m,:] $\circ$ B[:,n]
              # TeX: color line reorder_loops[1]
              #             gggggggggggggggggggggg
                accum += (A[m2 * M1 + m1 * M0 + m0, k]
              # TeX: color line reorder_loops[1]
              #                vvvvvvvvvvvvvvvvvvvvvv
                        * B[k, n2 * N1 + n1 * N0 + n0])
              # TeX: color line reorder_loops[1]
              # gggggggggggggggggggggg  vvvvvvvvvvvvvvvvvvvvvv
              C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] = accum
  # TeX: end reorder_loops


# TeX: version simple_gpu 6
# TeX: begin simple_gpu[0]
@proc
def nyc25_gemm_simple_gpu(M: size, N: size, K: size,
# TeX: color line *
#                                      bbbbbbbbbbbbbbbb
                          A: f32[M, K] @ CudaGmemLinear,  # ... B, C
# TeX: end simple_gpu[0]
# TeX: color line *
#                                      bbbbbbbbbbbbbbbb
                          B: f32[K, N] @ CudaGmemLinear,
# TeX: color line *
#                                      bbbbbbbbbbbbbbbb
                          C: f32[M, N] @ CudaGmemLinear):
# TeX: begin simple_gpu[0]
  assert M % M1 == 0  # GMEM = global (device-wide) memory
  assert N % N1 == 0
  # TeX: begin simple_gpu[1:7]
  # TeX: color line simple_gpu[0]
  #    bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
  # TeX: color line simple_gpu[1]
  #                       bbbbbbbbbbbb
  with CudaDeviceFunction(blockDim=256):  # User chose 256 threads per block
    # TeX: end simple_gpu
    # These are needed for intermediate slides to avoid GPU loops too early.
    # TeX: begin simple_gpu[0]
   if False:
    for m2 in seq(0, M / M1):
      for n2 in seq(0, N / N1):
        for m1 in seq(0, M1 / M0):
          for n1 in seq(0, N1 / N0):
            # TeX: end simple_gpu[0]
            pass
   if True:
    # TeX: begin simple_gpu[1:]
    # TeX: color line simple_gpu[1]
    #   gg    gggggggggg
    for m2 in cuda_tasks(0, M / M1):
      # TeX: color line simple_gpu[1]
      #   vv    vvvvvvvvvv
      for n2 in cuda_tasks(0, N / N1):
        # TeX: color remark! simple_gpu[5:]
        #         bb
        # Reorder k1 loop to out here (thread-block cooperative)
        # TeX: color line simple_gpu[2]
        #   gg    gggggggggggg
        # TeX: color line simple_gpu[3]
        #                                  ggggggggggggggggggggg
        for m1 in cuda_threads(0, M1 / M0, unit=16 * cuda_thread):
          # TeX: color line simple_gpu[2]
          #   vv    vvvvvvvvvvvv
          # TeX: color line simple_gpu[3]
          #                                  vvvvvvvvvvvvvvvv
          for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
            # TeX: end simple_gpu[1:]
            # TeX: begin simple_gpu
            # TeX: color line simple_gpu[4]
            #   gg    ggg
            for m0 in seq(0, M0):
              # TeX: color line simple_gpu[4]
              #   vv    vvv
              for n0 in seq(0, N0):
                # TeX: color line simple_gpu[0]
                #          bbbbbbbbbb
                accum: f32 @ CudaRmem  # CUDA per-thread register
                # TeX: color line simple_gpu[5]
              # rrrrrrrrr
                accum = 0
                # TeX: remark! simple_gpu[5:]
                # Divide loop: $\texttt{k} \mapsto \texttt{k1 * K0 + k0}$
                # TeX: color line simple_gpu[5]
                #   b    bbbbbbbbb
                for k in seq(0, K):
                  # TeX: remark simple_gpu[0]
                  # ... dot products
                  # TeX: end simple_gpu[0]
                  accum += (
                      A[m2 * M1 + m1 * M0 + m0, k]
                    * B[k, n2 * N1 + n1 * N0 + n0]
                  )
                # TeX: color line simple_gpu[5]
              # rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
                C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] = accum
# TeX: end simple_gpu[1:]

nyc25_gemm_simple_gpu = simplify(nyc25_gemm_simple_gpu)


# TeX: version loops 2

if False:
  # TeX: begin loops
  # TeX: color line loops[0]
  #         gggggggggg    gggggg
  for m2 in cuda_tasks(0, M / M1):
    # TeX: color line loops[0]
    #         vvvvvvvvvv    vvvvvv
    for n2 in cuda_tasks(0, N / N1):
      # TeX: color line loops[1]
      #         gggggggggggg    ggggg
      for m1 in cuda_threads(0, M1/M0,
          unit=16 * cuda_thread):
        # TeX: color line loops[1]
        #         vvvvvvvvvvvv    vvvvv
        for n1 in cuda_threads(0, N1/N0,
              unit=cuda_thread):
          # TeX: color line loops[1]
          #                gg
          for m0 in seq(0, M0):
            # TeX: color line loops[1]
            #                vv
            for n0 in seq(0, N0):
              # Compute dot products
    # Note: N1/N0 = 16
              # TeX: end loops
              pass


# TeX: version cuda_threads 1

if False:
  # TeX: begin cuda_threads[0]
  # TeX: color line loops[0]
  #         gggggggggg    gggggg
  for m2 in cuda_tasks(0, M / M1):
    # TeX: color line loops[0]
    #         vvvvvvvvvv    vvvvvv
    for n2 in cuda_tasks(0, N / N1):
      # Collective unit: thread block (CTA)
      # TeX: color line cuda_threads[0]
      #   yy                             ggggggggggggggggggggg
      for m1 in cuda_threads(0, M1 / M0, unit=16 * cuda_thread):
        # TeX: remark cuda_threads
        # Collective unit: 16 threads (N1/N0 = 16)
        # TeX: color line cuda_threads[0]
        #   rr                             vvvvvvvvvvvvvvvv
        for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
          # TeX: remark cuda_threads
          # Collective unit: 1 thread
          # TeX: end cuda_threads[0]
          pass


# TeX: version fission_def 2

if False:
  # TeX: begin fission_def[0]
  for i1 in loop_mode_1(lo1, hi1):
    # ...
    for iN in loop_mode_N(loN, hiN):
    # TeX: color line *
    # rr
      s1
      # FISSION HERE
    # TeX: color line *
    # bb
      s2
  # TeX: end fission_def[0]

  # TeX: begin fission_def[1]
  for i1 in loop_mode_1(lo1, hi1):
    # ...
    for iN in loop_mode_N(loN, hiN):
    # TeX: color line *
    # rr
      s1
  for i1 in loop_mode_1(lo1, hi1):
    # ...
    for iN in loop_mode_N(loN, hiN):
    # TeX: color line *
    # bb
      s2
  # TeX: end fission_def[1]


# TeX: version fission 4


@proc
def nyc25_gemm_fission(M: size, N: size, K: size,
                       A: f32[M, K] @ CudaGmemLinear,
                       B: f32[K, N] @ CudaGmemLinear,
                       C: f32[M, N] @ CudaGmemLinear):

  assert M % M1 == 0
  assert N % N1 == 0
  with CudaDeviceFunction(blockDim=256):
    # TeX: begin fission
    # TeX: color line fission[2]
    #   gg    ggggggggggggggggggggg
    for m2 in cuda_tasks(0, M / M1):
      # TeX: color line fission[2]
      #   vv    vvvvvvvvvvvvvvvvvvvvv
      for n2 in cuda_tasks(0, N / N1):  # Did not fission cuda_tasks
        accum: f32[M1/M0, N1/N0, M0, N0] @ CudaRmem
        # TeX: remark *
        # Zero-initialize accumulators
        # TeX: color line fission[1]
        #   gg    gggggggggggg
        for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
          # TeX: color line fission[1]
          #   vv    vvvvvvvvvvvv
          for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
            # TeX: color line fission[1]
            #   gg
            for m0 in seq(0, M0):
              # TeX: color line fission[1]
              #   vv
              for n0 in seq(0, N0):
                # TeX: color line fission[0]
              # rrrrrrrrrrrrrrrrrrrrrrrrr
                accum[m1, n1, m0, n0] = 0
        # TeX: remark fission[:3]
        # Main loop: computes dot products
        # TeX: remark! fission[3]
        # Main loop: computes dot products (focus of the talk)
        # TeX: color line fission[3]
        #   bb    bbbbbbbbbbbbbb
        for k1 in seq(0, K / K0):  # $\texttt{k} \mapsto \texttt{k1 * K0 + k0}$, reordered k1 loop
          # TeX: color line fission[1]
          #   gg    gggggggggggg
          for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
            # TeX: color line fission[1]
            #   vv    vvvvvvvvvvvv
            for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
              # TeX: color line fission[1]
              #   gg
              for m0 in seq(0, M0):
                # TeX: color line fission[1]
                #   vv
                for n0 in seq(0, N0):
                  # TeX: color line fission[3]
                  #   bb    bbbbbbbbbb
                  for k0 in seq(0, K0):
                    # TeX: color line fission[0]
                   #rrrrrrrrrrrrrrrrrrrrrrrrr  ......................                    ............  ......................
                    # TeX: color line fission[1:3]
                    #                          ......................                    ............  ......................
                    # TeX: color line fission[3]
                    #                        gg......................  bbbbbbbbbbbbg   vv............  ......................v
                    accum[m1, n1, m0, n0] += A[m2 * M1 + m1 * M0 + m0, k1 * K0 + k0] * B[k1 * K0 + k0, n2 * N1 + n1 * N0 + n0]
        # TeX: remark *
        # Epilogue: write to global memory
        # TeX: color line fission[1]
        #   gg    gggggggggggg
        for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
          # TeX: color line fission[1]
          #   vv    vvvvvvvvvvvv
          for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
            # TeX: color line fission[1]
            #   gg
            for m0 in seq(0, M0):
              # TeX: color line fission[1]
              #   vv
              for n0 in seq(0, N0):
                # TeX: color line fission[0]
              # rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr      ..............
                # TeX: color line fission[1:]
                #                                                         ..............
                C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] = accum[m1, n1, m0, n0]
# TeX: end fission


nyc25_gemm_fission = simplify(nyc25_gemm_fission)


if False:
    # TeX: version k1_before_smem 1
    # TeX: begin k1_before_smem
    # TeX: color line *
    #   bb    bbbbbbbbbbbbbb
    for k1 in seq(0, K / K0):
      for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
        for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
          for m0 in seq(0, M0):
            for n0 in seq(0, N0):
              # TeX: color line *
              #   bb    bbbbbbbbbb
              for k0 in seq(0, K0):
                accum[m1, n1, m0, n0] += (
                # TeX: color line *
                #   gg                                    g
                    A[m2 * M1 + m1 * M0 + m0, k1 * K0 + k0]
                # TeX: color line *
                #   vv                                    v
                  * B[k1 * K0 + k0, n2 * N1 + n1 * N0 + n0])
# TeX: end k1_before_smem


@proc
def nyc25_gemm_smem_broken(M: size, N: size, K: size, A: f32[M, K] @ CudaGmemLinear, B: f32[K, N] @ CudaGmemLinear, C: f32[M, N] @ CudaGmemLinear):
  assert M % M1 == 0
  assert N % N1 == 0
  assert K % K0 == 0
  with CudaDeviceFunction(blockDim=256):
    for m2 in cuda_tasks(0, M / M1):
      for n2 in cuda_tasks(0, N / N1):
        # TeX: remark *
        # Zero-initialize accumulators
        accum: f32[M1/M0, N1/N0, M0, N0] @ CudaRmem
        for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
          for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
            for m0 in seq(0, M0):
              for n0 in seq(0, N0):
                accum[m1, n1, m0, n0] = 0
        # TeX: begin smem_broken
        # TeX: remark! smem_broken[0]
        # k1 loop must be outside the cuda_threads loop for co-op!
        # TeX: color line smem_broken
        #   bb    bbbbbbbbbbbbbb
        for k1 in seq(0, K / K0):
          # TeX: end smem_broken[0]
          # TeX: color line smem_broken[1]
        # gggggg              gggggggggggggggg
          A_smem: f32[M1, K0] @ CudaSmemLinear  # CUDA shared memory
          # TeX: color line smem_broken[1]
        # vvvvvv              vvvvvvvvvvvvvvvv
          B_smem: f32[K0, N1] @ CudaSmemLinear  # CUDA shared memory
          for i0 in seq(0, M1):
            for i1 in seq(0, K0):
              # TeX: color line smem_broken[2]
            # ggggggg      g   gg                          g
              A_smem[i0, i1] = A[m2 * M1 + i0, k1 * K0 + i1]
          for i0 in seq(0, K0):
            for i1 in seq(0, N1):
            # TeX: color line smem_broken[2]
            # vvvvvvv      v   vv                          v
              B_smem[i0, i1] = B[k1 * K0 + i0, n2 * N1 + i1]
          # TeX: begin smem_broken[0]
          # TeX: color line smem_broken[0]
          #   yy
          for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
            # TeX: color line smem_broken[0]
            #   rr
            for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
              # TeX: color line smem_broken[0]
              #   gg
              for m0 in seq(0, M0):
                # TeX: color line smem_broken[0]
                #   vv
                for n0 in seq(0, N0):
                  # TeX: color line smem_broken
                  #   bb    bbbbbbbbbb
                  for k0 in seq(0, K0):
                  # TeX: end smem_broken[0]
                  # TeX: color line smem_broken[1]
                  #                           ggggggg                g
                    accum[m1, n1, m0, n0] += (A_smem[m1 * M0 + m0, k0]
                    # TeX: color line smem_broken[1]
                  #                             vvvvvvv                v
                                              * B_smem[k0, n1 * N0 + n0])
        # TeX: end smem_broken[1:]
        # TeX: remark *
        # Epilogue: write to global memory
        for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
          for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
            for m0 in seq(0, M0):
              for n0 in seq(0, N0):
                C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] = (
                    accum[m1, n1, m0, n0]
                )


del nyc25_gemm_smem_broken


# TeX: version smem_in_order 5


@proc
def nyc25_gemm_smem_in_order(M: size, N: size, K: size, A: f32[M, K] @ CudaGmemLinear, B: f32[K, N] @ CudaGmemLinear, C: f32[M, N] @ CudaGmemLinear):
  assert M % M1 == 0
  assert N % N1 == 0
  assert K % K0 == 0
  with CudaDeviceFunction(blockDim=256):
    for m2 in cuda_tasks(0, M / M1):
      for n2 in cuda_tasks(0, N / N1):
        accum: f32[M1/M0, N1/N0, M0, N0] @ CudaRmem
        for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
          for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
            for m0 in seq(0, M0):
              for n0 in seq(0, N0):
                accum[m1, n1, m0, n0] = 0
        # TeX: begin smem_in_order
        for k1 in seq(0, K / K0):
          A_smem: f32[M1, K0] @ CudaSmemLinear  # CUDA shared memory
          B_smem: f32[K0, N1] @ CudaSmemLinear  # CUDA shared memory
          # TeX: color line smem_in_order
          #   gg    gggggggggggg
          for m1 in cuda_threads(0, M1 / M0, unit=16 * cuda_thread):
            # TeX: color line smem_in_order
            #   gg    ggg
            for m0 in seq(0, M0):
              # TeX: color line smem_in_order
              #   gg    gggggggggggg
              for i1 in cuda_threads(0, K0, unit=cuda_thread):
                # TeX: color line smem_in_order[3]
              # rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
                A_smem[m1 * M0 + m0, i1] = A[m2 * M1 + m1 * M0 + m0, k1 * K0 + i1]
          # TeX: color line smem_in_order
          #   vv    vvvvvvvvvvvv
          for i0 in cuda_threads(0, K0, unit=16 * cuda_thread):
            # TeX: color line smem_in_order
            #   vv    vvvvvvvvvvvv
            for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
              # TeX: color line smem_in_order
              #   vv    vvv
              for n0 in seq(0, N0):
                # TeX: color line smem_in_order[3]
              # rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
                B_smem[i0, n1 * N0 + n0] = B[k1 * K0 + i0, n2 * N1 + n1 * N0 + n0]
          # TeX: end smem_in_order[0]
          # TeX: color line smem_in_order[1]
        # yyyyy ............................
          # TeX: color line smem_in_order[2]
        # yyyyy
          # TeX: color line smem_in_order[4]
          #     rrrrrrrrrrrrr
          Fence(cuda_in_order, cuda_in_order)  # __syncthreads()
          # TeX: begin smem_in_order[0]
          for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
            for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
              # TeX: end smem_in_order
              # TeX: summary
              # Per-thread accum code
              for m0 in seq(0, M0):
                for n0 in seq(0, N0):
                  for k0 in seq(0, K0):
                    accum[m1, n1, m0, n0] += (A_smem[m1 * M0 + m0, k0]
                                              * B_smem[k0, n1 * N0 + n0])
              # TeX: begin smem_in_order
          # TeX: end smem_in_order[0]
          # TeX: color line smem_in_order[1]
        # yyyyy ............................
          # TeX: color line smem_in_order[2]
        # yyyyy
          # TeX: color line smem_in_order[4]
          #                    rrrrrrrrrrrrr
          Fence(cuda_in_order, cuda_in_order)  # __syncthreads()
          # TeX: begin smem_in_order[0]
        # TeX: end smem_in_order
        for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
          for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
            for m0 in seq(0, M0):
              for n0 in seq(0, N0):
                C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] = accum[m1, n1, m0, n0]


nyc25_gemm_smem_in_order = simplify(nyc25_gemm_smem_in_order)


# TeX: version cp_async 2
# TeX: version ring_todo 4


@proc
def nyc25_gemm_smem_cp_async(M: size, N: size, K: size, A: f32[M, K] @ CudaGmemLinear, B: f32[K, N] @ CudaGmemLinear, C: f32[M, N] @ CudaGmemLinear):
  assert M % M1 == 0
  assert N % N1 == 0
  assert K % K0 == 0
  with CudaDeviceFunction(blockDim=256):
    for m2 in cuda_tasks(0, M / M1):
      for n2 in cuda_tasks(0, N / N1):
        accum: f32[M1/M0, N1/N0, M0, N0] @ CudaRmem
        for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
          for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
            for m0 in seq(0, M0):
              for n0 in seq(0, N0):
                accum[m1, n1, m0, n0] = 0
        # TeX: begin cp_async[1] ring_todo
        # TeX: color line cp_async[1] ring_todo
        #   bb
        for k1 in seq(0, K / K0):
          # TeX: color line ring_todo[1]
        # rrrrrr  rrrr      r
          A_smem: f32[M1, K0] @ CudaSmemLinear  # CUDA shared memory
          # TeX: color line ring_todo[1]
        # rrrrrr  rrrr      r
          B_smem: f32[K0, N1] @ CudaSmemLinear  # CUDA shared memory
          # TeX: color line cp_async[1]
          #              rrrrrrrrrrrrr
          # TeX: end cp_async[1]
          # TeX: summary
          # Fill A_smem, B_smem using cp.async instructions
          # TeX: summary
          # This is running on the Sm80_cp_async timeline
          # TeX: begin cp_async[0]
          # TeX: color line *
          #   gg
          for m1 in cuda_threads(0, M1 / M0, unit=16 * cuda_thread):
            # TeX: color line *
            #   gg
            for m0 in seq(0, M0):
              # TeX: color line *
              #   gg
              for i1 in cuda_threads(0, K0, unit=cuda_thread):
              # TeX: color line cp_async
              # rrrrrrrrrrrrrrrrr        .........................     .......................................................  .......
              # TeX: color line ring_todo
              #                          .........................     .......................................................  .......
                Sm80_cp_async_f32(A_smem[m1 * M0 + m0, i1 : i1 + 1], A[m2 * M1 + m1 * M0 + m0, k1 * K0 + i1 : k1 * K0 + i1 + 1], size=1)  # Exo's name for cp.async
          # TeX: color line *
          #   vv
          for i0 in cuda_threads(0, K0, unit=16 * cuda_thread):
            # TeX: remark ring_todo
            # ...
            # TeX: end ring_todo
            # TeX: color line *
            #   vv
            for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
              # TeX: color line *
              #   vv
              for n0 in seq(0, N0):
              # TeX: color line cp_async
              # rrrrrrrrrrrrrrrrr        ...................................     .................................................................  .......
              # TeX: color line ring_todo
              #                          ...................................     .................................................................  .......
                Sm80_cp_async_f32(B_smem[i0, n1 * N0 + n0 : n1 * N0 + n0 + 1], B[k1 * K0 + i0, n2 * N1 + n1 * N0 + n0 : n2 * N1 + n1 * N0 + n0 + 1], size=1)  # Exo's name for cp.async
            # TeX: end cp_async[0]
            # TeX: begin cp_async[1] ring_todo
          # TeX: color line cp_async[1]
        # yyyyy rrrrrrrrrrrrr
          # TeX: color line ring_todo[3]
        # rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
          Fence(Sm80_cp_async, cuda_in_order)
          # TeX: color line ring_todo[2]
        # rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
          for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
            for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
              # Per-thread accum code
              # This is running on the cuda_in_order timeline
              # TeX: end ring_todo
              for m0 in seq(0, M0):
                for n0 in seq(0, N0):
                  # TeX: color line cp_async[1]
                  #   bb
                  for k0 in seq(0, K0):
                    accum[m1, n1, m0, n0] += (A_smem[m1 * M0 + m0, k0]
                                              * B_smem[k0, n1 * N0 + n0])
          # TeX: begin ring_todo
          # TeX: color line cp_async[1]
        # yyyyy                rrrrrrrrrrrrr
          # TeX: color line ring_todo[3]
        # rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
          Fence(cuda_in_order, Sm80_cp_async)
          # TeX: end cp_async[1] ring_todo
        for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
          for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
            for m0 in seq(0, M0):
              for n0 in seq(0, N0):
                C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] = accum[m1, n1, m0, n0]


nyc25_gemm_smem_cp_async = simplify(nyc25_gemm_smem_cp_async)


@proc
def nyc25_gemm_ring(M: size, N: size, K: size, A: f32[M, K] @ CudaGmemLinear, B: f32[K, N] @ CudaGmemLinear, C: f32[M, N] @ CudaGmemLinear):
  assert M % M1 == 0
  assert N % N1 == 0
  assert K % K0 == 0
  with CudaDeviceFunction(blockDim=256):
    for m2 in cuda_tasks(0, M / M1):
      for n2 in cuda_tasks(0, N / N1):
        accum: f32[M1/M0, N1/N0, M0, N0] @ CudaRmem
        for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
          for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
            for m0 in seq(0, M0):
              for n0 in seq(0, N0):
                accum[m1, n1, m0, n0] = 0
        # TeX: version ring 2
        # TeX: version split 3
        # TeX: begin split
        # TeX: color line split[0]
      # yyy
        raw: barrier @ CudaMbarrier
      # TeX: color line split[0]
      # yyy
        war: barrier(raw) @ CudaMbarrier
        # TeX: begin ring[0]
        # TeX: color line ring[0] split[2]
      # gggggg      b
        A_smem: f32[3, M1, K0] @ CudaSmemLinear  # CUDA shared memory
        # TeX: color line ring[0] split[2]
      # vvvvvv      b
        B_smem: f32[3, K0, N1] @ CudaSmemLinear  # CUDA shared memory
        # TeX: begin ring[1]
        # TeX: color line ring[0]
        #   bb
        # TeX: color line ring[1]
        #   bb                  rrr
        for k1 in seq(0, K / K0 + 1):
          # TeX: end ring[0]
          # TeX: color line ring[1]
          #  rrrrrrrrrrr
          if k1 < K / K0:
            # TeX: end ring[1]
            # TeX: color line split[2]
          # yyyyy yyy  rrrrrrrrrrrrr  bb
            Await(war, Sm80_cp_async, ~3)
            # TeX: end split
            # TeX: begin ring[0]
            # TeX: summary
            # Fill A_smem, B_smem using cp.async instructions
            # TeX: summary
            # This is running on the Sm80_cp_async timeline
            # TeX: color line *
            #   gg
            for m1 in cuda_threads(0, M1 / M0, unit=16 * cuda_thread):
              # TeX: color line *
              #   gg
              for m0 in seq(0, M0):
                # TeX: color line *
                #   gg
                for i1 in cuda_threads(0, K0, unit=cuda_thread):
                # TeX: color line *
                # rrrrrrrrrrrrrrrrr gggggg bbbbbb  .........................     .......................................................  .......
                  Sm80_cp_async_f32(A_smem[k1 % 3, m1 * M0 + m0, i1 : i1 + 1], A[m2 * M1 + m1 * M0 + m0, k1 * K0 + i1 : k1 * K0 + i1 + 1], size=1)
            # TeX: color line *
            #   vv
            for i0 in cuda_threads(0, K0, unit=16 * cuda_thread):
              # TeX: color line *
              #   vv
              for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
                # TeX: color line *
                #   vv
                for n0 in seq(0, N0):
                # TeX: color line *
                # rrrrrrrrrrrrrrrrr vvvvvv bbbbbb  ...................................     .................................................................  .......
                  Sm80_cp_async_f32(B_smem[k1 % 3, i0, n1 * N0 + n0 : n1 * N0 + n0 + 1], B[k1 * K0 + i0, n2 * N1 + n1 * N0 + n0 : n2 * N1 + n1 * N0 + n0 + 1], size=1)
          # TeX: end ring[0]
          # TeX: begin split
          # TeX: color line split[1]
          # yyyyyy rrrrrrrrrrrrr        yyy
            Arrive(Sm80_cp_async, 1) >> raw
          # TeX: begin ring[1]
          # TeX: color line ring[1]
          #  rrrrrrrrrrr
          if k1 >= 1:
          # TeX: end ring[1]
            # TeX: color line split[1]
          # yyyyy yyy
            Await(raw, cuda_in_order, ~0)
            # TeX: begin ring[1]
            for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
              for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
                # TeX: end split
                # TeX: summary
                # Per-thread accum code
                # TeX: summary
                # This is running on the cuda_in_order timeline
                for m0 in seq(0, M0):
                  for n0 in seq(0, N0):
                    for k0 in seq(0, K0):
                      accum[m1, n1, m0, n0] += (
                          # TeX: color line ring[1]
                          #       bbbbbbbbbbb
                          A_smem[(k1 - 1) % 3, m1 * M0 + m0, k0]
                            # TeX: color line ring[1]
                            #       bbbbbbbbbbb
                          * B_smem[(k1 - 1) % 3, k0, n1 * N0 + n0])
              # TeX: end ring[1]
            # TeX: begin split
            # TeX: color line split[2]
          # yyyyyy                   yyyyyy
            Arrive(cuda_in_order, 1) >> war
            # TeX: end split

        for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
          for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
            for m0 in seq(0, M0):
              for n0 in seq(0, N0):
                C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] = accum[m1, n1, m0, n0]
        Fence(cuda_in_order, cuda_in_order)  # __syncthreads()


nyc25_gemm_ring = simplify(nyc25_gemm_ring)


"""
# TeX: version cp_async_pseudocode 1
def cp_async_pseudocode():
    # TeX: begin cp_async_pseudocode[0]
    # TeX: color line *
    #        rrrrrrrr
    cp.async(smem_dst, gmem_src)
    # Do non-dependent work here
    # MUST NOT reference smem_dst
    wait_for_cp_async()
    # TeX: color line *
    #       rrrrrrrr
    consume(smem_dst)
    # TeX: end cp_async_pseudocode[0]
"""


def tensor_core():
  # TeX: version tensor_core_instr 2
  # TeX: begin tensor_core_instr[0]
  # TeX: color line *
  #   g
  for m in seq(0, M_MMA):
    # TeX: color line *
    #   v
    for n in seq(0, N_MMA):
      # TeX: color line *
      #   b
      for k in seq(0, K_MMA):
        # TeX: color line *
        # g v       g b      b v
        C[m,n] += A[m,k] * B[k,n]
        # TeX: end tensor_core_instr[0]

  # TeX: begin tensor_core_instr[1]
  # TeX: color line *
  #               .     .     .
  Sm80_mma_tf32(C[:], A[:], B[:])
  # TeX: end tensor_core_instr[1]
