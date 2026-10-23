from __future__ import annotations

# Listing source for the beamer_nyc25_test deck + paper (subset of spork/docs/nyc25.py,
# directives rewritten in the new code_to_tex syntax; see code_to_tex/README.md).

from exo import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.stdlib.scheduling import *

# ---------------------------------------------------------------------------
# cpu: the starting GEMM. Versions 0-3 highlight parts of the full proc;
# version 4 drops the signature (lines removed) and highlights the body.
# ---------------------------------------------------------------------------
if True:
# TeX: listing cpu 5
# TeX: begin cpu[:4]
# TeX: color cpu[0] blue "@proc"
  @proc
  # TeX: color cpu[1] blue "size" 3:*
  def nyc25_gemm_cpu(M: size, N: size, K: size,
  # TeX: color cpu[1] blue "f32[M, K]"
  # TeX: color cpu[2] blue "@ DRAM"
                     A: f32[M, K] @ DRAM,
  # TeX: color cpu[1] blue "f32[K, N]"
  # TeX: color cpu[2] blue "@ DRAM"
                     B: f32[K, N] @ DRAM,
  # TeX: color cpu[1] blue "f32[M, N]"
  # TeX: color cpu[2] blue "@ DRAM"
                     C: f32[M, N] @ DRAM):
    # TeX: begin cpu[4]
    # TeX: color cpu[3:] green "m" "seq"
    for m in seq(0, M):
      # TeX: color cpu[3:] violet "n" "seq"
      for n in seq(0, N):
        # TeX: color cpu[1] blue "f32"
        # TeX: color cpu[2] blue "@ DRAM"
        accum: f32 @ DRAM
        accum = 0
        # TeX: color cpu[3] blue "k" "seq"
        # TeX: color cpu[4] blue "k"
        for k in seq(0, K):
          # TeX: color cpu[4] green "m" violet "n"
          accum += A[m, k] * B[k, n]
        # TeX: color cpu[4] green "m" violet "n"
        C[m, n] = accum
        # TeX: end cpu[4]
# TeX: end cpu[:4]

# ---------------------------------------------------------------------------
# m_divide_loop: meant for \sporkListing[reserve]: the assert appears in
# version 2 only, and with `reserve` nothing below it moves.
# ---------------------------------------------------------------------------
# TeX: listing m_divide_loop 3
# TeX: begin m_divide_loop
M1 = 192
M0 = 12
# TeX: end m_divide_loop
N1 = 256
N0 = 16
K0 = 16

# TeX: summary
# @proc body excerpt below

@proc
def nyc25_gemm_m_divide_loop(M: size, N: size, K: size, A: f32[M, K], B: f32[K, N], C: f32[M, N]):
  # TeX: begin m_divide_loop[2]
  # TeX: color m_divide_loop[2] green "assert M % M1 == 0"
  assert M % M1 == 0
  # TeX: end m_divide_loop[2]
  # TeX: begin m_divide_loop
  # TeX: color m_divide_loop[0] green "m2"
  for m2 in seq(0, M / M1):
    # TeX: color m_divide_loop[0] green "m1"
    for m1 in seq(0, M1 / M0):
      # TeX: color m_divide_loop[0] green "m0"
      for m0 in seq(0, M0):
        for n in seq(0, N):
          accum: f32 @ DRAM
          accum = 0
          for k in seq(0, K):
            accum += (
                # TeX: color m_divide_loop[1] green "m2 * M1 + m1 * M0 + m0"
                A[m2 * M1 + m1 * M0 + m0, k]
              * B[k, n]
            )
          # TeX: color m_divide_loop[1] green "m2 * M1 + m1 * M0 + m0"
          C[m2 * M1 + m1 * M0 + m0, n] = accum
  # TeX: end m_divide_loop


# ---------------------------------------------------------------------------
# loops: the loop nest shown beside the grid tile figure.
# ---------------------------------------------------------------------------
# TeX: listing loops 2

if False:
  # TeX: begin loops
  # TeX: color loops[0] green "cuda_tasks" "M / M1"
  for m2 in cuda_tasks(0, M / M1):
    # TeX: color loops[0] violet "cuda_tasks" "N / N1"
    for n2 in cuda_tasks(0, N / N1):
      # TeX: color loops[1] green "cuda_threads" "M1/M0"
      for m1 in cuda_threads(0, M1/M0,
          unit=16 * cuda_thread):
        # TeX: color loops[1] violet "cuda_threads" "N1/N0"
        for n1 in cuda_threads(0, N1/N0,
              unit=cuda_thread):
          # TeX: color loops[1] green "M0"
          for m0 in seq(0, M0):
            # TeX: color loops[1] violet "N0"
            for n0 in seq(0, N0):
              # Dot products into `accum`
    # Note: N1/N0 = 16
              # TeX: end loops
              pass


# ---------------------------------------------------------------------------
# smem_in_order: code grows (the two Fence lines appear in version 1);
# version 1 elides the Fence arguments.
# ---------------------------------------------------------------------------
# TeX: listing smem_in_order 5


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
        # TeX: color smem_in_order blue "k1"
        for k1 in seq(0, K / K0):
          A_smem: f32[M1, K0] @ CudaSmemLinear  # CUDA shared memory
          B_smem: f32[K0, N1] @ CudaSmemLinear  # CUDA shared memory
          # TeX: color smem_in_order green "m1" "cuda_threads"
          for m1 in cuda_threads(0, M1 / M0, unit=16 * cuda_thread):
            # TeX: color smem_in_order green "m0"
            for m0 in seq(0, M0):
              # TeX: color smem_in_order green "i1" "cuda_threads"
              for i1 in cuda_threads(0, K0, unit=cuda_thread):
                # TeX: color smem_in_order[3] red "A_smem[m1 * M0 + m0, i1] = A[m2 * M1 + m1 * M0 + m0, k1 * K0 + i1]"
                A_smem[m1 * M0 + m0, i1] = A[m2 * M1 + m1 * M0 + m0, k1 * K0 + i1]
          # TeX: color smem_in_order violet "i0" "cuda_threads"
          for i0 in cuda_threads(0, K0, unit=16 * cuda_thread):
            # TeX: color smem_in_order violet "n1" "cuda_threads"
            for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
              # TeX: color smem_in_order violet "n0"
              for n0 in seq(0, N0):
                # TeX: color smem_in_order[3] red "B_smem[i0, n1 * N0 + n0] = B[k1 * K0 + i0, n2 * N1 + n1 * N0 + n0]"
                B_smem[i0, n1 * N0 + n0] = B[k1 * K0 + i0, n2 * N1 + n1 * N0 + n0]
          # TeX: end smem_in_order[0]
          # TeX: color smem_in_order[1:3] yellow "Fence"
          # TeX: color smem_in_order[1] elide "cuda_in_order, cuda_in_order"
          # TeX: color smem_in_order[4] red "cuda_in_order" 2:0
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
          # TeX: color smem_in_order[1:3] yellow "Fence"
          # TeX: color smem_in_order[1] elide "cuda_in_order, cuda_in_order"
          # TeX: color smem_in_order[4] red "cuda_in_order" 2:1
          Fence(cuda_in_order, cuda_in_order)  # __syncthreads()
          # TeX: begin smem_in_order[0]
        # TeX: end smem_in_order
        for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
          for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
            for m0 in seq(0, M0):
              for n0 in seq(0, N0):
                C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] = accum[m1, n1, m0, n0]


nyc25_gemm_smem_in_order = simplify(nyc25_gemm_smem_in_order)


# ---------------------------------------------------------------------------
# ring: ring buffer + loop skew (versions show different lines; extra slides).
# split: split barriers; marks for arrows between Arrive and Await lines.
# ---------------------------------------------------------------------------
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
        # TeX: listing ring 2
        # TeX: listing split 3
        # TeX: begin split
        # TeX: color split[0] yellow "raw"
        # TeX: mark split left start
        raw: barrier[(K/K0) @ ring_buffer_by(3)] @ CudaMbarrier
        # TeX: color split[0] yellow "war"
        war: barrier[(K/K0 + 3) @ ring_buffer_by(3)] @ CudaMbarrierPreArrive(3)
        # TeX: begin ring[0]
        # TeX: color ring[0],split[2] green "A_smem" blue "3"
        A_smem: f32[3, M1, K0] @ CudaSmemLinear  # CUDA shared memory
        # TeX: color ring[0],split[2] violet "B_smem" blue "3"
        B_smem: f32[3, K0, N1] @ CudaSmemLinear  # CUDA shared memory
        # TeX: begin ring[1]
        # TeX: color ring blue "k1"
        # TeX: color ring[1] red "+ 1"
        for k1 in seq(0, K / K0 + 1):
          # TeX: end ring[0]
          # TeX: color ring[1] red "if k1 < K / K0"
          if k1 < K / K0:
            # TeX: end ring[1]
            # TeX: remark! split[2]
            # WAR: slot `k1 % 3` is free once its old readers arrived
            # TeX: color split[2] yellow "Await" "war[k1]"
            # TeX: mark split war_await end
            Await(war[k1], cuda_in_order, 0)
            # TeX: end split
            # TeX: begin ring[0]
            # TeX: summary
            # Fill A_smem, B_smem using cp.async instructions
            # TeX: summary
            # This is running on the Sm80_cp_async timeline
            # TeX: color ring[0] green "m1"
            for m1 in cuda_threads(0, M1 / M0, unit=16 * cuda_thread):
              # TeX: color ring[0] green "m0"
              for m0 in seq(0, M0):
                # TeX: color ring[0] green "i1"
                for i1 in cuda_threads(0, K0, unit=cuda_thread):
                  # TeX: color ring[0] red "Sm80_cp_async_f32" blue "k1 % 3" elide "m2 * M1 + m1 * M0 + m0, k1 * K0 + i1 : k1 * K0 + i1 + 1"
                  Sm80_cp_async_f32(A_smem[k1 % 3, m1 * M0 + m0, i1 : i1 + 1], A[m2 * M1 + m1 * M0 + m0, k1 * K0 + i1 : k1 * K0 + i1 + 1], size=1)
            # TeX: color ring[0] violet "i0"
            for i0 in cuda_threads(0, K0, unit=16 * cuda_thread):
              # TeX: color ring[0] violet "n1"
              for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
                # TeX: color ring[0] violet "n0"
                for n0 in seq(0, N0):
                  # TeX: color ring[0] red "Sm80_cp_async_f32" blue "k1 % 3" elide "k1 * K0 + i0, n2 * N1 + n1 * N0 + n0 : n2 * N1 + n1 * N0 + n0 + 1"
                  Sm80_cp_async_f32(B_smem[k1 % 3, i0, n1 * N0 + n0 : n1 * N0 + n0 + 1], B[k1 * K0 + i0, n2 * N1 + n1 * N0 + n0 : n2 * N1 + n1 * N0 + n0 + 1], size=1)
          # TeX: end ring[0]
          # TeX: begin split
            # TeX: color split[1] yellow "Arrive" "raw[k1]" red "Sm80_cp_async"
            # TeX: mark split raw_arrive end
            Arrive(Sm80_cp_async, 1) >> raw[k1]
          # TeX: begin ring[1]
          # TeX: color ring[1] red "if k1 >= 1"
          if k1 >= 1:
          # TeX: end ring[1]
            # TeX: color split[1] red "`cp.async`"
            # TeX: remark split[1:]
            # RAW: wait for the `cp.async` fill of this slot
            # TeX: color split[1] yellow "Await" "raw[k1]"
            # TeX: mark split raw_await end
            Await(raw[k1], cuda_in_order, 0)
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
                          # TeX: color ring[1] blue "(k1 - 1) % 3"
                          A_smem[(k1 - 1) % 3, m1 * M0 + m0, k0]
                            # TeX: color ring[1] blue "(k1 - 1) % 3"
                          * B_smem[(k1 - 1) % 3, k0, n1 * N0 + n0])
              # TeX: end ring[1]
            # TeX: begin split
            # TeX: color split[2] yellow "Arrive" "war[k1 + 3]"
            # TeX: mark split war_arrive end
            Arrive(cuda_in_order, 1) >> war[k1 + 3]
            # TeX: end split

        for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
          for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
            for m0 in seq(0, M0):
              for n0 in seq(0, N0):
                C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] = accum[m1, n1, m0, n0]
        Fence(cuda_in_order, cuda_in_order)  # __syncthreads()


nyc25_gemm_ring = simplify(nyc25_gemm_ring)
nyc25_gemm_ring.sync_check(M=128, N=128, K=256)
