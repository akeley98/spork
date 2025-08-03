from __future__ import annotations

from exo import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.stdlib.scheduling import *

@proc
def nyc25_gemm_cpu(M: size, N: size, K: size, A: f32[M, K] @ DRAM, B: f32[K, N] @ DRAM, C: f32[M, N] @ DRAM):
  for m in seq(0, M):
    for n in seq(0, N):
      C[m, n] = 0
      for k in seq(0, K):
        C[m, n] += A[m, k] * B[k, n]


M1 = 192
M0 = 12

N1 = 256
N0 = 16

K0 = 16

@proc
def nyc25_gemm_m_divide_loop(M: size, N: size, K: size, A: f32[M, K], B: f32[K, N], C: f32[M, N]):
  assert M % M1 == 0
  for m2 in seq(0, M / M1):
    for m1 in seq(0, M1 / M0):
      for m0 in seq(0, M0):
        for n in seq(0, N):
          C[m2 * M1 + m1 * M0 + m0, n] = 0
          for k in seq(0, K):
            C[m2 * M1 + m1 * M0 + m0, n] += A[m2 * M1 + m1 * M0 + m0, k] * B[k, n]


@proc
def nyc25_gemm_m_divide_loop(M: size, N: size, K: size, A: f32[M, K], B: f32[K, N], C: f32[M, N]):
  assert M % M1 == 0
  assert N % N1 == 0
  for m2 in seq(0, M / M1):
    for m1 in seq(0, M1 / M0):
      for m0 in seq(0, M0):
        for n2 in seq(0, N / N1):
          for n1 in seq(0, N1 / N0):
            for n0 in seq(0, N0):
              C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] = 0
              for k in seq(0, K):
                C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] += A[m2 * M1 + m1 * M0 + m0, k] * B[k, n2 * N1 + n1 * N0 + n0]


@proc
def nyc25_gemm_reorder_loop(M: size, N: size, K: size, A: f32[M, K] @ DRAM, B: f32[K, N] @ DRAM, C: f32[M, N] @ DRAM):
  assert M % M1 == 0
  assert N % N1 == 0
  for m2 in seq(0, M / M1):
    for n2 in seq(0, N / N1):
      for m1 in seq(0, M1 / M0):
        for n1 in seq(0, N1 / N0):
          for m0 in seq(0, M0):
            for n0 in seq(0, N0):
              C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] = 0
              for k in seq(0, K):
                C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] += A[m2 * M1 + m1 * M0 + m0, k] * B[k, n2 * N1 + n1 * N0 + n0]


@proc
def nyc25_gemm_simple_gpu(M: size, N: size, K: size, A: f32[M, K] @ CudaGmemLinear, B: f32[K, N] @ CudaGmemLinear, C: f32[M, N] @ CudaGmemLinear):
  assert M % M1 == 0
  assert N % N1 == 0
  with CudaDeviceFunction(blockDim=256):
    for m2 in cuda_tasks(0, M / M1):
      for n2 in cuda_tasks(0, N / N1):
        for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
          for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
            for m0 in seq(0, M0):
              for n0 in seq(0, N0):
                C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] = 0
                for k in seq(0, K):
                  C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] += A[m2 * M1 + m1 * M0 + m0, k] * B[k, n2 * N1 + n1 * N0 + n0]

nyc25_gemm_simple_gpu = simplify(nyc25_gemm_simple_gpu)




@proc
def nyc25_gemm_smem_broken(M: size, N: size, K: size, A: f32[M, K] @ CudaGmemLinear, B: f32[K, N] @ CudaGmemLinear, C: f32[M, N] @ CudaGmemLinear):
  assert M % M1 == 0
  assert N % N1 == 0
  assert K % K0 == 0
  with CudaDeviceFunction(blockDim=256):
    for m2 in cuda_tasks(0, M / M1):
      for n2 in cuda_tasks(0, N / N1):
        for k1 in seq(0, K / K0):
          A_smem: f32[M1, K0] @ CudaSmemLinear
          B_smem: f32[K0, N1] @ CudaSmemLinear
          for i0 in seq(0, M1):
            for i1 in seq(0, K0):
              A_smem[i0, i1] = A[m2 * M1 + i0, k1 * K0 + i1]
          for i0 in seq(0, K0):
            for i1 in seq(0, N1):
              B_smem[i0, i1] = B[k1 * K0 + i0, n2 * N1 + i1]
          for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
            for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
              for m0 in seq(0, M0):
                for n0 in seq(0, N0):
                  for k0 in seq(0, K0):
                    C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] += A_smem[m1 * M0 + m0, k0] * B_smem[k0, n1 * N0 + n0]


del nyc25_gemm_smem_broken


@proc
def nyc25_gemm_smem_in_order(M: size, N: size, K: size, A: f32[M, K] @ CudaGmemLinear, B: f32[K, N] @ CudaGmemLinear, C: f32[M, N] @ CudaGmemLinear):
  assert M % M1 == 0
  assert N % N1 == 0
  assert K % K0 == 0
  with CudaDeviceFunction(blockDim=256):
    for m2 in cuda_tasks(0, M / M1):
      for n2 in cuda_tasks(0, N / N1):
        C_reg: f32[M1/M0, N1/N0, M0, N0] @ CudaRmem
        for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
          for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
            for m0 in seq(0, M0):
              for n0 in seq(0, N0):
                C_reg[m1, n1, m0, n0] = 0
        for k1 in seq(0, K / K0):
          A_smem: f32[M1, K0] @ CudaSmemLinear
          B_smem: f32[K0, N1] @ CudaSmemLinear
          for m1 in cuda_threads(0, M1 / M0, unit=16 * cuda_thread):
            for m0 in seq(0, M0):
              for i1 in cuda_threads(0, K0, unit=cuda_thread):
                A_smem[m1 * M0 + m0, i1] = A[m2 * M1 + m1 * M0 + m0, k1 * K0 + i1]
          for i0 in cuda_threads(0, K0, unit=16 * cuda_thread):
            for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
              for n0 in seq(0, N0):
                B_smem[i0, n1 * N0 + n0] = B[k1 * K0 + i0, n2 * N1 + n1 * N0 + n0]
          Fence(cuda_in_order, cuda_in_order)
          for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
            for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
              for m0 in seq(0, M0):
                for n0 in seq(0, N0):
                  for k0 in seq(0, K0):
                    C_reg[m1, n1, m0, n0] += A_smem[m1 * M0 + m0, k0] * B_smem[k0, n1 * N0 + n0]
          Fence(cuda_in_order, cuda_in_order)
        for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
          for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
            for m0 in seq(0, M0):
              for n0 in seq(0, N0):
                C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] = C_reg[m1, n1, m0, n0]


nyc25_gemm_smem_in_order = simplify(nyc25_gemm_smem_in_order)


@proc
def nyc25_gemm_smem_cp_async_no_instr(M: size, N: size, K: size, A: f32[M, K] @ CudaGmemLinear, B: f32[K, N] @ CudaGmemLinear, C: f32[M, N] @ CudaGmemLinear):
  assert M % M1 == 0
  assert N % N1 == 0
  assert K % K0 == 0
  with CudaDeviceFunction(blockDim=256):
    for m2 in cuda_tasks(0, M / M1):
      for n2 in cuda_tasks(0, N / N1):
        C_reg: f32[M1/M0, N1/N0, M0, N0] @ CudaRmem
        for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
          for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
            for m0 in seq(0, M0):
              for n0 in seq(0, N0):
                C_reg[m1, n1, m0, n0] = 0
        for k1 in seq(0, K / K0):
          A_smem: f32[M1, K0] @ CudaSmemLinear
          B_smem: f32[K0, N1] @ CudaSmemLinear
          with CudaAsync(Sm80_cp_async):
            for mS in seq(0, M1 / 64):
              for mT in cuda_threads(0, 64, unit=4 * cuda_thread):
                for kT in cuda_threads(0, K0 / 4, unit=cuda_thread):
                  for kTmp in seq(0, 4):
                    A_smem[mS * 64 + mT, kT * 4 + kTmp] = A[m2 * M1 + mS * 64 + mT, k1 * K0 + kT * 4 + kTmp]
            for kT in cuda_threads(0, 16, unit=16 * cuda_thread):
              for nS in seq(0, N1 / 64):
                for nT in cuda_threads(0, 16, unit=cuda_thread):
                  for nTmp in seq(0, 4):
                    B_smem[kT, nS * 64 + nT * 4 + nTmp] = B[k1 * K0 + kT, n2 * N1 + nS * 64 + nT * 4 + nTmp]
          for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
            for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
              for m0 in seq(0, M0):
                for n0 in seq(0, N0):
                  for k0 in seq(0, K0):
                    C_reg[m1, n1, m0, n0] += A_smem[m1 * M0 + m0, k0] * B_smem[k0, n1 * N0 + n0]
        for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
          for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
            for m0 in seq(0, M0):
              for n0 in seq(0, N0):
                C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] = C_reg[m1, n1, m0, n0]


del nyc25_gemm_smem_cp_async_no_instr


@proc
def nyc25_gemm_smem_cp_async(M: size, N: size, K: size, A: f32[M, K] @ CudaGmemLinear, B: f32[K, N] @ CudaGmemLinear, C: f32[M, N] @ CudaGmemLinear):
  assert M % M1 == 0
  assert N % N1 == 0
  assert K % K0 == 0
  with CudaDeviceFunction(blockDim=256):
    for m2 in cuda_tasks(0, M / M1):
      for n2 in cuda_tasks(0, N / N1):
        C_reg: f32[M1/M0, N1/N0, M0, N0] @ CudaRmem
        for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
          for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
            for m0 in seq(0, M0):
              for n0 in seq(0, N0):
                C_reg[m1, n1, m0, n0] = 0
        for k1 in seq(0, K / K0):
          A_smem: f32[M1, K0] @ CudaSmemLinear
          B_smem: f32[K0, N1] @ CudaSmemLinear
          with CudaAsync(Sm80_cp_async):
            for mS in seq(0, M1 / 64):
              for mT in cuda_threads(0, 64, unit=4 * cuda_thread):
                for kT in cuda_threads(0, K0 / 4, unit=cuda_thread):
                  Sm80_cp_async_f32(
                    A_smem[mS * 64 + mT, kT * 4: kT * 4 + 4],
                    A[m2 * M1 + mS * 64 + mT, k1 * K0 + kT * 4: k1 * K0 + kT * 4 + 4],
                    size=4)
            for kT in cuda_threads(0, 16, unit=16 * cuda_thread):
              for nS in seq(0, N1 / 64):
                for nT in cuda_threads(0, 16, unit=cuda_thread):
                  Sm80_cp_async_f32(
                    B_smem[kT, nS * 64 + nT * 4 : nS * 64 + nT * 4 + 4],
                    B[k1 * K0 + kT, n2 * N1 + nS * 64 + nT * 4 : n2 * N1 + nS * 64 + nT * 4 + 4],
                    size=4)
          Fence(Sm80_cp_async, cuda_in_order)
          for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
            for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
              for m0 in seq(0, M0):
                for n0 in seq(0, N0):
                  for k0 in seq(0, K0):
                    C_reg[m1, n1, m0, n0] += A_smem[m1 * M0 + m0, k0] * B_smem[k0, n1 * N0 + n0]
          Fence(cuda_in_order, Sm80_cp_async)
        for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
          for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
            for m0 in seq(0, M0):
              for n0 in seq(0, N0):
                C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] = C_reg[m1, n1, m0, n0]


nyc25_gemm_smem_cp_async = simplify(nyc25_gemm_smem_cp_async)


@proc
def nyc25_gemm_ring(M: size, N: size, K: size, A: f32[M, K] @ CudaGmemLinear, B: f32[K, N] @ CudaGmemLinear, C: f32[M, N] @ CudaGmemLinear):
  assert M % M1 == 0
  assert N % N1 == 0
  assert K % K0 == 0
  with CudaDeviceFunction(blockDim=256):
    for m2 in cuda_tasks(0, M / M1):
      for n2 in cuda_tasks(0, N / N1):
        C_reg: f32[M1/M0, N1/N0, M0, N0] @ CudaRmem
        for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
          for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
            for m0 in seq(0, M0):
              for n0 in seq(0, N0):
                C_reg[m1, n1, m0, n0] = 0
        A_smem: f32[3, M1, K0] @ CudaSmemLinear
        B_smem: f32[3, K0, N1] @ CudaSmemLinear
        ringbar: barrier @ CudaMbarrier
        for k1 in seq(0, K / K0 + 1):
          if k1 < K / K0:
            with CudaAsync(Sm80_cp_async):
              Await(-ringbar, Sm80_cp_async, ~3)
              for mS in seq(0, M1 / 64):
                for mT in cuda_threads(0, 64, unit=4 * cuda_thread):
                  for kT in cuda_threads(0, K0 / 4, unit=cuda_thread):
                    Sm80_cp_async_f32(
                      A_smem[k1 % 3, mS * 64 + mT, kT * 4: kT * 4 + 4],
                      A[m2 * M1 + mS * 64 + mT, k1 * K0 + kT * 4: k1 * K0 + kT * 4 + 4],
                      size=4)
              for kT in cuda_threads(0, 16, unit=16 * cuda_thread):
                for nS in seq(0, N1 / 64):
                  for nT in cuda_threads(0, 16, unit=cuda_thread):
                    Sm80_cp_async_f32(
                      B_smem[k1 % 3, kT, nS * 64 + nT * 4 : nS * 64 + nT * 4 + 4],
                      B[k1 * K0 + kT, n2 * N1 + nS * 64 + nT * 4 : n2 * N1 + nS * 64 + nT * 4 + 4],
                      size=4)
              Arrive(Sm80_cp_async, 1) >> +ringbar
          if k1 >= 1:
            Await(+ringbar, cuda_in_order, ~0)
            for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
              for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
                for m0 in seq(0, M0):
                  for n0 in seq(0, N0):
                    for k0 in seq(0, K0):
                      C_reg[m1, n1, m0, n0] += A_smem[(k1-1) % 3, m1 * M0 + m0, k0] * B_smem[(k1-1) % 3, k0, n1 * N0 + n0]
            Arrive(cuda_in_order, 1) >> -ringbar

        for m1 in cuda_threads(0, M1 / M0, unit=(N1 / N0) * cuda_thread):
          for n1 in cuda_threads(0, N1 / N0, unit=cuda_thread):
            for m0 in seq(0, M0):
              for n0 in seq(0, N0):
                C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] = C_reg[m1, n1, m0, n0]
        Fence(cuda_in_order, cuda_in_order)


nyc25_gemm_ring = simplify(nyc25_gemm_ring)
