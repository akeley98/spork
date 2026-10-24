"""Code listing source for amspork3 (main.tex and decks); must pass exocc.

Listing directive format: spork/docs/code_to_tex/README.md ("Source format").
"""
from __future__ import annotations

from exo import *
from exo.scalars import f16, bf16, f32
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *

# TeX: listing intro_cp_async 1
# TeX: listing intro_mma 1
M = 64
N = 256
K = 32
@proc
def instr_usage_example(src: f32[4] @ CudaGmemLinear):
  with CudaDeviceFunction(blockDim=128):
    for task in cuda_tasks(0, 1):
      # I need a pile of BS code snippets to get the idea across.
      # Everything is at the same indent, in its own 1 thread or 1 warpgroup "loop".
      dst: f32[4] @ CudaSmemLinear
      for tid in cuda_threads(0, 1):
        # Manual loop for cp.async vs cp.async instr
        # TeX: begin intro_cp_async
        for i in seq(0, 4):
          dst[i] = src[i]
        # Replace with 4-at-once $\textbf{async}$ copy
        Sm80_cp_async_1d(
        # TeX: color intro_cp_async elide "dst=f32, src=f32, size0=4"
          dst[:], src[:], dst=f32, src=f32, size0=4)
        # TeX: end intro_cp_async
        C: f32[M, N] @ CudaRmemLinear
        A: f32[M, K] @ CudaRmemLinear
        B: f32[N, K] @ CudaRmemLinear
        # Manual loops for wgmma
        # TeX: begin intro_mma
        for m in seq(0, M):
          for n in seq(0, N):
            for k in seq(0, K):
              C[m, n] += (
                A[m, k] * B[n, k])
        # TeX: end intro_mma
      C: f32[4, 16, N] @ Sm90_TkRmemTileD(N)
      A: f32[M, K] @ Sm90_SmemSwizzled(128)
      B: f32[N, K] @ Sm90_SmemSwizzled(128)
      for wg in cuda_threads(0, 1, unit=cuda_warpgroup):
        # TeX: begin intro_mma
        # Replace with tensor core
        Sm90_tk_mma_row_col(
          # TeX: color intro_mma elide ":, :, :" ":, :" 3:1 ":, :" 3:2 "D=f32, A=f32, B=f32, N=N, K=K"
          C[:, :, :], A[:, :], B[:, :], D=f32, A=f32, B=f32, N=N, K=K)
        # TeX: end intro_mma
      Fence(Sm80_generic, cuda_in_order)

del M
del N
del K


# TeX: listing intra_thread 2
@proc
def intra_thread_example(src: f32[4] @ CudaGmemLinear):
  with CudaDeviceFunction(blockDim=128):
    for task in cuda_tasks(0, 1):
      dst: f32[4] @ CudaSmemLinear
      for tid in cuda_threads(0, 1):
        whatever: f32 @ CudaRmemLinear
        # TeX: begin intra_thread
        # $\textbf{Single threaded}$ code!
        #
        # 4-at-once $\textbf{async}$ copy
        Sm80_cp_async_1d(
        # TeX: color intra_thread elide "dst=f32, src=f32, size0=4"
        # TeX: color intra_thread[0] red "dst[:]"
          dst[:], src[:], dst=f32, src=f32, size0=4)
        # TeX: remark intra_thread[0]
        # No good, didn't wait for copy to retire
        # TeX: end intra_thread[0]
        # TeX: color intra_thread[1] green "Sm80_cp_async"
        Fence(Sm80_cp_async, cuda_in_order)
        # TeX: begin intra_thread[0]
        # TeX: color intra_thread[0] red "dst[0]"
        whatever += dst[0]
        # TeX: remark intra_thread[1]
        # (`Fence` will be explained later)
        # TeX: end intra_thread
