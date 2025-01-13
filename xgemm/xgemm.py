from __future__ import annotations
import exo
from exo import *
from exo.stdlib.scheduling import *
from exo.spork.cuda_memory import *

m_tile = 16
n_tile = 16
k_tile = 8

class ExampleAcceleratorTile(DRAM):
    @classmethod
    def global_(cls):
        return f"""#include <math.h>
// Example: placeholder for an actual matrix accelerator intrinsic
static void example_mma_tile(float C_tile[], const float A_tile[], const float B_tile[])
{{
    for (int m = 0; m < {m_tile}; ++m)
        for (int n = 0; n < {n_tile}; ++n)
            for (int k = 0; k < {k_tile}; ++k)
                C_tile[m * {n_tile} + n] = fmaf(A_tile[m * {k_tile} + k], B_tile[k * {n_tile} + n], C_tile[m * {n_tile} + n]);
}}
"""

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        size = 1
        for n in shape:
            if not n.isdecimal():
                raise ValueError(f"{srcinfo}: {n} must be constant decimal")
            size *= int(n)
        return f"{prim_type} {new_name}[{size}];  // Example: placeholder accelerator memory. Real accelerators would likely need custom load/store instructions as well"

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""

    @classmethod
    def window(cls, basetype, baseptr, indices, strides, srcinfo):
        for n in indices:
            if not n.isdecimal() or int(n) != 0:
                raise ValueError(f"{srcinfo}: all indices for window must be 0")
        return baseptr

@instr("example_mma_tile({C_tile_data}, {A_tile_data}, {B_tile_data});")
def example_mma_tile(C_tile: [f32][m_tile,n_tile] @ ExampleAcceleratorTile,
                     A_tile: [f32][m_tile,k_tile] @ ExampleAcceleratorTile,
                     B_tile: [f32][k_tile,n_tile] @ ExampleAcceleratorTile):
    for m in seq(0, m_tile):
        for n in seq(0, n_tile):
            for k in seq(0, k_tile):
                C_tile[m,n] += A_tile[m,k] * B_tile[k,n]

@proc
def original_gemm(M: size, N: size, K: size, A: f32[M,K], B: f32[K,N], C: f32[M,N]):
    # Avoid requiring predication
    assert M % m_tile == 0
    assert N % n_tile == 0
    assert K % k_tile == 0

    for m in seq(0, M):
        for n in seq(0, N):
            accum : f32
            accum = 0
            for k in seq(0, K):
                accum += A[m,k] * B[k,n]
            C[m,n] = accum


def schedule_gemm(p, new_name, use_cuda):
    p = rename(p, new_name)
    p = divide_loop(p, "m", m_tile, ("mo", "mi"), perfect = True)
    p = divide_loop(p, "n", n_tile, ("no", "ni"), perfect = True)
    p = divide_loop(p, "k", k_tile, ("ko", "ki"), perfect = True)
    p = reorder_loops(p, "mi no")

    c_accum_alloc = p.find("accum : f32")  # Cursor
    p = expand_dim(p, c_accum_alloc, n_tile, 'ni')
    p = expand_dim(p, c_accum_alloc, m_tile, 'mi')
    p = lift_alloc(p, c_accum_alloc, n_lifts = 2)

    c_accum_zero = p.find("accum = 0")
    c_accum_export = p.find("_ = accum")
    p = fission(p, c_accum_zero.after(), n_lifts = 2)
    p = fission(p, c_accum_export.before(), n_lifts = 2)
    p = reorder_loops(p, "ni ko")
    p = reorder_loops(p, "mi ko")

    c_tile_reduce = p.find("accum += _").parent().parent().parent()
    p = stage_mem(p, c_tile_reduce,
                  f"A[mo*{m_tile}:(mo+1)*{m_tile}, ko*{k_tile}:(ko+1)*{k_tile}]", "A_tile")
    p = stage_mem(p, c_tile_reduce,
                  f"B[ko*{k_tile}:(ko+1)*{k_tile}, no*{n_tile}:(no+1)*{n_tile}]", "B_tile")
    p = simplify(p)

    print(p)
    if use_cuda:
        # Temporary syntax: wrap entire body as cuda device function
        # with blockDim = m_tile * n_tile
        p = tmp_add_with(p, p.body(), CudaDeviceFunction(blockDim = m_tile * n_tile))

        p = set_memory(p, c_accum_alloc, CudaRmem)
        p = set_memory(p, "A_tile", CudaSmem)
        p = set_memory(p, "B_tile", CudaSmem)

        # "x #n" means n-th loop with x as iteration variable
        p = set_loop_mode(p, "mo", exo.loop_modes.cuda_blocks)
        p = set_loop_mode(p, "no", exo.loop_modes.cuda_blocks)
        p = set_loop_mode(p, "i0 #0", exo.loop_modes.cuda_threads)
        p = set_loop_mode(p, "i1 #0", exo.loop_modes.cuda_threads)
        p = set_loop_mode(p, "i0 #1", exo.loop_modes.cuda_threads)
        p = set_loop_mode(p, "i1 #1", exo.loop_modes.cuda_threads)
        p = set_loop_mode(p, "mi #0", exo.loop_modes.cuda_threads)
        p = set_loop_mode(p, "mi #1", exo.loop_modes.cuda_threads)
        p = set_loop_mode(p, "mi #2", exo.loop_modes.cuda_threads)
        p = set_loop_mode(p, "ni #0", exo.loop_modes.cuda_threads)
        p = set_loop_mode(p, "ni #1", exo.loop_modes.cuda_threads)
        p = set_loop_mode(p, "ni #2", exo.loop_modes.cuda_threads)

        p = insert_fence(p, c_tile_reduce.before(), exo.sync_types.cuda_syncthreads)
        p = insert_fence(p, c_tile_reduce.after(), exo.sync_types.cuda_syncthreads)
    else:
        p = set_memory(p, c_accum_alloc, ExampleAcceleratorTile)
        p = set_memory(p, "A_tile", ExampleAcceleratorTile)
        p = set_memory(p, "B_tile", ExampleAcceleratorTile)
        p = replace(p, c_tile_reduce, example_mma_tile)
    print(p)
    return p

exo_cpu_gemm = schedule_gemm(original_gemm, "exo_cpu_gemm", False)
exo_cuda_gemm = schedule_gemm(original_gemm, "exo_cuda_gemm", True)
