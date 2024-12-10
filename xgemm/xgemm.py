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


def schedule_gemm(proc, new_name, use_cuda):
    proc = rename(proc, new_name)
    proc = divide_loop(proc, "m", m_tile, ("mo", "mi"), perfect = True)
    proc = divide_loop(proc, "n", n_tile, ("no", "ni"), perfect = True)
    proc = divide_loop(proc, "k", k_tile, ("ko", "ki"), perfect = True)
    proc = reorder_loops(proc, "mi no")
    c_accum_alloc = proc.find("accum : f32")
    c_accum_zero = proc.find("accum = 0")
    c_accum_reduce = proc.find("accum += _")
    c_accum_export = proc.find("_ = accum")
    proc = expand_dim(proc, c_accum_alloc, n_tile, 'ni')
    proc = expand_dim(proc, c_accum_alloc, m_tile, 'mi')
    proc = lift_alloc(proc, c_accum_alloc, n_lifts = 2)
    proc = fission(proc, c_accum_zero.after(), n_lifts = 2)
    proc = fission(proc, c_accum_export.before(), n_lifts = 2)
    proc = reorder_loops(proc, "ni ko")
    proc = reorder_loops(proc, "mi ko")

    c_tile_reduce = proc.forward(c_accum_reduce).parent().parent().parent()
    proc = stage_mem(proc, c_tile_reduce, f"A[mo*{m_tile}:(mo+1)*{m_tile}, ko*{k_tile}:(ko+1)*{k_tile}]", "A_tile")
    proc = stage_mem(proc, c_tile_reduce, f"B[ko*{k_tile}:(ko+1)*{k_tile}, no*{n_tile}:(no+1)*{n_tile}]", "B_tile")
    proc = simplify(proc)

    if use_cuda:
        proc = set_memory(proc, c_accum_alloc, CudaRegisters)
        proc = set_memory(proc, "A_tile", CudaShared)
        proc = set_memory(proc, "B_tile", CudaShared)
        
        proc = set_loop_mode(proc, "mo", exo.loop_modes.CudaBlocks(m_tile * n_tile))
        proc = set_loop_mode(proc, "no", exo.loop_modes.CudaBlocks())
        proc = set_loop_mode(proc, "i0 #0", exo.loop_modes.cuda_threads)
        proc = set_loop_mode(proc, "i1 #0", exo.loop_modes.cuda_threads)
        proc = set_loop_mode(proc, "i0 #1", exo.loop_modes.cuda_threads)
        proc = set_loop_mode(proc, "i1 #1", exo.loop_modes.cuda_threads)
        proc = set_loop_mode(proc, "mi #0", exo.loop_modes.cuda_threads)
        proc = set_loop_mode(proc, "mi #1", exo.loop_modes.cuda_threads)
        proc = set_loop_mode(proc, "mi #2", exo.loop_modes.cuda_threads)
        proc = set_loop_mode(proc, "ni #0", exo.loop_modes.cuda_threads)
        proc = set_loop_mode(proc, "ni #1", exo.loop_modes.cuda_threads)
        proc = set_loop_mode(proc, "ni #2", exo.loop_modes.cuda_threads)
        
        proc = insert_fence(proc, c_tile_reduce.before(), exo.sync_types.cuda_syncthreads)
        proc = insert_fence(proc, c_tile_reduce.after(), exo.sync_types.cuda_syncthreads)
    else:
        proc = set_memory(proc, c_accum_alloc, ExampleAcceleratorTile)
        proc = set_memory(proc, "A_tile", ExampleAcceleratorTile)
        proc = set_memory(proc, "B_tile", ExampleAcceleratorTile)
        proc = replace(proc, c_tile_reduce, example_mma_tile)
    proc = simplify(proc)
    return proc

exo_cpu_gemm = schedule_gemm(original_gemm, "exo_cpu_gemm", False)
exo_cuda_gemm = schedule_gemm(original_gemm, "exo_cuda_gemm", True)


