# TeX: version instr_class 3

# TeX: begin instr_class[0]
@instr
class Sm80_mma_load_a_tf32:
    # PARSED as Exo code; specifies instr behavior for LoopIR_unification
    def behavior(
        K: size,
        rmem: [f32][16, K] @ Sm80_RmemMatrixA,
        smem: [f32][16, K] @ CudaSmemLinear,
    ):
        # Essentially, this is what would have gone into the function body of an old-style instr
        for m in seq(0, 16):
            for k in seq(0, K):
                rmem[m, k] = smem[m, k]

    # EXECUTED as Python code; configures information needed for
    # compiler/codegen, and synchronization checking.
    def instance(self, K):
        # Essentially, this is what would have gone into the @instr(...) for old-style instr
        self.instr_format = # ...
        self.c_includes = # ...
        # ... but also a lot of new information
        self.coll_unit = cuda_warp
        # ...
# TeX: end instr_class[0]

# TeX: begin instr_class[1]
# TeX: color line *
#                                     ggg
concrete_instr = Sm80_mma_load_a_tf32(K=8)

# Behavior: substitute K=8, delete K argument
# TeX: color line *
#                                  g
def concrete_instr(rmem: [f32][16, 8] @ Sm80_RmemMatrixA,
# TeX: color line *
#                                  g
                   smem: [f32][16, 8] @ CudaSmemLinear,
    for m in seq(0, 16):
        for k in seq(0, 8):
            rmem[m, k] = smem[m, k]
# TeX: end instr_class[1]

# TeX: begin instr_class[2]
Sm80_mma_load_a_tf32(A_rmem[k_seq,:,:],  # Runtime: RMEM dst
                     A_smem[1 - k1 % 2,  # Runtime: SMEM src
                            mw*Mw + m_seq*16 : mw*Mw + (m_seq+1)*16,
                            k_seq*MMA_K:(k_seq+1)*MMA_K],
# TeX: color line *
#                    ggggggg
                     K=MMA_K)  # Template: K
# TeX: end instr_class[2]


# TeX: version avx_example 1
# TeX: begin avx_example
# TeX: color line *
#        yyyy  b                                  b
foo: f32[4, 6, 8] @ AVX2  # Expected vector size: 8
# Lowered C++
# TeX: color line *
# bbbbbb     y  y
  __m256 foo[4][6];
# TeX: end avx_example

# TeX: version warp_example 1
# TeX: begin warp_example
# TeX: color line *
#        rrrr  y  bbbbb                                              bbbbbb
foo: f32[2, 4, 6, 8, 16] @ Sm80_RmemMatrixA  # Expected matrix tile: 8 x 16
for mw in cuda_threads(0, 2, unit=4 * cuda_warp):
    for nw in cuda_threads(0, 4, unit=cuda_warp):
        for array_idx in seq(0, 6):
            # Ownership of foo distributed to 2 x 4 grid of warps
            # TeX: color line *
            #   rr  rr
            foo[mw, nw, array_idx, :, :]
# Lowered C++
# TeX: color line *
#              y  b                             rrrrr
  unsigned foo[6][4];  // Executed uniformly by 2 x 4 grid of warps
# TeX: color line *
                        # bbbbbb               b
                       // 8 x 16 matrix tile = 4 registers per thread (32 threads/warp total)
# TeX: end warp_example

# TeX: version avx_window 1
# TeX: begin avx_window
# TeX: color line *
#          bbbbbbbb
src: [f32][packed:8] @ AVX2
# TeX: end avx_window

# TeX: version tma_window 2
# TeX: begin tma_window[0]
# TeX: color line *
#          yyyyyyyyyyyyyyyyyyyyyyyy
src: [f32][strided:box0, dense:box1] @ Sm90_CUtensorMap
# TeX: end tma_window[0]
# TeX: begin tma_window[1]
# TeX: color line *
#                           bbbbbbbb                          yyyyyyy
# Hardware defines swizzled 8 x box1 format; SMEM dst must be densely packed
# TeX: color line *
#                   rrrrrrr
# TMA multicasts to 2 CTAs' shared memory
# TeX: color line *
#          rrrrrrrrrrrrr  yyyyyyyyyyyyy  bbbbbbbbbbbbbb
dst: [f32][distributed:2, dense: box0/8, packed:8, box1] @ Sm90_CUtensorMap
# TeX: end tma_window[1]



# TeX: version pointer_strides 1
# TeX: begin pointer_strides
foo = bar[4:8, 6, :]  # Freely mix points/intervals

struct <sname>
{
    float * data;
    int_fast32_t strides[...];
};
float alloc[dim0 * dim1 * ... ];
# TeX: end pointer_strides

# TeX: version register_alloc 1
# TeX: begin register_alloc
float alloc[dim0][dim1][...];
# TeX: end register_alloc

# TeX: version pointer_offsets 1
# TeX: begin pointer_offsets
foo = bar[4:8, 6:, :]  # Intervals only
struct <sname>
{
    <pointer type> data;
    int32_t offsets[...];
};
# TeX: end pointer_offsets
