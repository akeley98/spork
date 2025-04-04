from __future__ import annotations
import exo
from exo import *
from exo.stdlib.scheduling import *
from exo.spork.cuda_memory import *


Sm80_cp_async = actor_kinds.Sm80_cp_async  # Maybe crappy, fixme
wgmma_async = actor_kinds.wgmma_async
tma_to_smem_async = actor_kinds.tma_to_smem_async

Mw = 96
Nw = 64

M1 = 192
N1 = 256  # Does not change gracefully

K0 = 16
MMA_K = 4


@instr
class tmp_cpAsync16B_f32:
    def behavior(smem: [f32][4] @ CudaSmemLinear, gmem: [f32][4] @ CudaGmemLinear):
        assert stride(smem, 0) == 1
        for i in seq(0, 4):
            smem[i] = gmem[i]

    def instance(self):
        self.instr_format = "exo_CudaUtil::Sm80_cpAsync16B(&{smem_data}, &{gmem_data});"
        self.actor_kind = actor_kinds.Sm80_cp_async
        self.access_info["smem"].actor_signature = actor_kinds.sig_Sm80_cp_async
        self.access_info["gmem"].actor_signature = actor_kinds.sig_Sm80_cp_async
        self.cu_util = self.cu_util_src

    cu_util_src = """
EXO_CUDA_INLINE void Sm80_cpAsync16B(void* smem_ptr, const void* gmem_ptr) {
  const int BYTES = 16;
  uint32_t smem = exo_smemU32(smem_ptr);
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], %2;" ::"r"(smem),
      "l"(gmem_ptr),
      "n"(BYTES) : "memory");
}
"""


Sm80_cu_util = r"""
EXO_CUDA_INLINE void Sm80_load_a_k8(unsigned rmem[4], const float* gmem, cuda::std::array<int_fast32_t, 2> element_strides)
{
  const unsigned row_stride = element_strides[0];
  const unsigned col_stride = element_strides[1];
  const unsigned warp_lane = threadIdx.x % 32u;
  const float* gmem_thread_baseaddr = &gmem[warp_lane / 4u * row_stride + warp_lane % 4u * col_stride];
  rmem[0] = __float_as_uint(gmem_thread_baseaddr[0]);
  rmem[1] = __float_as_uint(gmem_thread_baseaddr[8 * row_stride]);
  rmem[2] = __float_as_uint(gmem_thread_baseaddr[4 * col_stride]);
  rmem[3] = __float_as_uint(gmem_thread_baseaddr[8 * row_stride + 4 * col_stride]);
}

EXO_CUDA_INLINE void Sm80_load_a_k4(unsigned rmem[2], const float* gmem, cuda::std::array<int_fast32_t, 2> element_strides)
{
  const unsigned row_stride = element_strides[0];
  const unsigned col_stride = element_strides[1];
  const unsigned warp_lane = threadIdx.x % 32u;
  const float* gmem_thread_baseaddr = &gmem[warp_lane / 4u * row_stride + warp_lane % 4u * col_stride];
  rmem[0] = __float_as_uint(gmem_thread_baseaddr[0]);
  rmem[1] = __float_as_uint(gmem_thread_baseaddr[8 * row_stride]);
}

EXO_CUDA_INLINE void Sm80_load_b_k8(unsigned rmem[2], const float* gmem, cuda::std::array<int_fast32_t, 2> element_strides)
{
  const unsigned row_stride = element_strides[0];
  const unsigned col_stride = element_strides[1];
  const unsigned warp_lane = threadIdx.x % 32u;
  const float* gmem_thread_baseaddr = &gmem[warp_lane % 4u * row_stride + warp_lane / 4u * col_stride];
  rmem[0] = __float_as_uint(gmem_thread_baseaddr[0]);
  rmem[1] = __float_as_uint(gmem_thread_baseaddr[4 * row_stride]);
}

EXO_CUDA_INLINE void Sm80_load_b_k4(unsigned rmem[1], const float* gmem, cuda::std::array<int_fast32_t, 2> element_strides)
{
  const unsigned row_stride = element_strides[0];
  const unsigned col_stride = element_strides[1];
  const unsigned warp_lane = threadIdx.x % 32u;
  const float* gmem_thread_baseaddr = &gmem[warp_lane % 4u * row_stride + warp_lane / 4u * col_stride];
  rmem[0] = __float_as_uint(gmem_thread_baseaddr[0]);
}

EXO_CUDA_INLINE void Sm80_store_d(float* gmem, const unsigned rmem[4], cuda::std::array<int_fast32_t, 2> element_strides)
{
  const unsigned row_stride = element_strides[0];
  const unsigned col_stride = element_strides[1];
  const unsigned warp_lane = threadIdx.x % 32u;
  float* gmem_thread_baseaddr = &gmem[(warp_lane / 4u) * row_stride + (warp_lane % 4u) * 2u * col_stride];
  gmem_thread_baseaddr[0] = __uint_as_float(rmem[0]);
  gmem_thread_baseaddr[col_stride] = __uint_as_float(rmem[1]);
  gmem_thread_baseaddr[8 * row_stride] = __uint_as_float(rmem[2]);
  gmem_thread_baseaddr[8 * row_stride + col_stride] = __uint_as_float(rmem[3]);
}

EXO_CUDA_INLINE void Sm80_zero_d(unsigned rmem[4])
{
  rmem[0] = 0;
  rmem[1] = 0;
  rmem[2] = 0;
  rmem[3] = 0;
}

EXO_CUDA_INLINE void Sm80_mma_k8(unsigned d[4], const unsigned a[4], const unsigned b[2])
{
  asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32\n\t"
      "{%0,%1,%2,%3},\n\t"
      "{%4,%5,%6,%7},\n\t"
      "{%8,%9},\n\t"
      "{%10,%11,%12,%13};" : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(d[0]), "r"(d[1]), "r"(d[2]), "r"(d[3]));
}

EXO_CUDA_INLINE void Sm80_mma_k4(unsigned d[4], const unsigned a[2], const unsigned b[1])
{
  asm("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32\n\t"
      "{%0,%1,%2,%3},\n\t"
      "{%4,%5},\n\t"
      "{%6},\n\t"
      "{%7,%8,%9,%10};" : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(d[0]), "r"(d[1]), "r"(d[2]), "r"(d[3]));
}
"""

class mma_instr_base:
    def instance_common(self):
        for v in self.access_info.values():
            v.actor_signature = actor_kinds.sig_cuda_classic
        self.actor_kind = actor_kinds.cuda_classic
        self.coll_unit = cuda_warp
        self.cu_includes = ["cuda/std/array"]
        self.cu_util = Sm80_cu_util


@instr
class tmp_load_a(mma_instr_base):
    def behavior(K: size, rmem: [f32][16,K] @ Sm80_RmemMatrixA, smem: [f32][16,K] @ CudaSmemLinear):
        for m in seq(0, 16):
            for k in seq(0, K):
                rmem[m,k] = smem[m,k]

    def instance(self, K):
        self.instance_common()
        if K != 4 and K != 8:
            raise ValueError("Require K=4 or K=8")
        self.instr_format = "exo_CudaUtil::Sm80_load_a_k" + str(K) + "({rmem_data}, &{smem_data}, {smem_layout});"


@instr
class tmp_load_b(mma_instr_base):
    def behavior(K: size, rmem: [f32][K,8] @ Sm80_RmemMatrixB, smem: [f32][K,8] @ CudaSmemLinear):
        for k in seq(0, K):
            for n in seq(0, 8):
                rmem[k,n] = smem[k,n]

    def instance(self, K):
        self.instance_common()
        if K != 4 and K != 8:
            raise ValueError("Require K=4 or K=8")
        self.instr_format = "exo_CudaUtil::Sm80_load_b_k" + str(K) + "({rmem_data}, &{smem_data}, {smem_layout});"



@instr
class tmp_mma(mma_instr_base):
    def behavior(K: size, D: [f32][16,8] @ Sm80_RmemMatrixD, A: [f32][16,K] @ Sm80_RmemMatrixA, B: [f32][K,8] @ Sm80_RmemMatrixB):
        for m in seq(0, 16):
            for n in seq(0, 8):
                for k in seq(0, K):
                    D[m,n] += A[m,k] * B[k,n]

    def instance(self, K):
        self.instance_common()
        if K != 4 and K != 8:
            raise ValueError("Require K=4 or K=8")
        self.instr_format = "exo_CudaUtil::Sm80_mma_k" + str(K) + "({D_data}, {A_data}, {B_data});"


@instr
class tmp_store_d(mma_instr_base):
    def behavior(gmem: [f32][16,8] @ CudaDeviceVisibleLinear, rmem: [f32][16,8] @ Sm80_RmemMatrixD):
        for m in seq(0, 16):
            for n in seq(0, 8):
                gmem[m,n] = rmem[m,n]

    def instance(self):
        self.instance_common()
        self.instr_format = "exo_CudaUtil::Sm80_store_d(&{gmem_data}, {rmem_data}, {gmem_layout});"


@instr
class tmp_zero_d(mma_instr_base):
    def behavior(rmem: [f32][16,8] @ Sm80_RmemMatrixD):
        for m in seq(0, 16):
            for n in seq(0, 8):
                rmem[m,n] = 0

    def instance(self):
        self.instance_common()
        self.instr_format = "exo_CudaUtil::Sm80_zero_d({rmem_data});"


@proc
def xgemm_Sm80_fence(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[K,N] @ CudaGmemLinear, C: f32[M,N] @ CudaGmemLinear):
    assert M % M1 == 0
    assert N % N1 == 0
    assert K % K0 == 0

    with CudaDeviceFunction(blockDim = 256, blocks_per_sm = 1):
        for m2 in cuda_tasks(0, M / M1):
            for n2 in cuda_tasks(0, N / N1):
                # Per CTA code

                # Tiles (double buffered)
                A_smem : f32[2, M1, K0] @ CudaSmemLinear
                B_smem : f32[2, K0, N1] @ CudaSmemLinear

                # Zero-out accumulator (warp code)
                D_rmem : f32[M1/Mw, N1/Nw, Mw/16, Nw/8, 16, 8] @ Sm80_RmemMatrixD
                for mw in cuda_threads(0, M1/Mw, unit=(N1/Nw) * cuda_warp):
                    for nw in cuda_threads(0, N1/Nw, unit=cuda_warp):
                        for m_seq in seq(0, Mw/16):
                            for n_seq in seq(0, Nw/8):
                                tmp_zero_d(D_rmem[mw,nw,m_seq, n_seq,:,:])

                # K tiles loop, double buffered
                # Don't accum tile in first iteration.
                # Don't load tile in last iteration.
                # 1 iteration delay between load and use.
                for k1 in seq(0, K / K0 + 1):
                    if k1 < K / K0:
                        with CudaAsync(Sm80_cp_async):
                            # Load A tile
                            for m1 in seq(0, M1 / 64):
                                for m0 in cuda_threads(0, 64, unit=4 * cuda_thread):
                                    for k0 in cuda_threads(0, 4, unit=cuda_thread):
                                        tmp_cpAsync16B_f32(A_smem[k1 % 2, m1 * 64 + m0, 4 * k0 : 4 * k0 + 4],
                                                           A[m2 * M1 + m1 * 64 + m0,
                                                             k1 * K0 + k0 * 4 : k1 * K0 + k0 * 4 + 4])

                            # Load B tile
                            for k0_seq in seq(0, 4):
                                for k0_par in cuda_threads(0, 4, unit=64 * cuda_thread):
                                    for n0 in cuda_threads(0, 64, unit=cuda_thread):
                                        tmp_cpAsync16B_f32(B_smem[k1 % 2, k0_seq * 4 + k0_par, 4 * n0 : 4 * n0 + 4],
                                                           B[k1 * K0 + k0_seq * 4 + k0_par,
                                                             n2 * N1 + 4 * n0 : n2 * N1 + 4 * n0 + 4])
                        # end CudaAsync(Sm80_cp_async)
                # for-k1 (K tiles) loop continues
                    if k1 > 0:
                        for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                            for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                                # Load all B matrix tiles ahead of time
                                B_rmem : f32[K0/MMA_K, Nw/8, MMA_K, 8] @ Sm80_RmemMatrixB
                                for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        tmp_load_b(B_rmem[k_seq,n_seq,:,:],
                                                   B_smem[1 - k1 % 2,
                                                          k_seq*MMA_K:(k_seq+1)*MMA_K,
                                                          nw*Nw + n_seq*8 : nw*Nw + (n_seq+1)*8], K=MMA_K)

                                for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                                    # Load A matrix tiles needed for m iteration
                                    A_rmem : f32[K0/MMA_K, 16, MMA_K] @ Sm80_RmemMatrixA
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        tmp_load_a(A_rmem[k_seq,:,:],
                                                       A_smem[1 - k1 % 2,
                                                              mw*Mw + m_seq*16 : mw*Mw + (m_seq+1)*16,
                                                              k_seq*MMA_K:(k_seq+1)*MMA_K], K=MMA_K)
                                    # Accumulate to tile of warp tiles owned by warp.
                                    for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                        for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                            tmp_mma(D_rmem[mw,nw,m_seq,n_seq,:,:],
                                                    A_rmem[k_seq,:,:],
                                                    B_rmem[k_seq,n_seq,:,:], K=MMA_K)

                    # Sm80_generic actor kind = (cuda_classic | Sm80_cp_async)
                    # Fence(Sm80_generic, Sm80_generic)
                    for tid in cuda_threads(0, 256):
                        cg : cuda_commit_group
                        Arrive(Sm80_cp_async, cg)
                        Await(cg, cuda_classic)
                        # Fence(Sm80_generic, Sm80_generic)
                    Fence(cuda_classic, Sm80_generic)

                # for-k1 (K tiles) loop ends

                # Write out accumulator
                for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                    for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                        for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                            for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                tmp_store_d(C[m2 * M1 + mw * Mw + m_seq * 16 : m2 * M1 + mw * Mw + (m_seq+1) * 16,
                                              n2 * N1 + nw * Nw + n_seq * 8 : n2 * N1 + nw * Nw + (n_seq+1) * 8],
                                            D_rmem[mw,nw,m_seq,n_seq,:,:])

xgemm_Sm80_fence = simplify(xgemm_Sm80_fence)
print(xgemm_Sm80_fence)

RING = 3
LAG = 1

@proc
def xgemm_Sm80_mbarrier(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[K,N] @ CudaGmemLinear, C: f32[M,N] @ CudaGmemLinear):
    assert M % M1 == 0
    assert N % N1 == 0
    assert K % K0 == 0

    with CudaDeviceFunction(blockDim = 256, blocks_per_sm = 1):
        for m2 in cuda_tasks(0, M / M1):
            for n2 in cuda_tasks(0, N / N1):
                # Per CTA code
                ringbar: cuda_mbarrier

                # Tiles (double buffered)
                A_smem : f32[RING, M1, K0] @ CudaSmemLinear
                B_smem : f32[RING, K0, N1] @ CudaSmemLinear

                # Zero-out accumulator (warp code)
                D_rmem : f32[M1/Mw, N1/Nw, Mw/16, Nw/8, 16, 8] @ Sm80_RmemMatrixD
                for mw in cuda_threads(0, M1/Mw, unit=(N1/Nw) * cuda_warp):
                    for nw in cuda_threads(0, N1/Nw, unit=cuda_warp):
                        for m_seq in seq(0, Mw/16):
                            for n_seq in seq(0, Nw/8):
                                tmp_zero_d(D_rmem[mw,nw,m_seq, n_seq,:,:])

                # K tiles loop, double buffered
                # Don't accum tile in first LAG-many iterations.
                # Don't load tile in last LAG-many iterations.
                # LAG iteration delay between load and use.
                for k1 in seq(0, K / K0 + LAG):
                    if k1 < K / K0:
                        with CudaAsync(Sm80_cp_async):
                            # Wait for ring buffer to be consumed; don't wait for first RING-many iterations
                            ReverseAwait(ringbar, Sm80_cp_async, RING)

                            # Load A tile
                            for m1 in seq(0, M1 / 64):
                                for m0 in cuda_threads(0, 64, unit=4 * cuda_thread):
                                    for k0 in cuda_threads(0, 4, unit=cuda_thread):
                                        tmp_cpAsync16B_f32(A_smem[k1 % RING, m1 * 64 + m0, 4 * k0 : 4 * k0 + 4],
                                                           A[m2 * M1 + m1 * 64 + m0,
                                                             k1 * K0 + k0 * 4 : k1 * K0 + k0 * 4 + 4])

                            # Load B tile
                            for k0_seq in seq(0, 4):
                                for k0_par in cuda_threads(0, 4, unit=64 * cuda_thread):
                                    for n0 in cuda_threads(0, 64, unit=cuda_thread):
                                        tmp_cpAsync16B_f32(B_smem[k1 % RING, k0_seq * 4 + k0_par, 4 * n0 : 4 * n0 + 4],
                                                           B[k1 * K0 + k0_seq * 4 + k0_par,
                                                             n2 * N1 + 4 * n0 : n2 * N1 + 4 * n0 + 4])
                            Arrive(Sm80_cp_async, ringbar)
                        # end CudaAsync(Sm80_cp_async)
                # for-k1 (K tiles) loop continues
                    if k1 >= LAG:
                        # Wait for ring buffer to be filled
                        Await(ringbar, cuda_classic)

                        for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                            for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                                # Load all B matrix tiles ahead of time
                                B_rmem : f32[K0/MMA_K, Nw/8, MMA_K, 8] @ Sm80_RmemMatrixB
                                for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        tmp_load_b(B_rmem[k_seq,n_seq,:,:],
                                                   B_smem[(k1 - LAG) % RING,
                                                          k_seq*MMA_K:(k_seq+1)*MMA_K,
                                                          nw*Nw + n_seq*8 : nw*Nw + (n_seq+1)*8], K=MMA_K)

                                for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                                    # Load A matrix tiles needed for m iteration
                                    A_rmem : f32[K0/MMA_K, 16, MMA_K] @ Sm80_RmemMatrixA
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        tmp_load_a(A_rmem[k_seq,:,:],
                                                   A_smem[(k1 - LAG) % RING,
                                                          mw*Mw + m_seq*16 : mw*Mw + (m_seq+1)*16,
                                                          k_seq*MMA_K:(k_seq+1)*MMA_K], K=MMA_K)
                                    # Accumulate to tile of warp tiles owned by warp.
                                    for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                        for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                            tmp_mma(D_rmem[mw,nw,m_seq,n_seq,:,:],
                                                    A_rmem[k_seq,:,:],
                                                    B_rmem[k_seq,n_seq,:,:], K=MMA_K)
                        # Signal that it's safe to overwrite ring buffer entry
                        ReverseArrive(cuda_classic, ringbar)
                # for-k1 (K tiles) loop ends

                # Write out accumulator
                for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                    for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                        for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                            for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                tmp_store_d(C[m2 * M1 + mw * Mw + m_seq * 16 : m2 * M1 + mw * Mw + (m_seq+1) * 16,
                                              n2 * N1 + nw * Nw + n_seq * 8 : n2 * N1 + nw * Nw + (n_seq+1) * 8],
                                            D_rmem[mw,nw,m_seq,n_seq,:,:])

xgemm_Sm80_mbarrier = simplify(xgemm_Sm80_mbarrier)
