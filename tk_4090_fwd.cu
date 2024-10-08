#include "tk_4090_fwd.hpp"
#include "attn_main.hpp"
#include "matrix_util.hpp"

#include "kittens.cuh"

 // this kernel is more of an example kernel to show some TK programming models, rather than a kernel we think you should put into production, though it is pretty fast!

// akeley98: 1 worker = 1 warp?
#define NUM_WORKERS 16 // This kernel uses 16 workers in parallel per block, to help issue instructions more quickly.
#define SMEM_BYTES (sizeof(st_bf_1x4) * NUM_WORKERS * 2)

// this kernel only handles headdim=64 for simplicity. Also n should be a multiple of 256 here.
// akeley98: Seems all matrices are row-major?

using namespace kittens;

__global__ void __launch_bounds__(NUM_WORKERS*32, 1)
attend_ker64(int n, const bf16* __restrict__ __q__, const bf16* __restrict__ __k__, const bf16* __restrict__ __v__, bf16* __o__)
{
    assert(blockDim.x == 32 * NUM_WORKERS);
    auto warpid        = kittens::warpid();
    auto block_start   = blockIdx.x*(n*64);
    const bf16 *_q = __q__ + block_start, *_k = __k__ + block_start, *_v = __v__ + block_start;
          bf16 *_o = __o__ + block_start;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);

    // K and V live in shared memory -- this is about all that will fit.
    st_bf_1x4 (&k_smem)[NUM_WORKERS] = al.allocate<st_bf_1x4, NUM_WORKERS>();
    st_bf_1x4 (&v_smem)[NUM_WORKERS] = al.allocate<st_bf_1x4, NUM_WORKERS>();

    // Initialize all of the register tiles.
    rt_bf_1x4<> q_reg, k_reg, v_reg; // v_reg need to be swapped into col_l
    rt_fl_1x1<> att_block;
    rt_bf_1x1<> att_block_mma;
    rt_fl_1x4<> o_reg;
    rt_fl_1x1<>::col_vec max_vec_last, max_vec; // these are column vectors for the attention block
    rt_fl_1x1<>::col_vec norm_vec_last, norm_vec; // these are column vectors for the attention block

    int qo_blocks = n / (q_reg.rows*NUM_WORKERS), kv_blocks = n / (q_reg.rows*NUM_WORKERS);

    for(auto q_blk = 0; q_blk < qo_blocks; q_blk++) {

        // each warp loads its own Q tile of 16x64, and then multiplies by 1/sqrt(d)
        load(q_reg, _q + (q_blk*NUM_WORKERS + warpid)*q_reg.num_elements, q_reg.cols);
        mul(q_reg, q_reg, __float2bfloat16(0.125f)); // temperature adjustment

        // zero flash attention L, M, and O registers.
        neg_infty(max_vec); // zero registers for the Q chunk
        zero(norm_vec);
        zero(o_reg);

        // iterate over k, v for these q's that have been loaded
        for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {

            // each warp loads its own chunk of k, v into shared memory
            load(v_smem[warpid], _v + (kv_idx*NUM_WORKERS + warpid)*q_reg.num_elements, q_reg.cols);
            load(k_smem[warpid], _k + (kv_idx*NUM_WORKERS + warpid)*q_reg.num_elements, q_reg.cols);
            __syncthreads(); // we need to make sure all memory is loaded before we can begin the compute phase

            // now each warp goes through all of the subtiles, loads them, and then does the flash attention internal alg.
            for(int subtile = 0; subtile < NUM_WORKERS; subtile++) {

                load(k_reg, k_smem[subtile]); // load k from shared into registers

                zero(att_block); // zero 16x16 attention tile
                mma_ABt(att_block, q_reg, k_reg, att_block); // Q@K.T

                copy(norm_vec_last, norm_vec);
                copy(max_vec_last,  max_vec);

                row_max(max_vec, att_block, max_vec); // accumulate onto the max_vec
                sub_row(att_block, att_block, max_vec); // subtract max from attention -- now all <=0
                exp(att_block, att_block); // exponentiate the block in-place.

                sub(max_vec_last, max_vec_last, max_vec); // subtract new max from old max to find the new normalization.
                exp(max_vec_last, max_vec_last); // exponentiate this vector -- this is what we need to normalize by.
                mul(norm_vec, norm_vec, max_vec_last); // and the norm vec is now normalized.

                row_sum(norm_vec, att_block, norm_vec); // accumulate the new attention block onto the now-rescaled norm_vec
                div_row(att_block, att_block, norm_vec); // now the attention block is correctly normalized

                mul(norm_vec_last, norm_vec_last, max_vec_last); // normalize the previous norm vec according to the new max
                div(norm_vec_last, norm_vec_last, norm_vec); // normalize the previous norm vec according to the new norm

                copy(att_block_mma, att_block); // convert to bf16 for mma_AB

                load(v_reg, v_smem[subtile]); // load v from shared into registers.
                rt_bf_1x4<ducks::rt_layout::col> &v_reg_col = swap_layout_inplace(v_reg); // this is a reference and the call has invalidated v_reg

                mul_row(o_reg, o_reg, norm_vec_last); // normalize o_reg in advance of mma_AB'ing onto it
                mma_AB(o_reg, att_block_mma, v_reg_col, o_reg); // mfma onto o_reg with the local attention@V matmul.
            }
            __syncthreads(); // we need to make sure all warps are done before we can start loading the next kv chunk
        }

        store(_o + (q_blk*NUM_WORKERS + warpid)*q_reg.num_elements, o_reg, q_reg.cols); // write out o. compiler has an issue with register usage if d is made constexpr q_reg.rows :/
    }
}

void launch_tk_4090_fwd_kernel(const HostAttnArgs& args, __nv_bfloat16* o)
{
    cudaFuncSetAttribute(attend_ker64, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    assert(args.N % 256 == 0);
    assert(args.d == 64);
    const uint32_t grid = args.d / 64u;
    attend_ker64<<<grid, NUM_WORKERS * 32, SMEM_BYTES, args.stream>>>(args.N, args.q.get(), args.k.get(), args.v.get(), o);
}

void htod_f32_tk_4090_fwd(int N, int d, const float* q, const float* k, const float* v, float* o)
{
    std::unique_ptr<__nv_bfloat16[]> host_bfloat16(new __nv_bfloat16[N * d]);
    auto convert_and_upload = [N, d, &host_bfloat16] (__nv_bfloat16* device_bfloat16, const float* host_f32)
    {
        for (int row = 0; row < N; ++row) {
            for (int col = 0; col < d; ++col) {
                host_bfloat16[row * d + col] = __bfloat162float(host_f32[row * d + col]);
            }
        }
        cudaMemcpy(device_bfloat16, host_bfloat16.get(), sizeof(__nv_bfloat16) * N * d, cudaMemcpyHostToDevice);
    };

    HostAttnArgs args{};
    args.N = N;
    args.d = d;
    args.q = make_unique_cuda_array<__nv_bfloat16>(N, d);
    convert_and_upload(args.q.get(), q);
    args.k = make_unique_cuda_array<__nv_bfloat16>(N, d);
    convert_and_upload(args.k.get(), k);
    args.v = make_unique_cuda_array<__nv_bfloat16>(N, d);
    convert_and_upload(args.v.get(), v);
    auto device_o = make_unique_cuda_array<__nv_bfloat16>(N, d);

    launch_tk_4090_fwd_kernel(args, device_o.get());
    cudaMemcpy(host_bfloat16.get(), device_o.get(), sizeof(__nv_bfloat16) * N * d, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < N * d; ++i) {
        o[i] = __bfloat162float(host_bfloat16[i]);
    }
}
