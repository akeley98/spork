#include "matrix_util.hpp"

static constexpr uint32_t block_x = 16, block_y = 16;

// https://jcgt.org/published/0009/03/02/paper.pdf
constexpr uint32_t matrix_util_pcg3d(uint32_t x, uint32_t y, uint32_t z)
{
    x = x * 1664525u + 1013904223u;
    y = y * 1664525u + 1013904223u;
    z = z * 1664525u + 1013904223u;
    x += y*z;
    y += z*x;
    z += x*y;
    x ^= x >> 16u;
    y ^= y >> 16u;
    z ^= z >> 16u;
    x += y*z;
    y += z*x;
    z += x*y;
    return x ^ y ^ z;
}

__global__ void matrix_util_random_bf16_kernel(size_t w, size_t h, __nv_bfloat16* array, int seed)
{
    uint32_t x = blockIdx.x * block_x + threadIdx.x;
    uint32_t y = blockIdx.y * block_y + threadIdx.y;
    if (x < w && y < h) {
        uint32_t hashed_bits = matrix_util_pcg3d(x, y, uint32_t(seed));
        hashed_bits = (hashed_bits >> 16u) ^ (hashed_bits & 0xFFFFu);
        array[x + y*w] = __nv_bfloat16(hashed_bits) * __nv_bfloat16(1.0f / 32768);
    }
}

void launch_random_bf16_kernel(cudaStream_t stream, size_t w, size_t h, __nv_bfloat16* array, int seed)
{
    dim3 grid = {uint32_t(w + block_x - 1) / block_x, uint32_t(h + block_y - 1) / block_y, 1};
    dim3 block = {block_x, block_y, 1};
    matrix_util_random_bf16_kernel<<<grid, block, 0, stream>>>(w, h, array, seed);
}
