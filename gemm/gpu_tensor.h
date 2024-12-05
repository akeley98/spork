#pragma once

#include <stdint.h>

struct GPU_Tensors
{
    uint32_t M;
    uint32_t N;
    uint32_t K;
    float const *a; /* pointer to GPU memory */
    float const *b; /* pointer to GPU memory */
    float *c;       /* pointer to GPU memory */

    bool a_col_major;
    bool b_col_major;
    bool c_col_major;
};
