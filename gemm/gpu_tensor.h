#pragma once

#include <new>
#include <stdint.h>
#include <stdio.h>

#include <cuda_runtime.h>

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

class StreamWorkspace
{
  public:
    cudaStream_t stream = {};
  private:
    void* d_workspace = nullptr;
    size_t current_workspace_bytes = 0;

  public:
    explicit StreamWorkspace(cudaStream_t _stream) : stream(_stream)
    {
    }

    StreamWorkspace(StreamWorkspace&&) = delete;

    ~StreamWorkspace()
    {
        cudaFreeAsync(d_workspace, stream);
    }

    operator cudaStream_t() const
    {
        return stream;
    }

    void* alloc_at_least(size_t bytes)
    {
        if (d_workspace) {
            cudaFreeAsync(d_workspace, stream);
            d_workspace = nullptr;
        }
        if (!d_workspace || bytes > current_workspace_bytes) {
            if (const cudaError_t err = cudaGetLastError()) {
                fprintf(stderr, "Error prior to allocating workspace: %i (%s)\n", (int)err, cudaGetErrorString(err));
                throw std::bad_alloc{};
            }
            cudaMallocAsync(&d_workspace, bytes, stream);
            if (const cudaError_t err = cudaGetLastError()) {
                fprintf(stderr, "Error allocating workspace: %i (%s)\n", (int)err, cudaGetErrorString(err));
                throw std::bad_alloc{};
            }
            current_workspace_bytes = bytes;
        }
        return d_workspace;
    }
};
