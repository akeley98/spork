import math
import torch

def make_rand(N, d, seed):
    generator = torch.Generator()
    tensor = torch.rand((N, d), dtype=torch.float32, generator = generator.manual_seed(seed))
    return torch.pow(tensor, 3.6)

# Adapting pseudocode from here with dropout removed
# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
def simple_attn(Q, K, V, scale = None):
    scale_factor = 1 / math.sqrt(Q.size(-1)) if scale is None else scale
    S = torch.einsum("ik,jk->ij", Q, K) * scale_factor
    P = torch.softmax(S, dim=-1) # row-wise
    O = P @ V
    return O

# Returns (value, row, col) where row, col is the location of maximum difference and value is said difference.
def diff(A, B):
    shape = A.shape
    assert(A.shape == B.shape)
    diff = torch.abs(A - B)
    r, c = max(((r, c) for r in range(shape[0]) for c in range(shape[1])), key = lambda tup: diff[tup[0], tup[1]])
    return diff[r, c], r, c

N, d = 2560, 64
Q = make_rand(N, d, 101)
K = make_rand(N, d, 237)
V = make_rand(N, d, 280)

simple_o = simple_attn(Q, K, V)
simple1_o = simple_attn(Q, K, V, scale = 1)
torch_o = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
torch1_o = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale = 1)

# CUDA stuff

import ctypes

lib = ctypes.cdll.LoadLibrary("./libattn.so")
cuda_func = lib.htod_f32_tk_4090_fwd
f32_ptr = ctypes.POINTER(ctypes.c_float)
cuda_func.argtypes = (ctypes.c_int, ctypes.c_int, f32_ptr, f32_ptr, f32_ptr, f32_ptr)
as_f32_ptr = lambda tensor: ctypes.cast(tensor.data_ptr(), f32_ptr)

def cuda_attn(Q, K, V):
    N, d = Q.shape
    O = torch.empty((N, d), dtype = torch.float32)
    assert(K.shape == (N, d))
    assert(V.shape == (N, d))
    assert(O.shape == (N, d))
    assert(Q.dtype == torch.float32)
    assert(K.dtype == torch.float32)
    assert(V.dtype == torch.float32)
    assert(O.dtype == torch.float32)
    cuda_func(N, d, as_f32_ptr(Q), as_f32_ptr(K), as_f32_ptr(V), as_f32_ptr(O))
    return O

cuda_o = cuda_attn(Q, K, V)
