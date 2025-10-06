from xgemm_Sm90 import make_Sm90_gemm

xgemm_Sm90_n256_tma_to_gmem = make_Sm90_gemm(256, 2, 1, tma_to_gmem=True)
