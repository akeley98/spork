from xgemm_Sm90 import make_Sm90_gemm

xgemm_Sm90_n256_tma_K1 = make_Sm90_gemm(256, 2, 1, tma_to_gmem=True)
xgemm_Sm90_n256_tma_K4 = make_Sm90_gemm(256, 2, 1, tma_to_gmem=True, K_tasks=4)
