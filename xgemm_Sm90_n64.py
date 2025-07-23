import sys, os
sys.path.append(os.path.split(__name__)[0])

from xgemm_Sm90 import make_Sm90_gemm

xgemm_Sm90_n64 = make_Sm90_gemm(64, 2, 4)
