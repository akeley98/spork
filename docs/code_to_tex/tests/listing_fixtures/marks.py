# Mark calibration fixture: dots are drawn at every mark by the tests /
# calibration page, so the rendered PNG shows where marks land.
# TeX: listing marks 1
# TeX: begin marks
# TeX: mark marks a0 start
# TeX: mark marks a1 "accum" before
# TeX: mark marks a2 "accum" after
# TeX: mark marks a3 end
for k in seq(0, K):  accum = 0
    # TeX: color marks blue "accum"
    # TeX: mark marks b1 "accum" before
    # TeX: mark marks b2 "accum" after
    # TeX: mark marks b3 "k" 2:1 after
    # TeX: mark marks b4 end
    accum += A[m, k] * B[k, n]  # x comment
    # TeX: mark marks c1 "xyz" before
    # TeX: mark marks c2 "Hpq" after
    # TeX: mark marks c3 "Hpq" before yshift=0pt
    C[m, n] = xyz + Hpq
# TeX: end marks
