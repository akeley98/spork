# Snippets of code for spork_guide.pdf that are not expected to compile.

# TeX: version loop_mode_syntax 1
# TeX: begin loop_mode_syntax[0]
# TeX: color line *
#           gggggggggggggg         ggggggggggggggggggggggggggggg
for iter in loop-mode-name(lo, hi, optional-keyword=optional-arg, ...):
# TeX: end loop_mode_syntax[0]

# TeX: version cuda_threads_syntax 3
# TeX: begin cuda_threads_syntax[0]
for iter in cuda_threads(0, hi, unit=<collective-unit>):
    ...body
# TeX: end cuda_threads_syntax[0]
