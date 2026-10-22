"""code_to_tex: shared paper/slides LaTeX toolchain (listing generator, styles, build driver).

Self-contained (stdlib only, Python 3.10); see README.md.

* ``python3 -m code_to_tex SRC OUTDIR`` -- listing generator (`code_to_tex.listing`)
* ``python3 -m code_to_tex.build DIR ...`` -- build driver (`code_to_tex.build`)
* ``latex/*.tex`` -- styles copied into each document's ``copied_styles/``
"""
