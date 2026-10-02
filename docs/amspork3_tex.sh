exocc amspork3_samples.py && python3 code_to_tex.py amspork3_samples.py amspork3/code/ && xelatex amspork3.tex </dev/null || exit 1

# Only update the copy that evince watches if the build is clean;
# evince crashes reloading PDFs with undefined references.
if grep -qi "there were undefined references" amspork3.log; then
    echo "amspork3_tex.sh: undefined references; not updating amspork3_copy.pdf" >&2
    exit 1
fi
# Copy then rename so the viewer never sees a partially written file.
cp amspork3.pdf .amspork3_copy.pdf.tmp && mv .amspork3_copy.pdf.tmp amspork3_copy.pdf
