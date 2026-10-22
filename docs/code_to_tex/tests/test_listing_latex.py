"""LaTeX integration tests for \\sporkListing (need xelatex; poppler for some checks).

Each test builds a tiny document directory: copied_styles/ (a stub
spork_style.tex from listing_fixtures/ plus the real code_to_tex_support.tex),
code/ (generated listings) and a main file, then compiles it with xelatex.
"""

import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_PARENT))

from code_to_tex.listing import generate  # noqa: E402

PACKAGE = PACKAGE_PARENT / "code_to_tex"
FIXTURES = Path(__file__).resolve().parent / "listing_fixtures"

pytestmark = pytest.mark.skipif(shutil.which("xelatex") is None, reason="xelatex not installed")
needs_poppler = pytest.mark.skipif(
    shutil.which("pdftotext") is None or shutil.which("pdfinfo") is None, reason="poppler-utils not installed"
)

DEMO = """\
# TeX: listing demo 3
# TeX: begin demo
x = 1  # first
# TeX: end demo
# TeX: begin demo[1:]
# TeX: mark demo[1:] m "y"
y = 2
# TeX: end demo[1:]
# TeX: begin demo[2]
z = 3
# TeX: end demo[2]
# TeX: begin demo
# TeX: color demo[2] red "end"
end = 4
# TeX: end demo
"""

DECK = r"""\documentclass[aspectratio=169]{beamer}
\newif\ifsporkSlides \sporkSlidestrue
\input{copied_styles/spork_style.tex}
\input{copied_styles/code_to_tex_support.tex}
\begin{document}
%s
\end{document}
"""

PAPER = r"""\documentclass{article}
\newif\ifsporkSlides \sporkSlidesfalse
\input{copied_styles/spork_style.tex}
\input{copied_styles/code_to_tex_support.tex}
\begin{document}
%s
\end{document}
"""


def make_doc(tmp_path, body, template, sources=((DEMO, "samples.py"),)):
    (tmp_path / "copied_styles").mkdir()
    shutil.copy(FIXTURES / "stub_spork_style.tex", tmp_path / "copied_styles" / "spork_style.tex")
    shutil.copy(PACKAGE / "latex" / "code_to_tex_support.tex", tmp_path / "copied_styles")
    for text, name in sources:
        (tmp_path / name).write_text(text)
        generate(str(tmp_path / name), str(tmp_path / "code"))
    (tmp_path / "main.tex").write_text(template % textwrap.dedent(body))
    return tmp_path


def xelatex(doc_dir, passes=1, expect_ok=True):
    for _ in range(passes):
        r = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=doc_dir, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            break
    log = (doc_dir / "main.log").read_text(errors="replace")
    if expect_ok:
        assert r.returncode == 0, log[-3000:]
    return r.returncode, log


def page_count(doc_dir):
    info = subprocess.run(["pdfinfo", "main.pdf"], cwd=doc_dir, capture_output=True, text=True).stdout
    return int(re.search(r"Pages:\s+(\d+)", info)[1])


def page_text(doc_dir, page):
    return subprocess.run(
        ["pdftotext", "-f", str(page), "-l", str(page), "-layout", "main.pdf", "-"],
        cwd=doc_dir, capture_output=True, text=True,
    ).stdout


def word_y(doc_dir, page, word):
    """yMin of the first occurrence of word on a page (pdftotext -bbox)."""
    html = subprocess.run(
        ["pdftotext", "-f", str(page), "-l", str(page), "-bbox", "main.pdf", "-"],
        cwd=doc_dir, capture_output=True, text=True,
    ).stdout
    m = re.search(r'yMin="([\d.]+)"[^>]*>' + re.escape(word) + "<", html)
    return float(m[1]) if m else None


def lines_of(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


@needs_poppler
def test_slides_one_step_per_version(tmp_path):
    doc = make_doc(tmp_path, r"\begin{frame}[t]{T}\sporkListing{demo}\end{frame}", DECK)
    xelatex(doc)
    assert page_count(doc) == 3
    assert lines_of(page_text(doc, 1)) == ["T", "x = 1 # first", "end = 4"]
    assert lines_of(page_text(doc, 2)) == ["T", "x = 1 # first", "y = 2", "end = 4"]
    assert lines_of(page_text(doc, 3)) == ["T", "x = 1 # first", "y = 2", "z = 3", "end = 4"]


@needs_poppler
def test_slides_offset_and_last_version_persists(tmp_path):
    body = r"""
    \begin{frame}[t]{T}
    \sporkListing[offset=1]{demo}
    \only<6>{extra}
    \end{frame}
    """
    doc = make_doc(tmp_path, body, DECK)
    xelatex(doc)
    assert page_count(doc) == 6
    assert lines_of(page_text(doc, 1)) == ["T"]
    assert "y = 2" not in page_text(doc, 2)
    assert "z = 3" in page_text(doc, 4)
    assert "z = 3" in page_text(doc, 5) and "z = 3" in page_text(doc, 6)  # last version stays
    assert "extra" in page_text(doc, 6)


@needs_poppler
def test_slides_fixed_version(tmp_path):
    doc = make_doc(tmp_path, r"\begin{frame}[t]{T}\sporkListing[version=1]{demo}\end{frame}", DECK)
    xelatex(doc)
    assert page_count(doc) == 1
    assert lines_of(page_text(doc, 1)) == ["T", "x = 1 # first", "y = 2", "end = 4"]


@needs_poppler
def test_slides_reserve_keeps_space(tmp_path):
    body = r"""
    \begin{frame}[t]{Grow}\sporkListing{demo}\end{frame}
    \begin{frame}[t]{Reserve}\sporkListing[reserve]{demo}\end{frame}
    """
    doc = make_doc(tmp_path, body, DECK)
    xelatex(doc)
    assert page_count(doc) == 6
    grow = [word_y(doc, p, "end") for p in (1, 2, 3)]
    reserve = [word_y(doc, p, "end") for p in (4, 5, 6)]
    assert grow[0] < grow[1] < grow[2]  # code grows: last line moves down
    assert reserve[0] == reserve[1] == reserve[2]  # reserve: last line stays put
    assert reserve[0] == pytest.approx(grow[2], abs=0.5)
    assert "y = 2" not in page_text(doc, 4)


@needs_poppler
def test_paper_default_is_last_version_and_version_option(tmp_path):
    body = r"""
    LAST \sporkListing{demo}
    FIRST \sporkListing[version=0]{demo}
    MIDDLE \sporkListing[version=1]{demo}
    """
    doc = make_doc(tmp_path, body, PAPER)
    xelatex(doc)
    assert page_count(doc) == 1
    text = lines_of(page_text(doc, 1))
    assert text[:13] == [
        "LAST", "x = 1 # first", "y = 2", "z = 3", "end = 4",
        "FIRST", "x = 1 # first", "end = 4",
        "MIDDLE", "x = 1 # first", "y = 2", "end = 4",
        "1",
    ]


@needs_poppler
def test_paper_line_spacing_is_uniform(tmp_path):
    """Colored (smashed) boxes and comments do not change line spacing."""
    doc = make_doc(tmp_path, r"\sporkListing{demo}", PAPER)
    xelatex(doc)
    ys = [word_y(doc, 1, w) for w in ("x", "y", "z", "end")]
    gaps = [b - a for a, b in zip(ys, ys[1:])]
    assert max(gaps) - min(gaps) < 0.05, ys


def test_version_out_of_range_is_a_latex_error(tmp_path):
    doc = make_doc(tmp_path, r"\sporkListing[version=3]{demo}", PAPER)
    rc, log = xelatex(doc, expect_ok=False)
    assert rc != 0
    assert "version=3 out of range (the listing has versions 0..2)" in re.sub(r"\s*\n\(spork\)\s*", " ", log)


def test_direct_input_is_a_latex_error(tmp_path):
    doc = make_doc(tmp_path, r"\input{code/demo.tex}", PAPER)
    rc, log = xelatex(doc, expect_ok=False)
    assert rc != 0
    assert "must be included with" in log


def test_marks_written_to_aux_paper_and_slides(tmp_path):
    body = r"""
    \sporkListing{demo}
    \sporkListing[prefix=again]{demo}
    \begin{tikzpicture}[remember picture, overlay]
    \fill (pic cs:demo-m) circle (1pt);
    \end{tikzpicture}
    """
    doc = make_doc(tmp_path, body, PAPER)
    xelatex(doc, passes=2)
    aux = (doc / "main.aux").read_text()
    assert r"\savepointas{demo-m}" in aux
    assert r"\savepointas{again-m}" in aux


def test_marks_in_slides_are_per_step(tmp_path):
    doc = make_doc(tmp_path, r"\begin{frame}[t]{T}\sporkListing{demo}\end{frame}", DECK)
    xelatex(doc, passes=2)
    aux = (doc / "main.aux").read_text()
    # tikzmark suffixes the slide number in Beamer; mark m is in versions 1 and 2 = steps 2 and 3.
    assert sorted(re.findall(r"\\savepointas\{(demo-m-\d+)\}", aux)) == ["demo-m-2", "demo-m-3"]


@needs_poppler
def test_nyc25_cpu_deck(tmp_path):
    """The realistic sample: 5 versions -> 5 overlay steps, legacy-equivalent text."""
    doc = make_doc(
        tmp_path,
        r"\begin{frame}[t]{Starting out with Exo}\small\sporkListing{cpu}\end{frame}",
        DECK,
        sources=(((FIXTURES / "nyc25_cpu.py").read_text(), "samples.py"),),
    )
    xelatex(doc)
    assert page_count(doc) == 5
    def squash(text):  # pdftotext column layout does not preserve exact spacing
        return [re.sub(r" +", " ", line) for line in lines_of(text)]

    for step in range(1, 6):
        expected = (doc / "code" / f"cpu.{step - 1}.txt").read_text()
        assert squash(page_text(doc, step))[1:] == squash(expected), step
