"""LaTeX-level tests for latex/spork_style.tex (the master paper/slides style).

Compiles the fixture documents in tests/style_fixtures/ with xelatex in a temp dir laid out like a
document directory: copied_styles/ holds a copy of spork_style.tex plus an empty stub
code_to_tex_support.tex (so these tests don't depend on the listing support file).

Checks: page counts (overlay steps), paper/slide visibility of every overlay intent
(text via pdftotext, TikZ fills via exact pixel colors from pdftoppm), the \\sporkOnlyFrame
single-frame build, fonts (incl. sporkfigure / \\sporkFigFont), \\sporkCodeRule width, and the
sporkfigure environment: scaling, per-slide additions to a shared body, the width warning, and
local node names (references to names it doesn't define are errors).
Skipped if xelatex is missing; individual checks skip if poppler tools are missing.
"""

import os
import re
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "style_fixtures")
STYLE = os.path.join(os.path.dirname(HERE), "latex", "spork_style.tex")

pytestmark = pytest.mark.skipif(shutil.which("xelatex") is None, reason="xelatex not installed")

# Fill colors used in fixtures/fig_intent_body.tex (values from spork_style.tex).
RED_BG = (255, 211, 225)        # REVEALKEY: spork reveal=<2->
INV_GREEN_BG = (0, 170, 149)    # SLIDEKEY: spork slide only=<3>
INV_BLUE_BG = (0, 0, 225)       # PAPERKEY: spork paper only
YELLOW_BG = (255, 232, 0)       # STYLEDKEY: spork slide only=<2>{spork yellow style}
VIOLET_BG = (218, 204, 255)     # PAPERSTYLED: spork paper style={spork violet style}
FIG_COLORS = [RED_BG, INV_GREEN_BG, INV_BLUE_BG, YELLOW_BG, VIOLET_BG]


def _need(tool):
    if shutil.which(tool) is None:
        pytest.skip(f"{tool} not installed")


def _xelatex(workdir, args, passes=1):
    for _ in range(passes):
        proc = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error"] + args,
            cwd=workdir, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=120)
    return proc


class Doc:
    def __init__(self, workdir, job):
        self.workdir = workdir
        self.job = job
        self.pdf = os.path.join(workdir, job + ".pdf")
        with open(os.path.join(workdir, job + ".log"), encoding="utf-8", errors="replace") as f:
            self.log = f.read()

    def pages(self):
        # Beamer writes one /Type /Page per page; avoid needing pdfinfo.
        if shutil.which("pdfinfo"):
            out = subprocess.run(["pdfinfo", self.pdf], stdout=subprocess.PIPE, text=True).stdout
            return int(re.search(r"^Pages:\s+(\d+)", out, re.M).group(1))
        m = re.search(r"Output written on .*?\((\d+) pages?", self.log, re.S)
        return int(m.group(1))

    def text(self, page=None):
        _need("pdftotext")
        args = ["pdftotext"]
        if page is not None:
            args += ["-f", str(page), "-l", str(page)]
        return subprocess.run(args + [self.pdf, "-"], stdout=subprocess.PIPE, text=True).stdout

    def colors(self, page):
        """Set of exact RGB colors covering >= 20 pixels on a page (rendered at 40 dpi)."""
        _need("pdftoppm")
        out = os.path.join(self.workdir, f"{self.job}_p{page}")
        subprocess.run(["pdftoppm", "-r", "40", "-f", str(page), "-l", str(page), "-singlefile",
                        self.pdf, out], check=True)
        with open(out + ".ppm", "rb") as f:
            data = f.read()
        fields = data.split(maxsplit=4)
        assert fields[0] == b"P6"
        width, height = int(fields[1]), int(fields[2])
        pix = fields[4][: width * height * 3]
        counts = {}
        for i in range(0, len(pix), 3):
            c = (pix[i], pix[i + 1], pix[i + 2])
            counts[c] = counts.get(c, 0) + 1
        return {c for c, n in counts.items() if n >= 20}

    def words(self, page=None):
        """[(word, xMin, xMax)] from pdftotext -bbox (points)."""
        _need("pdftotext")
        args = ["pdftotext", "-bbox"]
        if page is not None:
            args += ["-f", str(page), "-l", str(page)]
        out = subprocess.run(args + [self.pdf, "-"], stdout=subprocess.PIPE, text=True).stdout
        return [(m.group(3), float(m.group(1)), float(m.group(2)))
                for m in re.finditer(r'xMin="([\d.]+)"[^>]*xMax="([\d.]+)"[^>]*>([^<]*)<', out)]

    def fonts(self):
        _need("pdffonts")
        return subprocess.run(["pdffonts", self.pdf], stdout=subprocess.PIPE, text=True).stdout


STYLE_JOBS = ["paper", "paper_fig", "deck", "frames", "rule", "paper_figfont", "sporkfig", "sporkfig_deck"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    work = str(tmp_path_factory.mktemp("style"))
    for name in os.listdir(FIXTURES):
        shutil.copy(os.path.join(FIXTURES, name), work)
    styles = os.path.join(work, "copied_styles")
    os.mkdir(styles)
    shutil.copy(STYLE, styles)
    open(os.path.join(styles, "code_to_tex_support.tex"), "w").close()

    docs = {}
    for job in STYLE_JOBS:
        _xelatex(work, [job + ".tex"], passes=2)
        docs[job] = Doc(work, job)
    _xelatex(work, ["-jobname=frames_b", r"\def\sporkOnlyFrame{b}\input{frames.tex}"])
    docs["frames_b"] = Doc(work, "frames_b")
    _xelatex(work, ["-jobname=frames_ac", r"\def\sporkOnlyFrame{a,c}\input{frames.tex}"])
    docs["frames_ac"] = Doc(work, "frames_ac")
    return docs


@pytest.mark.parametrize("job", STYLE_JOBS + ["frames_b", "frames_ac"])
def test_compiles_cleanly(built, job):
    doc = built[job]
    errors = [l for l in doc.log.splitlines() if l.startswith("!")]
    assert not errors, errors
    assert os.path.exists(doc.pdf)
    # \sporkPaperLink must not leave undefined references on slides; paper labels resolve in 2 passes.
    assert "There were undefined references" not in doc.log
    assert not re.search(r"(Reference|Hyper reference) `[^']*' .*undefined", doc.log)


@pytest.mark.parametrize("job,pages", [
    ("paper", 1), ("paper_fig", 1), ("deck", 11), ("frames", 4),
    ("frames_b", 2), ("frames_ac", 2), ("rule", 1), ("paper_figfont", 1),
    ("sporkfig", 1), ("sporkfig_deck", 3),
])
def test_page_counts(built, job, pages):
    assert built[job].pages() == pages


def test_paper_text_intent(built):
    text = built["paper"].text()
    for present in ["REVEALTEXT", "PAPERTEXT", "LINKTEXT", "sec:Intro", "sec:Fig"]:
        assert present in text
    for absent in ["SLIDETEXT", "SLIDEALWAYS", "TAGTEXT"]:
        assert absent not in text


def test_deck_text_intent(built):
    deck = built["deck"]
    steps = [deck.text(p) for p in (2, 3, 4)]  # frame "text"
    assert ["REVEALTEXT" in t for t in steps] == [False, True, True]
    assert ["SLIDETEXT" in t for t in steps] == [False, False, True]
    assert not any("PAPERTEXT" in t for t in steps)
    assert "LINKTEXT" in deck.text(1)


def test_deck_reveal_keeps_layout(built):
    # \sporkReveal is \uncover: "Last line." doesn't move when REVEALTEXT appears.
    _need("pdftotext")
    ys = []
    for p in (2, 3):
        out = subprocess.run(["pdftotext", "-bbox", "-f", str(p), "-l", str(p), built["deck"].pdf, "-"],
                             stdout=subprocess.PIPE, text=True).stdout
        ys.append(re.search(r'yMin="([\d.]+)"[^>]*>Last<', out).group(1))
    assert ys[0] == ys[1]


def _fig_expectations():
    # step -> (colors shown, macro texts shown) on slides
    return {
        1: (set(), set()),
        2: ({RED_BG, YELLOW_BG}, {"REVEALMACRO"}),
        3: ({RED_BG, INV_GREEN_BG}, {"REVEALMACRO", "SLIDEMACRO"}),
    }


# Frame "fig" (pages 5-7) and frame "fig_add" (pages 8-10, the same body plus an addition) of deck.tex.
@pytest.mark.parametrize("job,first_page", [("deck", 5), ("deck", 8)])
def test_figure_slides_intent(built, job, first_page):
    doc = built[job]
    for step, (colors, texts) in _fig_expectations().items():
        page = first_page + step - 1
        shown = doc.colors(page)
        for c in FIG_COLORS:
            assert (c in shown) == (c in colors), (job, step, c)
        text = doc.text(page)
        assert "BASENODE" in text
        for t in ["REVEALMACRO", "SLIDEMACRO", "PAPERMACRO"]:
            assert (t in text) == (t in texts), (job, step, t)


def test_figure_paper_intent(built):
    doc = built["paper_fig"]
    shown = doc.colors(1)
    assert {RED_BG, INV_BLUE_BG, VIOLET_BG} <= shown
    assert not ({INV_GREEN_BG, YELLOW_BG} & shown)
    text = doc.text(1)
    assert "REVEALMACRO" in text and "PAPERMACRO" in text
    assert "SLIDEMACRO" not in text


def test_single_frame_build(built):
    b = built["frames_b"].text()
    assert "FRAMEB" in b and "FRAMEBSTEP" in b
    assert "FRAMEA" not in b and "FRAMEC" not in b
    ac = built["frames_ac"].text()
    assert "FRAMEA" in ac and "FRAMEC" in ac and "FRAMEB" not in ac


def test_fonts(built):
    paper = built["paper_fig"].fonts()
    assert "FreeSans" in paper            # figure text is sans in the paper
    assert "LMRoman" in paper             # paper body (caption) is serif
    deck = built["deck"].fonts()
    assert "FreeSans" in deck and "UbuntuMono" in deck.replace(" ", "")


def test_per_slide_addition(built):
    # frame fig_add: the body's nodes plus ADDEDNODE right of the body's `rev` node, on every step;
    # frame fig (same body) and the paper don't have it.
    deck = built["deck"]
    for page in (8, 9, 10):
        words = {w: (x0, x1) for w, x0, x1 in deck.words(page)}
        assert "ADDEDNODE" in words and words["ADDEDNODE"][0] > words["REVEALKEY"][1]
    assert not any("ADDEDNODE" in deck.text(p) for p in (5, 6, 7))
    assert "ADDEDNODE" not in built["paper_fig"].text()


def test_sporkfigure_scale(built):
    # MEASURE at scale 1 and scale 2 (text is scaled with the picture); TikZ's own scale=0.5 inside
    # a node still applies (HALF at 2 x 0.5 is about the size of normal 9pt text).
    widths = [x1 - x0 for w, x0, x1 in built["sporkfig"].words(1) if w == "MEASURE"]
    assert len(widths) == 2, widths
    assert abs(widths[1] / widths[0] - 2) < 0.02, widths
    assert "HALF" in built["sporkfig"].text()


def _width_warnings(log):
    log = log.replace("\n", "")  # unwrap (the test runs xelatex without max_print_line)
    return re.findall(r"Package spork Warning: sporkfigure is ([\d.]+)pt wide after scaling, "
                      r"> \\linewidth ([\d.]+)pt", log)


def test_width_warning_paper(built):
    # sporkfig.tex: figure 1 (~125mm) fits, figure 2 (~2 x 125mm) doesn't; paper_fig fits.
    warnings = _width_warnings(built["sporkfig"].log)
    assert len(warnings) == 1, warnings
    w, lw = (float(x) for x in warnings[0])
    assert w > lw and abs(lw - 469.75) < 0.1
    for job in ["paper_fig", "paper_figfont", "deck"]:
        assert "Package spork Warning" not in built[job].log, job


def test_width_warning_slides(built):
    # sporkfig_deck.tex: frame "fits" (~115mm x 1.2) has no warning, frame "wide" (x 1.5) has one,
    # measured against the frame's 144mm \\linewidth.
    warnings = _width_warnings(built["sporkfig_deck"].log)
    assert len(warnings) == 1, warnings
    w, lw = (float(x) for x in warnings[0])
    assert w > lw and abs(lw - 409.71) < 0.1
    # Overlays inside the scaled figure: STEPTWO only on step 2 of frame "fits".
    assert ["STEPTWO" in built["sporkfig_deck"].text(p) for p in (1, 2)] == [False, True]


# Error cases: each value is (document body, start of pgf's error message naming the node).
# Positive control: sporkfig.tex references all these kinds of names inside the figure that defines them
# (and compiles cleanly, test_compiles_cleanly), so the errors below come from forgetting, not from typos.
_NAME_PREAMBLE = r"""\documentclass{article}
\newif\ifsporkSlides \sporkSlidesfalse
\input{copied_styles/spork_style.tex}
\input{copied_styles/code_to_tex_support.tex}
\begin{document}
"""
_NAME_ERRORS = {
    # misspelled / renamed inside the figure
    "undefined": (r"\begin{sporkfigure}\node (A) {a}; \draw (A) -- (Nope);\end{sporkfigure}",
                  "No shape named `Nope' is known"),
    # a name defined only in an earlier sporkfigure (the silent-wrong-arrow case of stock TikZ)
    "earlier_fig": (r"\begin{sporkfigure}\node (C2) {c};\end{sporkfigure}"
                    r"\begin{sporkfigure}\node (A) {a}; \draw (A) -- (C2);\end{sporkfigure}",
                    "No shape named `C2' is known"),
    # a name defined only in an earlier plain picture
    "earlier_plain": (r"\tikz\node (P) {p};"
                      r"\begin{sporkfigure}\node (A) {a}; \draw (A) -- (P.east);\end{sporkfigure}",
                      "No shape named `P' is known in this sporkfigure"),
    # a later plain picture referring to a sporkfigure's node
    "later_plain": (r"\begin{sporkfigure}\node (C2) {c};\end{sporkfigure}\tikz\draw (0,0) -- (C2);",
                    "No shape named `C2' is known"),
    # alias, pic, matrix cell, coordinate, prefixed names are forgotten too
    "alias": (r"\begin{sporkfigure}\node[alias=Al] (Q) {q};\end{sporkfigure}"
              r"\begin{sporkfigure}\node (A) {a}; \draw (A) -- (Al);\end{sporkfigure}",
              "No shape named `Al' is known"),
    "pic": (r"\begin{sporkfigure}\pic (pp-) {code={\node (inner) {I};}};\end{sporkfigure}"
            r"\begin{sporkfigure}\node (A) {a}; \draw (A) -- (pp-inner);\end{sporkfigure}",
            "No shape named `pp-inner' is known"),
    "matrix": (r"\begin{sporkfigure}\matrix (m) {\node (m1) {M}; \\};\end{sporkfigure}"
               r"\begin{sporkfigure}\node (A) {a}; \draw (A) -- (m1);\end{sporkfigure}",
               "No shape named `m1' is known"),
    "coordinate": (r"\begin{sporkfigure}\coordinate (cc) at (1,1);\end{sporkfigure}"
                   r"\begin{sporkfigure}\node (A) {a}; \draw (A) -- (cc);\end{sporkfigure}",
                   "No shape named `cc' is known"),
    "prefix": (r"\begin{sporkfigure}\begin{scope}[name prefix=p-]\node (X) {x};\end{scope}\end{sporkfigure}"
               r"\begin{sporkfigure}\node (A) {a}; \draw (A) -- (p-X);\end{sporkfigure}",
               "No shape named `p-X' is known"),
}


def _compile_snippet(built, job, body):
    work = built["paper"].workdir
    with open(os.path.join(work, job + ".tex"), "w") as f:
        f.write(_NAME_PREAMBLE + body + "\n\\end{document}\n")
    _xelatex(work, [job + ".tex"])
    return Doc(work, job).log.replace("\n", "")


@pytest.mark.parametrize("case", sorted(_NAME_ERRORS))
def test_node_names_are_local(built, case):
    body, expected = _NAME_ERRORS[case]
    log = _compile_snippet(built, f"names_{case}", body)
    errors = re.findall(r"! Package pgf Error: (No shape named `[^']*' is known[^.]*)", log)
    assert errors and errors[0].startswith(expected), (case, errors)


def test_remember_picture_warning(built):
    log = _compile_snippet(built, "names_remember",
                           r"\begin{sporkfigure}[remember picture]\node {r};\end{sporkfigure}")
    assert "Package spork Warning: remember picture in a sporkfigure" in log


def test_standalone_class_rejected(built):
    work = built["paper"].workdir
    with open(os.path.join(work, "old_fig.tex"), "w") as f:
        f.write("\\documentclass[beamer]{standalone}\n\\newif\\ifsporkSlides \\sporkSlidestrue\n"
                "\\input{copied_styles/spork_style.tex}\n\\begin{document}\\end{document}\n")
    _xelatex(work, ["old_fig.tex"])
    assert "class standalone is no longer supported" in Doc(work, "old_fig").log


def test_figure_fonts_in_paper(built):
    # Paper body is serif; pictures are sans. sporkfigure sets 9pt (\sporkFigFont) and node font keys
    # add to it: font=\bfseries stays sans 9pt, font=\large changes only the size. Plain inline
    # pictures keep the body size.
    doc = built["paper_figfont"]
    text = doc.text()
    for word in ["INLINESIZE10", "BOLDNODE9", "PLAINNODE9", "LARGENODE12"]:
        assert word in text, (word, text)
    fonts = doc.fonts()
    assert "FreeSansBold" in fonts
    assert "LMRoman10-Regular" in fonts           # the body
    assert not re.search(r"LMRoman\S*Bold", fonts)  # the bold node did not fall back to serif bold


def _rule_segments(doc, dpi=200, color=(191, 191, 191)):
    """Horizontal runs of `lightgray` (\\sporkCodeRule) on page 1: [(row, x0_mm, x1_mm)], no antialiasing."""
    _need("pdftoppm")
    out = os.path.join(doc.workdir, f"{doc.job}_rule")
    subprocess.run(["pdftoppm", "-r", str(dpi), "-aa", "no", "-aaVector", "no", "-f", "1", "-l", "1",
                    "-singlefile", doc.pdf, out], check=True)
    with open(out + ".ppm", "rb") as f:
        data = f.read()
    fields = data.split(maxsplit=4)
    width, height = int(fields[1]), int(fields[2])
    pix = fields[4][: width * height * 3]
    target = bytes(color)
    segs = []
    for y in range(height):
        row = pix[y * width * 3:(y + 1) * width * 3]
        xs = [x for x in range(width) if row[x * 3:x * 3 + 3] == target]
        if len(xs) > 20:
            segs.append((y, xs[0] * 25.4 / dpi, (xs[-1] + 1) * 25.4 / dpi))
    # One entry per rule (a 0.4pt rule may cover 1-2 pixel rows).
    rules = []
    for seg in segs:
        if rules and seg[0] - rules[-1][0] <= 2:
            continue
        rules.append(seg)
    return rules


def _close(a, b, tol=0.5):
    return abs(a - b) <= tol


def test_code_rule_width_slides(built):
    # 160mm page, 8mm margins: the frame rule is 144mm (\\linewidth), not running to the page edge;
    # the rule in a 0.4\\textwidth column is 57.6mm.
    rules = _rule_segments(built["rule"])
    assert len(rules) == 2, rules
    (_, x0, x1), (_, c0, c1) = rules
    assert _close(x0, 8) and _close(x1, 152), rules
    assert _close(c0, 8) and _close(c1, 8 + 57.6), rules


def test_code_rule_width_paper(built):
    # Letter paper, 2.54cm margins.
    rules = _rule_segments(built["paper"])
    assert len(rules) == 1, rules
    _, x0, x1 = rules[0]
    assert _close(x0, 25.4) and _close(x1, 215.9 - 25.4), rules
