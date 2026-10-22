"""LaTeX-level tests for latex/spork_style.tex (the master paper/slides style).

Compiles the fixture documents in tests/style_fixtures/ with xelatex in a temp dir laid out like a
document directory: copied_styles/ holds a copy of spork_style.tex plus an empty stub
code_to_tex_support.tex (so these tests don't depend on the listing support file).

Checks: page counts (overlay steps), paper/slide visibility of every overlay intent
(text via pdftotext, TikZ fills via exact pixel colors from pdftoppm), the \\sporkOnlyFrame
single-frame build, fonts (incl. `spork figure` / \\sporkFigFont), \\sporkCodeRule width, and the
\\sporkFigMaxWidth warning.
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

# Fill colors used in fixtures/fig.tex (values from spork_style.tex).
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
        # Beamer/standalone write one /Type /Page per page; avoid needing pdfinfo.
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

    def fonts(self):
        _need("pdffonts")
        return subprocess.run(["pdffonts", self.pdf], stdout=subprocess.PIPE, text=True).stdout


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
    for job in ["fig", "paper", "paper_fig", "deck", "frames", "rule", "paper_figfont"]:
        _xelatex(work, [job + ".tex"], passes=2)
        docs[job] = Doc(work, job)
    _xelatex(work, ["-jobname=frames_b", r"\def\sporkOnlyFrame{b}\input{frames.tex}"])
    docs["frames_b"] = Doc(work, "frames_b")
    _xelatex(work, ["-jobname=frames_ac", r"\def\sporkOnlyFrame{a,c}\input{frames.tex}"])
    docs["frames_ac"] = Doc(work, "frames_ac")
    return docs


@pytest.mark.parametrize("job", ["fig", "paper", "paper_fig", "deck", "frames", "frames_b", "frames_ac",
                                 "rule", "paper_figfont"])
def test_compiles_cleanly(built, job):
    doc = built[job]
    errors = [l for l in doc.log.splitlines() if l.startswith("!")]
    assert not errors, errors
    assert os.path.exists(doc.pdf)
    # \sporkPaperLink must not leave undefined references on slides; paper labels resolve in 2 passes.
    assert "There were undefined references" not in doc.log
    assert not re.search(r"(Reference|Hyper reference) `[^']*' .*undefined", doc.log)


@pytest.mark.parametrize("job,pages", [
    ("fig", 3), ("paper", 1), ("paper_fig", 1), ("deck", 8), ("frames", 4),
    ("frames_b", 2), ("frames_ac", 2), ("rule", 1), ("paper_figfont", 1),
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


@pytest.mark.parametrize("job,first_page", [("fig", 1), ("deck", 5)])
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


def test_fig_max_width_warning(built, tmp_path):
    work = built["fig"].workdir
    with open(os.path.join(work, "fig.tex")) as f:
        src = f.read()
    with open(os.path.join(work, "figw.tex"), "w") as f:
        f.write(src.replace(r"\begin{document}", "\\sporkFigMaxWidth{3cm}\n\\begin{document}", 1))
    _xelatex(work, ["figw.tex"])
    log = Doc(work, "figw").log
    assert "Package spork Warning: picture width" in log
    assert "Package spork Warning" not in built["fig"].log


def test_fig_page_is_wide(built):
    # Standalone figures use [beamer,aspectratio=169]: 160mm x 90mm pages.
    _need("pdfinfo")
    out = subprocess.run(["pdfinfo", built["fig"].pdf], stdout=subprocess.PIPE, text=True).stdout
    w, h = (float(x) for x in re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", out).groups())
    assert abs(w / h - 16 / 9) < 0.01


def test_figure_fonts_in_paper(built):
    # Paper body is serif; pictures are sans. `spork figure` sets 9pt (\sporkFigFont) and node font keys
    # add to it: font=\bfseries stays sans 9pt, font=\large changes only the size. Inline pictures
    # without the key keep the body size.
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
