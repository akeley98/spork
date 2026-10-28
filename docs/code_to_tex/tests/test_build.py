"""Tests for code_to_tex.build (the build driver) and its helpers build_log / build_aux / build_arxiv.

Run as: cd spork/docs && python3 -m pytest code_to_tex/tests/test_build.py
Compile tests use stub styles from build_fixtures/latex (via CODE_TO_TEX_LATEX_DIR)
and are skipped when xelatex is missing.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tarfile
import textwrap
import types

import pytest

from code_to_tex import build, build_aux, build_log

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "build_fixtures")
STUB_LATEX = os.path.join(FIXTURES, "latex")
HAVE_XELATEX = shutil.which("xelatex") is not None
needs_xelatex = pytest.mark.skipif(not HAVE_XELATEX, reason="xelatex not installed")


@pytest.fixture
def stub_latex(monkeypatch):
    monkeypatch.setenv(build.LATEX_DIR_ENV, STUB_LATEX)


@pytest.fixture
def docdir(tmp_path, stub_latex):
    """A fresh copy of the fixture document dir at tmp_path/mydoc (viewer copies land in tmp_path)."""
    d = tmp_path / "mydoc"
    shutil.copytree(os.path.join(FIXTURES, "mydoc"), d)
    return d


@pytest.fixture(autouse=True)
def keep_cwd():
    cwd = os.getcwd()
    yield
    os.chdir(cwd)


# ---------------------------------------------------------------------------
# Argument parsing


def test_parse_args_resolves_paths(tmp_path, monkeypatch):
    (tmp_path / "doc").mkdir()
    monkeypatch.chdir(tmp_path)
    o = build.parse_args(["doc", "--slides", "deck.tex", "--slides", "deck", "--arxiv", "out.tar.gz"])
    assert o.doc_dir == str(tmp_path / "doc")
    assert o.arxiv == str(tmp_path / "out.tar.gz")
    assert o.slides == ["deck"]
    assert o.build_paper and not o.where_only


def test_parse_args_modes(tmp_path):
    d = str(tmp_path)
    o = build.parse_args([d, "--slides", "deck", "--frame", "a, b"])
    assert o.frame == ["a", "b"] and not o.build_paper
    o = build.parse_args([d, "--slides", "deck", "--no-paper"])
    assert not o.build_paper and o.slides == ["deck"]
    o = build.parse_args([d, "--where", "lab"])
    assert o.where_only and not o.build_paper
    o = build.parse_args([d, "--where", "lab", "--png"])
    assert not o.where_only


@pytest.mark.parametrize("argv", [
    ["--frame", "a"],                                  # no deck
    ["--slides", "a", "--slides", "b", "--frame", "x"],  # two decks
    ["--figure", "fig_dag"],                           # removed: figures are sporkfigures, seen in context
    ["--figure", "fig_dag", "--paper"],
    ["--paper"],
    ["--slides", "../evil"],
    ["--slides", "d", "--frame", "a{b}"],
    ["--slides", "d", "--frame", "x", "--arxiv", "x.tar.gz"],
])
def test_parse_args_errors(tmp_path, argv, capsys):
    with pytest.raises(SystemExit) as e:
        build.parse_args([str(tmp_path)] + argv)
    assert e.value.code == 2


def test_parse_args_missing_dir(tmp_path):
    with pytest.raises(SystemExit):
        build.parse_args([str(tmp_path / "nope")])


# ---------------------------------------------------------------------------
# Config


def test_config_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert build.load_config() == {"check": [], "listing_sources": [], "extra_styles": [], "viewer_copy_dir": ".."}


def test_config_values_and_unknown_key(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tex_build.json").write_text(
        '{"check": ["exocc", "samples.py"], "listing_sources": ["samples.py"], "_comment": "x", "bogus": 1}'
    )
    cfg = build.load_config()
    assert cfg["check"] == ["exocc", "samples.py"]
    assert cfg["listing_sources"] == ["samples.py"]
    assert cfg["viewer_copy_dir"] == ".."
    assert "bogus" in capsys.readouterr().out


@pytest.mark.parametrize("text", ['{"check": "exocc samples.py"}', '{"viewer_copy_dir": 3}', "[1]", "{bad json"])
def test_config_errors(tmp_path, monkeypatch, text):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tex_build.json").write_text(text)
    with pytest.raises(build.BuildError):
        build.load_config()


# ---------------------------------------------------------------------------
# copied_styles


def _writable(path) -> bool:
    return bool(os.stat(path).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def test_copied_styles_copy_and_chmod(tmp_path, monkeypatch):
    src = tmp_path / "latex"
    src.mkdir()
    (src / "spork_style.tex").write_text("v1")
    (src / "code_to_tex_support.tex").write_text("s1")
    (src / "notes.txt").write_text("not copied")
    doc = tmp_path / "doc"
    doc.mkdir()
    (doc / "extra.tex").write_text("e1")
    monkeypatch.setenv(build.LATEX_DIR_ENV, str(src))
    monkeypatch.chdir(doc)

    build.refresh_copied_styles(["extra.tex"])
    cs = doc / "copied_styles"
    assert sorted(os.listdir(cs)) == ["code_to_tex_support.tex", "extra.tex", "spork_style.tex"]
    assert (cs / "spork_style.tex").read_text() == "v1"
    assert not any(_writable(cs / n) for n in os.listdir(cs))

    # Second refresh overwrites the read-only copies.
    (src / "spork_style.tex").write_text("v2")
    build.refresh_copied_styles([])
    assert (cs / "spork_style.tex").read_text() == "v2"
    assert not _writable(cs / "spork_style.tex")
    # extra.tex no longer has a source: left in place (never deleted).
    assert (cs / "extra.tex").exists()


def test_copied_styles_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(build.LATEX_DIR_ENV, str(tmp_path / "missing"))
    with pytest.raises(build.BuildError):
        build.refresh_copied_styles([])
    monkeypatch.setenv(build.LATEX_DIR_ENV, STUB_LATEX)
    with pytest.raises(build.BuildError, match="not found"):
        build.refresh_copied_styles(["nope.tex"])
    (tmp_path / "spork_style.tex").write_text("dup")
    with pytest.raises(build.BuildError, match="two sources"):
        build.refresh_copied_styles(["spork_style.tex"])


def test_default_latex_dir_is_package_relative(monkeypatch):
    monkeypatch.delenv(build.LATEX_DIR_ENV, raising=False)
    assert build.latex_source_dir() == os.path.join(os.path.dirname(build.__file__), "latex")


# ---------------------------------------------------------------------------
# check command and listing generator


def _write_config(d, **cfg):
    import json
    (d / "tex_build.json").write_text(json.dumps(cfg))


def test_check_failure_aborts(docdir, capsys):
    _write_config(docdir, check=[sys.executable, "-c", "print('checking'); raise SystemExit(3)"])
    rc = build.main([str(docdir), "--no-paper"])
    out = capsys.readouterr()
    assert rc == 1
    assert "check command failed (exit 3)" in out.err


def test_check_missing_program(docdir, capsys):
    _write_config(docdir, check=["definitely-not-a-program-xyz"])
    assert build.main([str(docdir), "--no-paper"]) == 1
    assert "could not be run" in capsys.readouterr().err


def test_skip_code_skips_check(docdir):
    _write_config(docdir, check=[sys.executable, "-c", "raise SystemExit(1)"])
    assert build.main([str(docdir), "--no-paper", "--skip-code"]) == 0


class _StubListingError(Exception):
    pass


def _stub_listing(monkeypatch, behavior):
    mod = types.ModuleType("code_to_tex.listing")
    mod.ListingError = _StubListingError
    calls = []

    def generate(src, out_dir):
        calls.append((src, out_dir, os.getcwd()))
        return behavior(src, out_dir)

    mod.generate = generate
    monkeypatch.setitem(sys.modules, "code_to_tex.listing", mod)
    return calls


def test_listing_error_propagates(docdir, monkeypatch, capsys):
    def boom(src, out_dir):
        raise _StubListingError("samples.py:12: unknown color 'mauve'")
    calls = _stub_listing(monkeypatch, boom)
    _write_config(docdir, listing_sources=["samples.py"])
    assert build.main([str(docdir), "--no-paper"]) == 1
    assert "samples.py:12: unknown color 'mauve'" in capsys.readouterr().err
    assert calls == [("samples.py", "code", str(docdir))]


def test_listing_runs_after_check(docdir, monkeypatch):
    order = []

    def gen(src, out_dir):
        order.append("listing")
        assert (docdir / "checked").exists()
        return [os.path.join(out_dir, "x.tex")]
    _stub_listing(monkeypatch, gen)
    _write_config(docdir, check=[sys.executable, "-c", "open('checked','w')"],
                  listing_sources=["a.py", "b.py"])
    assert build.main([str(docdir), "--no-paper"]) == 0
    assert order == ["listing", "listing"]
    assert (docdir / "code").is_dir()


# ---------------------------------------------------------------------------
# Log parsing (canned snippets)


def test_rerun_detection():
    assert build_log.log_requests_rerun("LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right.")
    assert build_log.log_requests_rerun("Package rerunfilecheck Warning: File `main.out' has changed.\n(rerunfilecheck)                Rerun to get outlines right")
    assert build_log.log_requests_rerun("Package tikzmark Warning: Please rerun LaTeX")
    assert not build_log.log_requests_rerun("Output written on main.pdf (3 pages).")


def test_output_pages():
    assert build_log.output_pages("foo\nOutput written on main.pdf (12 pages).\n") == 12
    assert build_log.output_pages("Output written on x.pdf (1 page).") == 1
    assert build_log.output_pages("No pages of output.") is None


CANNED_LOG = textwrap.dedent(r"""
    This is XeTeX, Version 3.141592653-2.6-0.999994 (TeX Live 2022/Debian) (preloaded format=xelatex 2024.1.1)
    (./main.tex
    LaTeX2e <2021-11-15> patch level 1
    (/usr/share/texlive/texmf-dist/tex/latex/base/article.cls
    Document Class: article 2021/10/04 v1.4n Standard LaTeX document class
    (/usr/share/texlive/texmf-dist/tex/latex/base/size10.clo
    File: size10.clo 2021/10/04 v1.4n Standard LaTeX file (size option)
    ))
    (./main.aux)
    (./Intro.tex
    Overfull \hbox (12.5pt too wide) in paragraph at lines 10--12
    []\TU/lmr/m/n/10 some (unbalanced text
     []

    LaTeX Warning: Reference `sec:nope' on page 1 undefined on input line 14.

    Missing character: There is no ☃ (U+2603) in font [lmroman10-regular]:mapping=tex-text;!
    )
    (./fig.tex
    Underfull \vbox (badness 10000) has occurred while \output is active []

    Overfull \hbox (3.0pt too wide) detected at line 7
    [][]
     []

    Overfull \hbox (3.0pt too wide) detected at line 7
    [][]
     []

    )
    ! Undefined control sequence.
    l.20 \foo
             (unbalanced
    The control sequence at the end of the top line
    of your error message was never \def'ed.

    LaTeX Warning: Label `fig:dup' multiply defined.

    Package spork Warning: sporkfigure is 500.0pt wide after scaling, > \linewidth
    (spork)                469.75pt on input line 9.

    Package hyperref Warning: Token not allowed in a PDF string (Unicode):
    (hyperref)                removing `math shift' on input line 30.

    LaTeX Warning: Citation `knuth' on page 2 undefined on input line 25.

    LaTeX Warning: There were undefined references.

    LaTeX Warning: There were multiply-defined labels.
     )
    Output written on main.pdf (2 pages).
    """)


def test_parse_canned_log(tmp_path):
    issues = build_log.parse_log(CANNED_LOG, str(tmp_path))
    by_kind: dict = {}
    for i in issues:
        by_kind.setdefault(i.kind, []).append(i)

    over = by_kind["overfull"]
    assert over[0].where == "Intro.tex:10-12" and "12.5pt too wide" in over[0].text
    assert over[1].where == "fig.tex:7" and over[2] == over[1]
    assert by_kind["underfull"][0].where == "fig.tex"
    undef = by_kind["undefined"]
    assert undef[0].text == "reference `sec:nope' undefined" and undef[0].where == "Intro.tex:14"
    assert any(i.text == "citation `knuth' undefined" and i.where == "main.tex:25" for i in undef)
    assert by_kind["missing_char"][0].text.startswith("no ☃ (U+2603) in font")
    err = by_kind["error"][0]
    assert err.text == "Undefined control sequence." and err.where == "main.tex:20"
    assert by_kind["multiply"][0].text == "label `fig:dup' multiply defined"
    spork = by_kind["spork"][0]
    assert spork.text == "spork: sporkfigure is 500.0pt wide after scaling, > \\linewidth 469.75pt"
    assert spork.where == "main.tex:9"
    assert any("Token not allowed" in i.text for i in by_kind["other"])
    # Noise is dropped.
    assert not any("multiply-defined labels" in i.text for i in issues)

    assert build_log.has_undefined_refs(issues) and build_log.has_errors(issues)
    grouped = build_log.group_issues(issues)
    assert list(grouped) == ["error", "undefined", "multiply", "overfull", "underfull",
                             "missing_char", "spork", "other"]
    assert [c for _i, c in grouped["overfull"]] == [1, 2]
    # Generic "there were undefined references" is dropped when specifics exist.
    assert all(not i.text.startswith("there were") for i, _c in grouped["undefined"])


def test_parse_clean_log():
    log = "(./main.tex (./main.aux) [1] (./main.aux) )\nOutput written on main.pdf (1 page).\n"
    assert build_log.parse_log(log) == []


def test_parse_only_generic_undefined():
    issues = build_log.parse_log("LaTeX Warning: There were undefined references.\n")
    assert build_log.has_undefined_refs(issues)


def test_summary_format():
    job = build.Job("main", "main.tex", "paper", "main.tex")
    res = build.JobResult(job, passes=2, pages=3, issues=build_log.parse_log(CANNED_LOG))
    lint = build_log.lint_overlay_text("\\only<2>{x}\\only<3>{y}\n", "deck.tex")
    s = build.format_summary([res], lint, ["a note"])
    assert "BUILD WARNINGS" in s
    assert "main.pdf: 3 pages, 2 passes" in s
    assert "Overfull boxes: 3 (2 distinct)" in s
    assert "fig.tex:7: Overfull \\hbox (3.0pt too wide)  (x2)" in s
    assert "deck.tex:1: bare overlay spec \\only<...>" in s and "(x2)" in s
    assert "a note" in s
    clean = build.format_summary([build.JobResult(job, passes=1, pages=1)], [], [])
    assert "BUILD SUMMARY" in clean


def test_summary_points_into_latex_log():
    """Errors carry TeX's own context excerpt and their .log line; the summary names the full log."""
    log = textwrap.dedent(r"""
        (./main.tex
        Some preamble chatter
        ! Undefined control sequence.
        l.6 Some text \undefinedmacro
                                     {here} and more.
        The control sequence at the end of the top line
        of your error message was never \def'ed.

        LaTeX Warning: Reference `fig:x' on page 1 undefined on input line 8.
        )
    """).lstrip("\n")
    issues = build_log.parse_log(log)
    err = [i for i in issues if i.kind == "error"][0]
    assert err.log_line == 3 and err.where.endswith("main.tex:6")
    assert err.context == ("! Undefined control sequence.", "l.6 Some text \\undefinedmacro",
                           "                             {here} and more.")
    undef = [i for i in issues if i.kind == "undefined"][0]
    assert undef.log_line == 9
    # log_line/context don't affect grouping of repeated issues.
    assert build_log.Issue("error", "x", "f:1", 5, ("a",)) == build_log.Issue("error", "x", "f:1", 99)
    job = build.Job("main", "main.tex", "paper", "main.tex")
    s = build.format_summary([build.JobResult(job, passes=1, pages=1, issues=issues)], [], [])
    assert f"LaTeX log: {os.path.abspath('main.log')}" in s
    assert "Undefined control sequence.  [main.log:3]" in s
    assert "      | l.6 Some text \\undefinedmacro" in s
    assert "[main.log:9]" in s


# ---------------------------------------------------------------------------
# Lint


def test_lint_overlay_specs():
    text = textwrap.dedent(r"""
        \sporkReveal<2->{ok} \sporkSlideOnly<3>{ok}
        \node[spork reveal=<2->] {ok};
        \only<2>{bad}
        \uncover<2->{bad} \visible<3>{bad}
        \onslide<4>
        \node<3>[draw] {bad}; \draw<2> (0,0) -- (1,1); \path<2> (0,0); \fill<1> (0,0) circle (1pt);
        \pause
        % \only<2>{commented out} \pause
        50\% \only<9>{after escaped percent}
        \pauseX
    """)
    issues = build_log.lint_overlay_text(text, "f.tex")
    lines = [int(i.where.split(":")[1]) for i in issues]
    assert lines == [4, 5, 5, 6, 7, 7, 7, 7, 8, 10]
    assert all(i.kind == "lint" for i in issues)


def test_lint_figure_host_relative_sizes():
    fig = textwrap.dedent(r"""
        % figure body: raw TikZ, no preamble
        \coordinate (o) at (0,0);
        \node [font=\small] {bad}; \node [font=\sporkFigLarge\bfseries] {ok};
        \node [text width=0.3\linewidth] {bad}; \node [text width=40mm] {ok};
        % \Large in a comment is fine
        \sporkListing{early}
        \fontsize{9}{10.5}\selectfont
        \sporkListing{ok}
        \smallskip \textwidthX
    """)
    issues = build_log.lint_figure_text(fig, "fig_dag_body.tex")
    lines = [int(i.where.split(":")[1]) for i in issues]
    assert lines == [4, 5, 7]
    assert "\\small" in issues[0].text and "\\linewidth" in issues[1].text and "sporkListing" in issues[2].text
    assert build_log.lint_figure_text(fig, os.path.join("figs", "fig_dag_body.tex"))[0].where.startswith("figs")
    # Not a figure body (paper, deck, a figure written inline, a listing-mark snippet): sizes are fine there.
    for name in ["deck.tex", "main.tex", "fig_dag.tex", "split_marks.tex", "fig_body.tex"]:
        assert build_log.lint_figure_text(fig, name) == [], name


def test_lint_script_text_commands():
    text = textwrap.dedent(r"""
        $e_\textrm{lo}$ $x^\textsf{pre}$
        $e_{\textrm{hi}}$ $e_ {\texttt{k1}}$
        $e_\mathrm{lo}$ $x^\mathsf{pre}$ $e_\text{Llo}$
        $\sigma|_{\textrm{dom}(\sigma) \setminus X}$
        % $e_\textrm{commented}$
        $e_\textrmX{y}$
    """)
    issues = build_log.lint_script_text(text, "f.tex")
    # Only the unbraced form (which doesn't compile) is flagged; braced forms are a style choice.
    assert [int(i.where.split(":")[1]) for i in issues] == [2, 2]
    assert "\\textrm fails" in issues[0].text and "consider \\mathrm" in issues[0].text
    assert "\\textsf fails" in issues[1].text and "consider \\mathsf" in issues[1].text


# ---------------------------------------------------------------------------
# Aux files: rerun state and --where


DECK_AUX = textwrap.dedent(r"""
    \relax
    \newlabel{intro<1>}{{1}{1}{}{Doc-Start}{}}
    \newlabel{intro}{{1}{1}{}{Doc-Start}{}}
    \newlabel{lab:hello}{{1}{1}{}{Doc-Start}{}}
    \newlabel{intro<2>}{{1}{2}{}{Doc-Start}{}}
    \@writefile{nav}{\headcommand {\beamer@framepages {1}{2}}}
    \newlabel{lab:nolabel}{{2}{3}{}{Doc-Start}{}}
    \@writefile{nav}{\headcommand {\beamer@framepages {3}{3}}}
    \newlabel{second<1>}{{3}{4}{}{Doc-Start}{}}
    \newlabel{second}{{3}{4}{}{Doc-Start}{}}
    \newlabel{second<2>}{{3}{5}{}{Doc-Start}{}}
    \newlabel{second<3>}{{3}{6}{}{Doc-Start}{}}
    \@writefile{nav}{\headcommand {\beamer@framepages {4}{6}}}
    \newlabel{only3<3>}{{4}{7}{}{Doc-Start}{}}
    \@writefile{nav}{\headcommand {\beamer@framepages {7}{7}}}
    """)


def test_read_aux_deck(tmp_path):
    p = tmp_path / "deck.aux"
    p.write_text(DECK_AUX)
    info = build_aux.read_aux(str(p))
    assert [(f.number, f.first_page, f.last_page, f.labels) for f in info.frames] == [
        (1, 1, 2, ["intro"]), (2, 3, 3, []), (3, 4, 6, ["second"]),
        (4, 7, 7, ["only3"])]  # \begin{frame}<3>[label=only3] writes only only3<3>
    hits, exact = build_aux.where("lab:hello", [str(p)])
    assert exact and hits[0].frame.labels == ["intro"] and hits[0].label.page == "1"
    hits, exact = build_aux.where("second", [str(p)])
    assert exact and (hits[0].frame.first_page, hits[0].frame.last_page) == (4, 6)
    hits, exact = build_aux.where("lab:", [str(p)])
    assert not exact and [h.label.name for h in hits] == ["lab:hello", "lab:nolabel"]
    out = "\n".join(build_aux.format_where("lab:nolabel", *build_aux.where("lab:nolabel", [str(p)]), "doc"))
    assert "frame #2" in out and "no label" in out
    assert build_aux.frame_labels(str(p)) == {"intro", "second", "only3"}
    # A frame with an overlay spec only has only3<3>; --where only3 still finds the frame (once).
    hits, exact = build_aux.where("only3", [str(p)])
    assert exact and len(hits) == 1 and hits[0].frame.labels == ["only3"] and hits[0].frame.first_page == 7


def test_read_aux_paper(tmp_path):
    p = tmp_path / "main.aux"
    p.write_text("\\newlabel{sec:a}{{2.1}{7}{A {nested} title}{section.2.1}{}}\n\\@input{sub.aux}\n")
    (tmp_path / "sub.aux").write_text("\\newlabel{fig:b}{{3}{9}{}{figure.3}{}}\n")
    hits, exact = build_aux.where("fig:b", [str(p)])
    assert exact and hits[0].label.page == "9" and hits[0].frame is None
    hits, _ = build_aux.where("sec:a", [str(p)])
    assert hits[0].label.number == "2.1" and hits[0].label.page == "7"
    assert build_aux.where("zzz", [str(p)]) == ([], False)


def test_aux_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert build_aux.aux_state("job") is None
    (tmp_path / "job.aux").write_text("a")
    s1 = build_aux.aux_state("job")
    (tmp_path / "job.nav").write_text("n")
    s2 = build_aux.aux_state("job")
    (tmp_path / "job.aux").write_text("\\pgfsyspdfmark {pgfid1}{100}{200}")
    s3 = build_aux.aux_state("job")
    assert len({s1, s2, s3}) == 3
    assert build_aux.aux_state("job") == s3


# ---------------------------------------------------------------------------
# Rerun loop with a fake xelatex (no TeX needed)


FAKE_XELATEX = textwrap.dedent("""\
    #!{python}
    # Fake xelatex: writes JOB.log / JOB.aux / JOB.pdf. The aux content changes on
    # the first STABLE_AFTER passes (counted in JOB.count), then stays fixed.
    import os, sys
    args = sys.argv[1:]
    job = None
    for a in args:
        if a.startswith("-jobname="):
            job = a.split("=", 1)[1]
    if job is None:
        job = os.path.splitext(args[-1])[0]
    stable_after = int(os.environ.get("FAKE_STABLE_AFTER", "1"))
    n = int(open(job + ".count").read()) + 1 if os.path.exists(job + ".count") else 1
    open(job + ".count", "w").write(str(n))
    open(job + ".args", "a").write(" ".join(args) + "\\n")
    open(job + ".aux", "w").write("aux %d" % min(n, stable_after))
    log = os.environ.get("FAKE_LOG_EXTRA", "")
    if os.environ.get("FAKE_NO_PDF"):
        log += "\\nNo pages of output.\\n"
    else:
        open(job + ".pdf", "w").write("%PDF fake")
        log += "\\nOutput written on %s.pdf (2 pages).\\n" % job
    open(job + ".log", "w").write(log)
    sys.exit(1 if os.environ.get("FAKE_NO_PDF") else 0)
    """)


@pytest.fixture
def fake_tex(tmp_path, monkeypatch):
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    exe = bindir / "xelatex"
    exe.write_text(FAKE_XELATEX.replace("{python}", sys.executable))
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ["PATH"])
    work = tmp_path / "work"
    work.mkdir()
    (work / "doc.tex").write_text("x")
    monkeypatch.chdir(work)
    return work


def _count(work, job="doc"):
    return int((work / f"{job}.count").read_text())


def test_rerun_until_aux_stable(fake_tex, monkeypatch):
    monkeypatch.setenv("FAKE_STABLE_AFTER", "2")
    res = build.run_latex(build.Job("doc", "doc.tex", "paper", "doc.tex"))
    # pass 1: aux created; pass 2: aux changed; pass 3: stable.
    assert res.passes == 3 and res.pages == 2 and _count(fake_tex) == 3
    # A rebuild with an unchanged aux needs only one pass.
    res = build.run_latex(build.Job("doc", "doc.tex", "paper", "doc.tex"))
    assert res.passes == 1


def test_rerun_cap(fake_tex, monkeypatch):
    monkeypatch.setenv("FAKE_STABLE_AFTER", "100")
    res = build.run_latex(build.Job("doc", "doc.tex", "paper", "doc.tex"))
    assert res.passes == build.MAX_PASSES == 4


def test_rerun_on_log_request(fake_tex, monkeypatch, capsys):
    (fake_tex / "doc.aux").write_text("aux 1")  # aux will not change
    monkeypatch.setenv("FAKE_LOG_EXTRA", "LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right.")
    res = build.run_latex(build.Job("doc", "doc.tex", "paper", "doc.tex"))
    assert res.passes == 4  # the fake keeps asking
    assert "log asks for rerun" in capsys.readouterr().out


def test_no_pdf_stops_and_fails(fake_tex, monkeypatch):
    monkeypatch.setenv("FAKE_NO_PDF", "1")
    monkeypatch.setenv("FAKE_STABLE_AFTER", "100")
    res = build.run_latex(build.Job("doc", "doc.tex", "paper", "doc.tex"))
    assert res.passes == 1 and res.pages is None and "no PDF" in res.failed


def test_stale_pdf_does_not_count(fake_tex, monkeypatch):
    (fake_tex / "doc.pdf").write_text("old")
    monkeypatch.setenv("FAKE_NO_PDF", "1")
    res = build.run_latex(build.Job("doc", "doc.tex", "paper", "doc.tex"))
    assert res.pages is None


def test_frame_command_line(fake_tex):
    job = build.plan_jobs(
        build.Options(str(fake_tex), ".", ["doc"], False, ["a", "b"], True, False, None, None, False),
        build.CONFIG_DEFAULTS,
    )[0]
    assert job.name == "doc_frame"
    build.run_latex(job)
    args = (fake_tex / "doc_frame.args").read_text().split()
    assert args[:3] == ["-interaction=nonstopmode", "-recorder", "-jobname=doc_frame"]
    assert args[3] == "\\def\\sporkOnlyFrame{a,b}\\input{doc.tex}"


# ---------------------------------------------------------------------------
# Viewer copies


def _result(tmp_path, issues=(), pages=1, name="main", exit_code=0):
    (tmp_path / f"{name}.pdf").write_text("%PDF new")
    job = build.Job(name, f"{name}.tex", "paper", f"{name}.tex", str(tmp_path / "view" / "doc.pdf"))
    return build.JobResult(job, passes=1, pages=pages, issues=list(issues), exit_code=exit_code)


def test_viewer_copy_clean(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "view").mkdir()
    msg = build.update_viewer_copy(_result(tmp_path))
    assert "updated" in msg
    assert (tmp_path / "view" / "doc.pdf").read_text() == "%PDF new"
    assert os.listdir(tmp_path / "view") == ["doc.pdf"]  # no temp file left


def test_viewer_copy_rules(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "view").mkdir()
    (tmp_path / "view" / "doc.pdf").write_text("old")
    blocked = {
        "undefined references": build_log.parse_log("LaTeX Warning: Reference `x' on page 1 undefined on input line 3.\n"),
        "undefined references/citations": build_log.parse_log("LaTeX Warning: Citation `k' on page 1 undefined on input line 3.\n"),
        "1 LaTeX error": build_log.parse_log("! Undefined control sequence.\nl.3 \\foo\n\n"),
    }
    for reason, issues in blocked.items():
        msg = build.update_viewer_copy(_result(tmp_path, issues))
        assert "NOT updated" in msg and reason in msg, msg
        assert str(tmp_path / "main.pdf") in msg  # points at the fresh PDF instead
    msg = build.update_viewer_copy(_result(tmp_path, pages=None))
    assert "NOT updated" in msg and "no PDF" in msg
    msg = build.update_viewer_copy(_result(tmp_path, exit_code=1))
    assert "NOT updated" in msg and "exited with status 1" in msg
    assert (tmp_path / "view" / "doc.pdf").read_text() == "old"
    # Box warnings and other warnings never block the copy.
    warns = build_log.parse_log("Overfull \\hbox (1pt too wide) detected at line 2\n\n"
                                "Underfull \\hbox (badness 10000) detected at line 4\n\n"
                                "LaTeX Warning: Something odd on input line 5.\n")
    assert warns and not build.viewer_copy_blocker(_result(tmp_path, warns))
    assert "updated" in build.update_viewer_copy(_result(tmp_path, warns))
    assert (tmp_path / "view" / "doc.pdf").read_text() == "%PDF new"
    shutil.rmtree(tmp_path / "view")
    assert "does not exist" in build.update_viewer_copy(_result(tmp_path))


def test_run_latex_records_exit_code(fake_tex, monkeypatch):
    res = build.run_latex(build.Job("doc", "doc.tex", "paper", "doc.tex"))
    assert res.exit_code == 0
    monkeypatch.setenv("FAKE_NO_PDF", "1")
    res = build.run_latex(build.Job("doc", "doc.tex", "paper", "doc.tex"))
    assert res.exit_code == 1


@pytest.mark.parametrize("deck,dirname,expected", [
    ("nyc26", "amspork3", "amspork3_nyc26.pdf"),
    ("beamer_nyc25_test", "beamer_nyc25_test", "beamer_nyc25_test_slides.pdf"),
    ("amspork3_nyc26", "amspork3", "amspork3_nyc26.pdf"),
    ("amspork3x", "amspork3", "amspork3_amspork3x.pdf"),  # prefix without the underscore
])
def test_deck_viewer_name(deck, dirname, expected):
    assert build.deck_viewer_name(deck, dirname) == expected


def test_viewer_copy_paths(tmp_path, monkeypatch):
    d = tmp_path / "mydoc"
    d.mkdir()
    (d / "main.tex").write_text("x")
    monkeypatch.chdir(d)
    o = build.parse_args([str(d), "--slides", "mydoc", "--slides", "talk", "--slides", "mydoc_v2"])
    views = [os.path.basename(j.viewer_copy) for j in build.plan_jobs(o, build.CONFIG_DEFAULTS)]
    assert views == ["mydoc.pdf", "mydoc_slides.pdf", "mydoc_talk.pdf", "mydoc_v2.pdf"]


def test_frame_jobs_have_no_viewer_copy(tmp_path):
    o = build.parse_args([str(tmp_path), "--slides", "d", "--frame", "x"])
    assert all(j.viewer_copy is None for j in build.plan_jobs(o, build.CONFIG_DEFAULTS))


# ---------------------------------------------------------------------------
# Compile tests (real xelatex, stub styles)


def _pdf_pages(log_path) -> int | None:
    with open(log_path, encoding="utf-8", errors="replace") as f:
        return build_log.output_pages(f.read())


@needs_xelatex
def test_full_build_paper_and_deck(docdir, capsys):
    rc = build.main([str(docdir), "--slides", "deck"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert _pdf_pages(docdir / "main.log") == 1
    assert _pdf_pages(docdir / "deck.log") == 7  # 2 + 1 + 3 + 1 overlay pages
    assert (docdir.parent / "mydoc.pdf").exists() and (docdir.parent / "mydoc_deck.pdf").exists()
    assert "viewer copy updated" in out
    assert "main: pass 2" in out  # labels needed a second pass
    # Lint reports the bare specs in the figure body (read by main.tex) and the deck,
    # and the host-relative size in the body.
    assert "fig_demo_body.tex:7: bare overlay spec \\uncover<...>" in out
    assert "fig_demo_body.tex:8: bare overlay spec \\node<...>" in out
    assert "fig_demo_body.tex:9: figure body uses \\small" in out
    assert "deck.tex:13" in out
    assert not os.access(docdir / "copied_styles" / "spork_style.tex", os.W_OK)


@needs_xelatex
def test_paper_only_default_and_no_paper(docdir, capsys):
    assert build.main([str(docdir)]) == 0
    assert (docdir / "main.pdf").exists() and not (docdir / "deck.pdf").exists()
    os.remove(docdir / "main.pdf")
    assert build.main([str(docdir), "--no-paper", "--slides", "deck"]) == 0
    assert (docdir / "deck.pdf").exists() and not (docdir / "main.pdf").exists()


@needs_xelatex
def test_undefined_reference_blocks_viewer_copy(docdir, capsys):
    deck = docdir / "deck.tex"
    deck.write_text(deck.read_text().replace("Done.", "Done \\ref{no:such:label}."))
    (docdir.parent / "mydoc_deck.pdf").write_text("old copy")
    rc = build.main([str(docdir), "--no-paper", "--slides", "deck"])
    out = capsys.readouterr().out
    assert rc == 0  # warnings are not fatal
    assert "reference `no:such:label' undefined" in out
    assert "NOT updated" in out
    assert (docdir.parent / "mydoc_deck.pdf").read_text() == "old copy"
    assert (docdir / "deck.pdf").exists()


@needs_xelatex
def test_latex_error_without_pdf_fails(docdir, capsys):
    (docdir / "main.tex").write_text("\\documentclass{article}\n\\begin{document}\n\\input{missing_file}\n")
    rc = build.main([str(docdir)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAILED" in out and "LaTeX errors" in out


@needs_xelatex
def test_latex_error_with_pdf_is_not_fatal(docdir, capsys):
    main = docdir / "main.tex"
    main.write_text(main.read_text().replace("\\myDocMacro", "\\myDocMacro \\undefinedMacroXYZ"))
    rc = build.main([str(docdir)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "LaTeX errors" in out and "main.tex:8: Undefined control sequence." in out
    # ... but the viewer copy is not updated (evince crashes on such PDFs).
    assert "NOT updated: 1 LaTeX error" in out
    assert not (docdir.parent / "mydoc.pdf").exists()


@needs_xelatex
def test_frame_build(docdir, capsys):
    assert build.main([str(docdir), "--slides", "deck", "--frame", "steps", "--png"]) == 0
    out = capsys.readouterr().out
    assert _pdf_pages(docdir / "deck_frame.log") == 3
    assert not (docdir / "deck.pdf").exists() and not (docdir / "main.pdf").exists()
    assert not (docdir.parent / "mydoc_deck.pdf").exists()
    pngs = sorted(os.listdir(docdir / "preview"))
    assert pngs == ["deck_frame,steps,1.png", "deck_frame,steps,2.png", "deck_frame,steps,3.png"]
    assert f"PNG previews (paths relative to {docdir}):" in out
    assert "  preview/deck_frame,steps,{1..3}.png\n" in out
    # Two frames; stale PNGs of this job are replaced, other jobs' kept (also deck_frame_x, whose
    # name extends this job's).
    (docdir / "preview" / "other,p1.png").write_bytes(b"x")
    (docdir / "preview" / "deck_frame,old,1.png").write_bytes(b"x")
    (docdir / "preview" / "deck_frame_x,p1.png").write_bytes(b"x")
    assert build.main([str(docdir), "--slides", "deck", "--frame", "intro,last", "--png", "--skip-code"]) == 0
    assert _pdf_pages(docdir / "deck_frame.log") == 3  # 2 + 1
    assert sorted(os.listdir(docdir / "preview")) == ["deck_frame,intro,1.png", "deck_frame,intro,2.png",
                                                      "deck_frame,last,1.png", "deck_frame_x,p1.png",
                                                      "other,p1.png"]


@needs_xelatex
def test_frame_unknown_label(docdir, capsys):
    assert build.main([str(docdir), "--no-paper", "--slides", "deck"]) == 0
    capsys.readouterr()
    rc = build.main([str(docdir), "--slides", "deck", "--frame", "nosuchframe"])
    out = capsys.readouterr().out
    assert "nosuchframe not found in deck.aux" in out
    assert rc == 1 and "are the frame labels right" in out


@needs_xelatex
def test_deck_png_names_and_duplicate_label(docdir, capsys):
    deck = docdir / "deck.tex"
    deck.write_text(deck.read_text().replace("[label=last]", "[label=intro]"))
    assert build.main([str(docdir), "--no-paper", "--slides", "deck", "--png"]) == 0
    out = capsys.readouterr().out
    assert sorted(os.listdir(docdir / "preview")) == [
        "deck,,f2,1.png", "deck,intro,f1,1.png", "deck,intro,f1,2.png", "deck,intro,f4,1.png",
        "deck,steps,1.png", "deck,steps,2.png", "deck,steps,3.png"]
    assert "  preview/deck,steps,{1..3}.png  (frame 3, pages 4-6)\n" in out
    assert "  preview/deck,,f2,1.png  (frame 2, page 3)\n" in out
    assert "deck: frame label 'intro' is used by frames 1, 4" in out


def test_png_names_without_aux(tmp_path):
    # No .aux (or pages no frame claims): page-numbered names, padded to a fixed width.
    names = build.png_names("deck", "deck", 3, str(tmp_path / "none.aux"))
    assert names == [(["deck,p01", "deck,p02", "deck,p03"], [1, 2, 3], "")]


def test_png_names_odd_labels(tmp_path):
    # Only "/" is rewritten (to %); labels that look like generated names stay distinct from them.
    labels = ["a/b", "a_b", None, "f3", "x:y", "über", "../up"]
    lines = []
    for i, lab in enumerate(labels, 1):
        if lab is not None:
            lines.append(f"\\newlabel{{{lab}<1>}}{{{{{i}}}{{{i}}}}}")
        lines.append(f"\\@writefile{{nav}}{{\\headcommand {{\\beamer@framepages {{{i}}}{{{i}}}}}}}")
    aux = tmp_path / "deck.aux"
    aux.write_text("\n".join(lines) + "\n")
    groups = build.png_names("deck", "deck", len(labels), str(aux))
    names = [n for g in groups for n in g[0]]
    assert names == ["deck,a%b,1", "deck,a_b,1", "deck,,f3,1", "deck,f3,1",
                     "deck,x:y,1", "deck,über,1", "deck,..%up,1"]
    assert len(set(names)) == len(names)


def test_png_group_describe():
    assert build.PngGroup(["preview/d,x,1.png"]).describe() == "preview/d,x,1.png"
    g = build.PngGroup([f"preview/d,x,{i:02d}.png" for i in range(1, 12)], "frame 2, pages 3-13")
    assert g.describe() == "preview/d,x,{01..11}.png  (frame 2, pages 3-13)"


def _png_size(path) -> tuple[int, int]:
    with open(path, "rb") as f:
        head = f.read(24)
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


@needs_xelatex
def test_paper_png_is_whole_page(docdir, capsys):
    # --png renders whole pages (no cropping): a letter page at 110 dpi.
    assert build.main([str(docdir), "--png"]) == 0
    assert os.listdir(docdir / "preview") == ["main,p01.png"]
    w, h = _png_size(docdir / "preview" / "main,p01.png")
    assert abs(w - 935) <= 2 and abs(h - 1210) <= 2, (w, h)


@needs_xelatex
def test_where(docdir, capsys):
    assert build.main([str(docdir), "--slides", "deck"]) == 0
    capsys.readouterr()
    assert build.main([str(docdir), "--where", "lab:steps"]) == 0
    out = capsys.readouterr().out
    assert "deck.aux: `lab:steps' page 4; frame #3 [label=steps], pages 4-6" in out
    assert "--slides deck --frame steps" in out
    assert "[build]" not in out  # pure query: nothing built
    assert build.main([str(docdir), "--where", "fig:fig"]) == 0
    assert "main.aux: `fig:fig' page 1" in capsys.readouterr().out
    assert build.main([str(docdir), "--slides", "deck", "--where", "unlabelled"]) == 0
    out = capsys.readouterr().out
    assert "lab:unlabelled" in out and "frame has no label" in out


@needs_xelatex
def test_where_builds_missing_aux(docdir, capsys):
    assert build.main([str(docdir), "--slides", "deck", "--where", "intro"]) == 0
    out = capsys.readouterr().out
    assert "deck: pass 1" in out and "main: pass" not in out
    assert "frame #1 [label=intro], pages 1-2" in out


@needs_xelatex
def test_arxiv(docdir, tmp_path, capsys):
    assert build.main([str(docdir), "--slides", "deck"]) == 0
    before = sorted(os.listdir(docdir))
    out_tar = tmp_path / "sub.tar.gz"
    assert build.main([str(docdir), "--arxiv", str(out_tar)]) == 0
    assert "verification OK" in capsys.readouterr().out
    with tarfile.open(out_tar) as tar:
        names = sorted(tar.getnames())
        readme = tar.extractfile("00README.json").read().decode()
    assert names == ["00README.json", "copied_styles/code_to_tex_support.tex",
                     "copied_styles/spork_style.tex", "fig_demo_body.tex", "main.tex", "mydoc_defs.tex"]
    assert '"compiler": "xelatex"' in readme and '"usage": "toplevel"' in readme
    assert sorted(os.listdir(docdir)) == before  # the document dir was not touched
