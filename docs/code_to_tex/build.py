"""Build driver for a code_to_tex document directory (paper + Beamer decks).

    python3 -m code_to_tex.build <dir> [--slides DECK]... [--no-paper]
        [--frame LABEL[,LABEL]] [--skip-code]
        [--png] [--where LABEL] [--arxiv OUT.tar.gz]

Replaces the old ``spork/docs/amspork3_tex.sh``. See ``endgame_plan/plan_slides.md``
("Build driver") for the design. Python 3.10 stdlib only; never uses a shell.

Viewer copies (for evince, which crashes reloading half-written PDFs or PDFs with undefined
references): the paper goes to ``<viewer_copy_dir>/DIR.pdf`` and deck DECK to ``DIR_slides.pdf``
(DECK == DIR), ``DECK.pdf`` (DECK starts with ``DIR_``) or ``DIR_DECK.pdf``. They are written only
when xelatex produced a PDF with no LaTeX errors, exit status 0, and no undefined
references/citations; box warnings never block. Frame builds have no viewer copy.

Figures have no build mode of their own: they are ``sporkfigure`` environments in the paper and
decks, seen in context (``--frame ... --png``, or the paper page found with ``--where``).

Removal policy: the driver only ever deletes or overwrites files it generates itself
(``copied_styles/*`` copies, ``preview/<job>-N.png``, viewer-copy temp files, and its own
temp directories).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time

if __package__ in (None, ""):  # pragma: no cover - run as a plain script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "code_to_tex"

from . import build_aux, build_log  # noqa: E402

PROG = "python3 -m code_to_tex.build"

# Package-internal: tests point this at a fixture directory of stub styles.
LATEX_DIR_ENV = "CODE_TO_TEX_LATEX_DIR"
PACKAGE_LATEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latex")

MAX_PASSES = 4

# Unwrapped logs (kpathsea reads texmf.cnf variables from the environment).
LATEX_ENV = {"max_print_line": "100000", "error_line": "254", "half_error_line": "238"}

CONFIG_DEFAULTS = {
    "check": [],
    "listing_sources": [],
    "extra_styles": [],
    "viewer_copy_dir": "..",
}

_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.+-]*$")
_FRAME_LABEL_RE = re.compile(r"^[^\s{}\\%#,]+$")


class BuildError(Exception):
    """A real failure: aborts the build with exit code 1."""


# ---------------------------------------------------------------------------
# Terminal output


def _use_color(stream) -> bool:
    return hasattr(stream, "isatty") and stream.isatty() and os.environ.get("NO_COLOR") is None and os.environ.get("TERM") != "dumb"


def _c(text: str, code: str, stream=None) -> str:
    stream = stream or sys.stdout
    if _use_color(stream):
        return f"\033[{code}m{text}\033[0m"
    return text


def say(msg: str) -> None:
    print(f"[build] {msg}", flush=True)


def warn(msg: str) -> None:
    print(_c(f"[build] WARNING: {msg}", "33"), flush=True)


# ---------------------------------------------------------------------------
# Arguments


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Build a code_to_tex document directory: refresh copied_styles/, run the check "
            "command and the listing generator, then xelatex the paper (main.tex) and/or "
            "slide decks, rerunning until labels/tikzmarks are stable. Warnings are "
            "summarized at the end; only real failures give a nonzero exit code."
        ),
        epilog=(
            "Typical agent loop for one slide: --slides DECK --frame LABEL --skip-code --png, "
            "then look at preview/DECK_frame-N.png. Use --where LABEL to find which frame "
            "(and pages) contain a \\label."
        ),
    )
    p.add_argument("dir", help="document directory (contains main.tex, decks, tex_build.json); becomes the cwd")
    p.add_argument("--slides", metavar="DECK", action="append", default=[],
                   help="also build slide deck DECK.tex (repeatable). Viewer copy (in viewer_copy_dir, "
                        "default ..): DIR_slides.pdf if DECK is the dir name DIR, DECK.pdf if DECK starts "
                        "with DIR_, else DIR_DECK.pdf; the paper's is DIR.pdf. Copied only after a clean "
                        "build (PDF, no LaTeX errors, xelatex exit 0, no undefined references)")
    p.add_argument("--no-paper", action="store_true",
                   help="do not build main.tex (default: build it if it exists)")
    p.add_argument("--frame", metavar="LABEL[,LABEL]",
                   help="build only these frames of the single --slides DECK (jobname DECK_frame); "
                        "skips the paper and the full deck; never updates viewer copies")
    p.add_argument("--skip-code", action="store_true",
                   help="skip the check command and the listing generator")
    p.add_argument("--png", action="store_true",
                   help="render the PDFs built in this run to preview/<job>-<page>.png at 110 dpi "
                        "(whole pages: paper pages and slides keep their page bounds)")
    p.add_argument("--where", metavar="LABEL",
                   help="report the page(s) and frame(s) containing LABEL from main.aux and the deck .aux "
                        "files (the --slides decks, else every beamer deck in the dir). Query only: builds "
                        "nothing unless a needed .aux is missing, so build first for fresh results")
    p.add_argument("--arxiv", metavar="OUT.tar.gz",
                   help="clean-build main.tex in a temp copy and package its inputs for arXiv")
    return p


@dataclasses.dataclass
class Options:
    doc_dir: str
    doc_dir_arg: str
    slides: list[str]
    build_paper: bool
    frame: list[str] | None
    skip_code: bool
    png: bool
    where: str | None
    arxiv: str | None
    where_only: bool


def _strip_tex(name: str) -> str:
    return name[:-4] if name.endswith(".tex") else name


def parse_args(argv: list[str] | None = None) -> Options:
    parser = make_parser()
    a = parser.parse_args(argv)

    doc_dir = os.path.abspath(a.dir)
    if not os.path.isdir(doc_dir):
        parser.error(f"document directory {a.dir!r} does not exist")
    arxiv = os.path.abspath(a.arxiv) if a.arxiv else None

    slides = []
    for s in a.slides:
        s = _strip_tex(s)
        if not _NAME_RE.match(s):
            parser.error(f"--slides {s!r}: expected a deck name like 'nyc26' (DECK.tex in the document dir)")
        if s not in slides:
            slides.append(s)
    frame = None
    if a.frame is not None:
        frame = [x.strip() for x in a.frame.split(",") if x.strip()]
        if not frame:
            parser.error("--frame: no labels given")
        for x in frame:
            if not _FRAME_LABEL_RE.match(x):
                parser.error(f"--frame: invalid frame label {x!r}")
        if len(slides) != 1:
            parser.error("--frame requires exactly one --slides DECK")
    if arxiv is not None and frame is not None:
        parser.error("--arxiv cannot be combined with --frame")

    build_paper = not a.no_paper and frame is None
    # Pure query: --where with nothing else that selects something to build.
    where_only = a.where is not None and frame is None and arxiv is None and not a.png
    if where_only:
        build_paper = False
    return Options(
        doc_dir=doc_dir,
        doc_dir_arg=a.dir,
        slides=slides,
        build_paper=build_paper,
        frame=frame,
        skip_code=a.skip_code,
        png=a.png,
        where=a.where,
        arxiv=arxiv,
        where_only=where_only,
    )


# ---------------------------------------------------------------------------
# Config


def load_config(path: str = "tex_build.json") -> dict:
    cfg = dict(CONFIG_DEFAULTS)
    cfg = {k: (list(v) if isinstance(v, list) else v) for k, v in cfg.items()}
    if not os.path.exists(path):
        return cfg
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise BuildError(f"{path}: {e}") from e
    if not isinstance(data, dict):
        raise BuildError(f"{path}: expected a JSON object")
    for key, value in data.items():
        if key.startswith("_"):
            continue  # comments
        if key not in CONFIG_DEFAULTS:
            warn(f"{path}: unknown key {key!r} ignored (known: {', '.join(CONFIG_DEFAULTS)})")
            continue
        if key == "viewer_copy_dir":
            if not isinstance(value, str):
                raise BuildError(f"{path}: {key} must be a string")
        elif not (isinstance(value, list) and all(isinstance(x, str) for x in value)):
            raise BuildError(f"{path}: {key} must be a list of strings")
        cfg[key] = value
    return cfg


# ---------------------------------------------------------------------------
# copied_styles


def latex_source_dir() -> str:
    return os.environ.get(LATEX_DIR_ENV) or PACKAGE_LATEX_DIR


def refresh_copied_styles(extra_styles: list[str], dest_dir: str = "copied_styles") -> list[str]:
    """Copy the package's latex/*.tex (+ extra_styles) into dest_dir, read-only. Returns dest paths."""
    src_dir = latex_source_dir()
    try:
        sources = sorted(
            os.path.join(src_dir, n) for n in os.listdir(src_dir) if n.endswith(".tex")
        )
    except OSError as e:
        raise BuildError(f"cannot list LaTeX style directory {src_dir}: {e}") from e
    if not sources:
        raise BuildError(f"no .tex files in LaTeX style directory {src_dir}")
    for extra in extra_styles:
        if not os.path.isfile(extra):
            raise BuildError(f"tex_build.json extra_styles: {extra!r} not found")
        sources.append(os.path.abspath(extra))

    by_name: dict[str, str] = {}
    for src in sources:
        name = os.path.basename(src)
        if name in by_name:
            raise BuildError(f"copied_styles: two sources named {name}: {by_name[name]} and {src}")
        by_name[name] = src

    os.makedirs(dest_dir, exist_ok=True)
    written = []
    for name, src in by_name.items():
        dest = os.path.join(dest_dir, name)
        if os.path.lexists(dest):
            if os.path.islink(dest) or not os.path.isfile(dest):
                raise BuildError(f"{dest} is not a regular file; refusing to overwrite")
            mode = os.stat(dest).st_mode
            os.chmod(dest, mode | stat.S_IWUSR)
        shutil.copyfile(src, dest)
        mode = os.stat(dest).st_mode
        os.chmod(dest, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        written.append(dest)
    stale = sorted(set(os.listdir(dest_dir)) - set(by_name))
    if stale:
        warn(f"{dest_dir}/ has files with no current source (left in place): {', '.join(stale)}")
    return written


# ---------------------------------------------------------------------------
# Code generation


def run_check(cmd: list[str]) -> None:
    if not cmd:
        return
    say("check: " + " ".join(cmd))
    try:
        rc = subprocess.run(cmd, stdin=subprocess.DEVNULL).returncode
    except OSError as e:
        raise BuildError(f"check command {cmd[0]!r} could not be run: {e}") from e
    if rc != 0:
        raise BuildError(f"check command failed (exit {rc}): {' '.join(cmd)}")


def _load_listing():
    """Import code_to_tex.listing lazily (tests may stub it via sys.modules)."""
    return importlib.import_module(f"{__package__}.listing")


def run_listings(sources: list[str], out_dir: str = "code") -> list[str]:
    if not sources:
        return []
    listing = _load_listing()
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    for src in sources:
        try:
            paths = listing.generate(src, out_dir)
        except listing.ListingError as e:
            raise BuildError(f"listing generator: {e}") from e
        written.extend(paths or [])
        say(f"listings: {src} -> {len(paths or [])} files in {out_dir}/")
    return written


# ---------------------------------------------------------------------------
# xelatex


@dataclasses.dataclass
class Job:
    """One xelatex job: produces <name>.pdf from source (a file name or command-line TeX)."""

    name: str  # jobname (output basename)
    tex_input: str  # last xelatex argument
    kind: str  # "paper", "deck", "frame"
    source_file: str  # the .tex file that must exist
    viewer_copy: str | None = None  # destination for the viewer copy, if any


@dataclasses.dataclass
class JobResult:
    job: Job
    passes: int = 0
    pages: int | None = None
    issues: list = dataclasses.field(default_factory=list)
    log_text: str = ""
    seconds: float = 0.0
    failed: str | None = None  # reason no PDF was produced
    exit_code: int | None = None  # xelatex exit status of the last pass (None: not run)

    @property
    def pdf(self) -> str:
        return self.job.name + ".pdf"


def xelatex_command(job: Job, recorder: bool = True) -> list[str]:
    cmd = ["xelatex", "-interaction=nonstopmode"]
    if recorder:
        cmd.append("-recorder")
    if job.kind == "frame" or job.tex_input != job.source_file:
        cmd.append(f"-jobname={job.name}")
    cmd.append(job.tex_input)
    return cmd


def _latex_env() -> dict:
    env = dict(os.environ)
    env.update(LATEX_ENV)
    return env


def _read_log(job_name: str) -> str:
    try:
        with open(job_name + ".log", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _needs_bibtex(job_name: str) -> bool:
    try:
        with open(job_name + ".aux", encoding="utf-8", errors="replace") as f:
            return "\\bibdata{" in f.read()
    except OSError:
        return False


def run_latex(job: Job, max_passes: int = MAX_PASSES, recorder: bool = True) -> JobResult:
    """Run xelatex until the log no longer asks for a rerun and the aux files are stable."""
    res = JobResult(job)
    if not os.path.isfile(job.source_file):
        res.failed = f"{job.source_file} not found"
        return res
    if shutil.which("xelatex") is None:
        raise BuildError("xelatex not found on PATH")
    env = _latex_env()
    cmd = xelatex_command(job, recorder)
    ran_bibtex = False
    t_start = time.monotonic()
    for n in range(1, max_passes + 1):
        before = build_aux.aux_state(job.name)
        t0 = time.monotonic()
        proc = subprocess.run(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
        )
        dt = time.monotonic() - t0
        log_text = _read_log(job.name)
        if not log_text:
            # xelatex could not even start the log; keep its terminal output for the report.
            log_text = proc.stdout.decode("utf-8", errors="replace")
        res.passes = n
        res.exit_code = proc.returncode
        res.log_text = log_text
        res.pages = build_log.output_pages(log_text)
        after = build_aux.aux_state(job.name)

        reasons = []
        if res.pages is not None:
            if build_log.log_requests_rerun(log_text):
                reasons.append("log asks for rerun")
            if after != before:
                reasons.append("aux changed")
            if not ran_bibtex and _needs_bibtex(job.name) and shutil.which("bibtex"):
                ran_bibtex = True
                b = subprocess.run(["bibtex", job.name], stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                reasons.append("bibtex" + ("" if b.returncode == 0 else " (failed, see .blg)"))
        pages = "no PDF" if res.pages is None else f"{res.pages} page{'s' if res.pages != 1 else ''}"
        errs = sum(1 for i in build_log.parse_log(log_text) if i.kind == "error") if res.pages is not None else 0
        err_note = f", {errs} error{'s' if errs != 1 else ''}" if errs else ""
        more = f" -> rerun ({', '.join(reasons)})" if reasons and n < max_passes else ""
        if reasons and n == max_passes:
            more = f" (still unstable after {max_passes} passes: {', '.join(reasons)})"
        say(f"{job.name}: pass {n}: {pages}{err_note}, {dt:.1f}s{more}")
        if res.pages is None or not reasons:
            break
    res.seconds = time.monotonic() - t_start
    res.issues = build_log.parse_log(res.log_text, os.getcwd())
    if res.pages is None:
        res.failed = f"xelatex produced no PDF (see {job.name}.log)"
    return res


# ---------------------------------------------------------------------------
# Viewer copies


def viewer_copy_blocker(res: JobResult) -> str | None:
    """Why res's PDF must not be copied to the viewer location, or None if it may be.

    evince crashes when reloading a half-written PDF or one with undefined references,
    so the copy needs a PDF, no LaTeX errors (``! `` lines), xelatex exit status 0, and no
    undefined references/citations. Overfull/underfull boxes and other warnings never block.
    """
    if res.pages is None:
        return "no PDF produced"
    errors = [i for i in res.issues if i.kind == "error"]
    if errors:
        return f"{len(errors)} LaTeX error{'s' if len(errors) != 1 else ''}"
    if res.exit_code not in (0, None):
        return f"xelatex exited with status {res.exit_code}"
    undefined = [i for i in res.issues if i.kind == "undefined"]
    if undefined:
        return "undefined references/citations"
    return None


def update_viewer_copy(res: JobResult) -> str:
    """Copy the PDF to the viewer location (temp file + rename) if the build is clean. Returns a message."""
    dest = res.job.viewer_copy
    if dest is None:
        return ""
    blocker = viewer_copy_blocker(res)
    if blocker is not None:
        return (f"viewer copy {dest} NOT updated: {blocker} "
                f"(evince crashes reloading such PDFs; the new PDF is {os.path.abspath(res.pdf)})")
    dest_dir = os.path.dirname(dest)
    if not os.path.isdir(dest_dir):
        return f"viewer copy {dest} NOT updated: directory {dest_dir} does not exist"
    tmp = os.path.join(dest_dir, "." + os.path.basename(dest) + ".tmp")
    shutil.copyfile(res.pdf, tmp)
    os.replace(tmp, dest)
    return f"viewer copy updated: {dest}"


# ---------------------------------------------------------------------------
# PNG previews


def render_png(res: JobResult, out_dir: str = "preview") -> list[str]:
    """Render res's PDF to out_dir/<job>-<page>.png (old PNGs of this job removed first).

    Pages are not cropped: placement on the paper page or slide is part of what to look at.
    """
    if res.pages is None:
        return []
    if shutil.which("pdftoppm") is None:
        warn("pdftoppm not found; cannot render PNGs")
        return []
    os.makedirs(out_dir, exist_ok=True)
    job = res.job.name
    pat = re.compile(rf"^{re.escape(job)}-\d+\.png$")
    for n in os.listdir(out_dir):
        if pat.match(n):
            os.remove(os.path.join(out_dir, n))
    proc = subprocess.run(
        ["pdftoppm", "-r", "110", "-png", res.pdf, os.path.join(out_dir, job)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        warn(f"pdftoppm failed on {res.pdf}: {proc.stdout.decode(errors='replace').strip()}")
    return sorted(
        (os.path.join(out_dir, n) for n in os.listdir(out_dir) if pat.match(n)),
        key=lambda p: int(re.search(r"-(\d+)\.png$", p).group(1)),
    )


# ---------------------------------------------------------------------------
# Lint


def _fls_inputs(job_name: str) -> list[str]:
    """Project-relative files read by a job according to its .fls (empty if no .fls)."""
    try:
        with open(job_name + ".fls", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    cwd = os.getcwd()
    pwd = cwd
    out = []
    for line in lines:
        if line.startswith("PWD "):
            pwd = line[4:]
            continue
        if not line.startswith("INPUT "):
            continue
        path = line[6:]
        absolute = os.path.normpath(path if os.path.isabs(path) else os.path.join(pwd, path))
        rel = os.path.relpath(absolute, cwd)
        if rel.startswith("..") or os.path.isabs(rel):
            continue
        if not os.path.isfile(rel):
            continue  # e.g. "INPUT extractbb --version" (a recorded command, not a file)
        if rel not in out:
            out.append(rel)
    return out


LINT_EXCLUDE_DIRS = ("copied_styles", "code", "samples", "preview")


def lint_sources(job_names: list[str]) -> list:
    """Lint .tex files (in the document dir) read by the given jobs: bare overlay specs, and
    host-relative sizes in figure bodies (fig_*_body.tex)."""
    files: list[str] = []
    for job in job_names:
        for rel in _fls_inputs(job):
            if not rel.endswith(".tex"):
                continue
            if rel.split(os.sep, 1)[0] in LINT_EXCLUDE_DIRS:
                continue
            if rel not in files:
                files.append(rel)
    issues = []
    for rel in files:
        try:
            with open(rel, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        issues.extend(build_log.lint_overlay_text(text, rel))
        issues.extend(build_log.lint_figure_text(text, rel))
        issues.extend(build_log.lint_script_text(text, rel))
    return issues


# ---------------------------------------------------------------------------
# Summary


def format_summary(results: list[JobResult], lint: list, notes: list[str], stream=None) -> str:
    stream = stream or sys.stdout
    width = 78
    lines: list[str] = []
    any_problem = False
    for res in results:
        head = f"{res.job.name}.pdf: "
        if res.pages is None:
            head += f"FAILED ({res.failed})"
        else:
            head += f"{res.pages} page{'s' if res.pages != 1 else ''}, {res.passes} pass{'es' if res.passes != 1 else ''}, {res.seconds:.1f}s"
        lines.append(head)
        # The full xelatex output, for reading errors with context; [NAME.log:N] below points into it.
        log_name = res.job.name + ".log"
        lines.append(f"  LaTeX log: {os.path.abspath(log_name)}")
        grouped = build_log.group_issues(res.issues)
        for kind, items in grouped.items():
            any_problem = True
            total = sum(c for _i, c in items)
            lines.append(f"  {build_log.KIND_TITLES[kind]}: {total}"
                         + (f" ({len(items)} distinct)" if len(items) != total else ""))
            limit = build_log.SHOW_LIMIT[kind]
            for issue, count in items[:limit]:
                where = f"{issue.where}: " if issue.where else ""
                times = f"  (x{count})" if count > 1 else ""
                at_log = f"  [{log_name}:{issue.log_line}]" if issue.log_line else ""
                lines.append(f"    {where}{issue.text}{times}{at_log}")
                lines.extend("      | " + ln[:200] for ln in issue.context)
            if len(items) > limit:
                lines.append(f"    ... {len(items) - limit} more (see {log_name})")
        if res.pages is None:
            any_problem = True
            tail = [ln for ln in res.log_text.splitlines() if ln.strip()][-8:]
            if tail:
                lines.append(f"  last lines of {log_name}:")
                lines.extend("    " + ln[:200] for ln in tail)
    if lint:
        any_problem = True
        counts: dict = {}
        for issue in lint:
            counts[issue] = counts.get(issue, 0) + 1
        lines.append(f"Lint (bare overlay specs, host-relative sizes in figure bodies, \\text.. in sub/superscripts): {len(lint)}")
        items = list(counts.items())
        for issue, count in items[:15]:
            times = f"  (x{count})" if count > 1 else ""
            lines.append(f"    {issue.where}: {issue.text}{times}")
        if len(items) > 15:
            lines.append(f"    ... {len(items) - 15} more")
    lines.extend(notes)
    if not lines:
        return ""
    title = " BUILD WARNINGS " if any_problem else " BUILD SUMMARY "
    bar = "=" * ((width - len(title)) // 2)
    top = bar + title + bar
    bottom = "=" * len(top)
    code = "1;31" if any(r.pages is None or build_log.has_errors(r.issues) for r in results) else ("1;33" if any_problem else "1;32")
    out = [_c(top, code, stream)]
    out.extend(lines)
    out.append(_c(bottom, code, stream))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main


def deck_viewer_name(deck: str, doc_basename: str) -> str:
    """Viewer copy file name of deck DECK in document dir DIR (the paper's is always DIR.pdf).

    DECK == DIR -> DIR_slides.pdf; DECK starts with DIR_ -> DECK.pdf; else DIR_DECK.pdf.
    """
    if deck == doc_basename:
        return f"{doc_basename}_slides.pdf"
    if deck.startswith(doc_basename + "_"):
        return f"{deck}.pdf"
    return f"{doc_basename}_{deck}.pdf"


def _deck_job(deck: str, cfg: dict, doc_basename: str) -> Job:
    viewer = os.path.join(os.path.abspath(cfg["viewer_copy_dir"]), deck_viewer_name(deck, doc_basename))
    return Job(deck, f"{deck}.tex", "deck", f"{deck}.tex", viewer)


def plan_jobs(opts: Options, cfg: dict) -> list[Job]:
    doc_basename = os.path.basename(opts.doc_dir.rstrip(os.sep))
    jobs: list[Job] = []
    if opts.frame is not None:
        deck = opts.slides[0]
        labels = ",".join(opts.frame)
        jobs.append(Job(
            f"{deck}_frame",
            f"\\def\\sporkOnlyFrame{{{labels}}}\\input{{{deck}.tex}}",
            "frame",
            f"{deck}.tex",
        ))
        return jobs
    if opts.build_paper and os.path.isfile("main.tex"):
        viewer = os.path.join(os.path.abspath(cfg["viewer_copy_dir"]), f"{doc_basename}.pdf")
        jobs.append(Job("main", "main.tex", "paper", "main.tex", viewer))
    for deck in opts.slides:
        jobs.append(_deck_job(deck, cfg, doc_basename))
    return jobs


_DECK_CLASS_RE = re.compile(r"^\s*\\documentclass\s*(\[[^\]]*\])?\s*\{beamer\}", re.MULTILINE)


def _deck_names_in_dir() -> list[str]:
    out = []
    for n in sorted(os.listdir(".")):
        if not n.endswith(".tex"):
            continue
        try:
            with open(n, encoding="utf-8", errors="replace") as f:
                head = f.read(4000)
        except OSError:
            continue
        if _DECK_CLASS_RE.search(head):
            out.append(n[:-4])
    return out


def run(opts: Options) -> int:
    os.chdir(opts.doc_dir)
    doc_basename = os.path.basename(opts.doc_dir.rstrip(os.sep))
    cfg = load_config()

    # --where alone: query existing aux files; build only what is missing.
    if opts.where_only:
        missing_decks = [d for d in opts.slides if not os.path.isfile(f"{d}.aux")]
        need_paper = (not opts.slides and os.path.isfile("main.tex")
                      and not os.path.isfile("main.aux"))
        if not missing_decks and not need_paper:
            hits, exact = build_aux.where(opts.where, _where_paths(opts))
            print("\n".join(build_aux.format_where(opts.where, hits, exact, opts.doc_dir_arg)))
            return 0
        say("--where: building targets whose .aux is missing")
        opts = dataclasses.replace(opts, slides=missing_decks, build_paper=need_paper)

    refresh_copied_styles(cfg["extra_styles"])
    if not opts.skip_code:
        run_check(cfg["check"])
        run_listings(cfg["listing_sources"])

    if opts.arxiv is not None:
        from . import build_arxiv
        build_arxiv.make_arxiv(opts.arxiv)
        return 0

    jobs = plan_jobs(opts, cfg)
    notes: list[str] = []
    if opts.build_paper and not os.path.isfile("main.tex") and opts.frame is None:
        if not opts.slides:
            notes.append("main.tex not found; nothing to build (use --slides DECK)")
    if opts.frame is not None:
        aux = f"{opts.slides[0]}.aux"
        if os.path.isfile(aux):
            known = build_aux.frame_labels(aux)
            unknown = [x for x in opts.frame if x not in known]
            if unknown:
                warn(f"frame label(s) {', '.join(unknown)} not found in {aux} (from the last full deck build)")
    results: list[JobResult] = []
    for job in jobs:
        res = run_latex(job)
        results.append(res)
        if res.failed and res.passes == 0:
            warn(f"{job.name}: {res.failed}")
        msg = update_viewer_copy(res)
        if msg:
            notes.append(msg)
        if job.kind == "frame" and res.pages is None and "No pages of output" in res.log_text:
            notes.append(f"{job.name}: no pages; are the frame labels right? (frames need [label=...])")

    lint = lint_sources([r.job.name for r in results if r.passes > 0])

    pngs: list[str] = []
    if opts.png:
        for res in results:
            pngs.extend(render_png(res))

    summary = format_summary(results, lint, notes)
    if summary:
        print(summary)
    if pngs:
        print("PNG previews:")
        for p in pngs:
            print("  " + os.path.join(opts.doc_dir, p))

    if opts.where is not None:
        hits, exact = build_aux.where(opts.where, _where_paths(opts))
        print("\n".join(build_aux.format_where(opts.where, hits, exact, opts.doc_dir_arg)))

    failed = [r for r in results if r.pages is None]
    if failed:
        print(_c(f"[build] FAILED: no PDF for {', '.join(r.job.name for r in failed)}", "1;31"), flush=True)
        return 1
    return 0


def _where_paths(opts: Options) -> list[str]:
    decks = opts.slides if opts.slides else _deck_names_in_dir()
    paths = ["main.aux"] + [f"{d}.aux" for d in decks]
    return [p for p in paths if os.path.isfile(p)]


def main(argv: list[str] | None = None) -> int:
    opts = parse_args(argv)
    old_cwd = os.getcwd()
    try:
        return run(opts)
    except BuildError as e:
        print(_c(f"[build] ERROR: {e}", "1;31", sys.stderr), file=sys.stderr, flush=True)
        return 1
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    sys.exit(main())
