"""--arxiv packaging for code_to_tex.build (port of ~/amspork_arxiv_14930/build_arxiv.sh).

1. Copy the document directory (after code generation) into a temp dir, minus build
   by-products, and do a clean xelatex build of main.tex there with -recorder
   (so nothing in the real document dir is deleted).
2. Package exactly the project-relative files listed as INPUT in main.fls, minus
   slides/samples/previews/standalone-figure PDFs and the job's own by-products,
   plus 00README.json (main.tex toplevel, xelatex).
3. Verify: extract into another temp dir and rebuild twice without -recorder.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import tarfile
import tempfile
import time

from . import build, build_log

JOB = "main"
BYPRODUCT_EXTS = (".aux", ".log", ".out", ".fls", ".nav", ".snm", ".toc", ".vrb", ".synctex.gz", ".blg")
EXCLUDE_DIRS = ("samples", "preview")
README = {
    "process": {"compiler": "xelatex"},
    "sources": [{"filename": "main.tex", "usage": "toplevel"}],
}


def _copy_ignore(src_dir: str, names: list[str]) -> set[str]:
    ignored = set()
    top = os.path.abspath(src_dir) == os.path.abspath(".")
    for n in names:
        if n.endswith(BYPRODUCT_EXTS):
            ignored.add(n)
        elif top and n in EXCLUDE_DIRS:
            ignored.add(n)
        elif top and n == "main.pdf":
            ignored.add(n)
    return ignored


def _deck_names() -> set[str]:
    return set(build._deck_names_in_dir())


def select_files(fls_inputs: list[str], decks: set[str], tex_names: set[str]) -> list[str]:
    """Filter the INPUT list of main.fls down to the files to package."""
    out = []
    for rel in fls_inputs:
        base = os.path.basename(rel)
        stem, ext = os.path.splitext(rel)
        if rel.split(os.sep, 1)[0] in EXCLUDE_DIRS:
            continue
        if os.path.dirname(rel) == "" and stem == JOB and (ext in BYPRODUCT_EXTS or ext == ".pdf"):
            continue
        if base.endswith(BYPRODUCT_EXTS):
            continue
        if stem in decks or re.sub(r"_frame$", "", stem) in decks:
            continue
        if ext == ".pdf" and stem in tex_names:
            continue  # standalone figure output, not a hand-made image
        if rel.endswith("_paperpreview.tex"):
            continue
        out.append(rel)
    return sorted(set(out))


def _run_quiet(cmd: list[str], cwd: str) -> None:
    import subprocess
    subprocess.run(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                   stderr=subprocess.STDOUT, env=build._latex_env())


def make_arxiv(out_path: str) -> None:
    if not os.path.isfile("main.tex"):
        raise build.BuildError("--arxiv: main.tex not found")
    decks = _deck_names()
    tex_names = {n[:-4] for n in os.listdir(".") if n.endswith(".tex")}
    doc_dir = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="code_to_tex_arxiv_") as tmp:
        work = os.path.join(tmp, "build")
        shutil.copytree(".", work, ignore=_copy_ignore, symlinks=False)
        build.say(f"arxiv: clean build in {work}")
        os.chdir(work)
        try:
            res = build.run_latex(build.Job(JOB, "main.tex", "paper", "main.tex"))
            if res.pages is None:
                raise build.BuildError(f"--arxiv: clean build produced no PDF ({res.failed})")
            issues = res.issues
            if build_log.has_errors(issues):
                raise build.BuildError("--arxiv: clean build reported LaTeX errors; fix them first")
            if build_log.has_undefined_refs(issues):
                build.warn("--arxiv: undefined references remain in the clean build")
            files = select_files(build._fls_inputs(JOB), decks, tex_names)
            if os.path.isfile(JOB + ".bbl") and JOB + ".bbl" not in files:
                files.append(JOB + ".bbl")
            build.say("arxiv: packaging " + str(len(files)) + " files:")
            for f in files:
                print("    " + f)
            tmp_out = out_path + ".tmp"
            try:
                with tarfile.open(tmp_out, "w:gz") as tar:
                    for f in files:
                        tar.add(f, arcname=f, recursive=False)
                    data = (json.dumps(README, indent=2) + "\n").encode()
                    info = tarfile.TarInfo("00README.json")
                    info.size = len(data)
                    info.mode = 0o644
                    info.mtime = int(time.time())
                    tar.addfile(info, io.BytesIO(data))
                os.replace(tmp_out, out_path)
            except OSError as e:
                if os.path.isfile(tmp_out):
                    os.remove(tmp_out)  # our own partial output
                raise build.BuildError(f"--arxiv: cannot write {out_path}: {e}") from e
        finally:
            os.chdir(doc_dir)

        verify = os.path.join(tmp, "verify")
        os.makedirs(verify)
        with tarfile.open(out_path, "r:gz") as tar:
            for m in tar.getmembers():
                if m.name.startswith("/") or ".." in m.name.split("/") or not (m.isfile() or m.isdir()):
                    raise build.BuildError(f"--arxiv: unexpected tar member {m.name}")
            tar.extractall(verify)
        for _ in range(2):
            _run_quiet(["xelatex", "-interaction=nonstopmode", "main.tex"], verify)
        log = ""
        try:
            with open(os.path.join(verify, "main.log"), encoding="utf-8", errors="replace") as f:
                log = f.read()
        except OSError:
            pass
        pages = build_log.output_pages(log)
        vissues = build_log.parse_log(log)
        if pages is None or build_log.has_errors(vissues):
            errs = [i.text for i in vissues if i.kind == "error"][:5]
            raise build.BuildError(f"--arxiv: verification rebuild FAILED (pages={pages}; errors: {errs})")
        if build_log.has_undefined_refs(vissues):
            build.warn("--arxiv: verification rebuild has undefined references")
        build.say(f"arxiv: verification OK ({pages} page{'s' if pages != 1 else ''}); wrote {out_path}")
