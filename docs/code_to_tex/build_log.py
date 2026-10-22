"""Parsing of xelatex .log files and linting of overlay specs for code_to_tex.build.

Pure text processing (no subprocesses), so everything here is unit-testable
with canned log snippets.

The build driver runs xelatex with ``max_print_line`` etc. set very large
(see ``build.LATEX_ENV``), so messages are not wrapped at 79 columns;
the parser still tolerates the ``(pkg)`` continuation lines of package warnings.
"""

from __future__ import annotations

import dataclasses
import os
import re

# ---------------------------------------------------------------------------
# Rerun detection


_RERUN_RE = re.compile(
    r"Rerun to get"
    r"|Label\(s\) may have changed"
    r"|Please rerun LaTeX"
    r"|Please \(re\)run"
    r"|\(rerunfilecheck\).*has changed"
    r"|Rerun LaTeX"
    r"|Temporary extra page added at the end\. Rerun"
)


def log_requests_rerun(log_text: str) -> bool:
    """True if the log says (in any of the usual phrasings) to run LaTeX again."""
    return _RERUN_RE.search(log_text) is not None


_OUTPUT_RE = re.compile(r"^Output written on (.+?) \((\d+) pages?", re.MULTILINE)


def output_pages(log_text: str) -> int | None:
    """Number of pages written according to the log, or None if no output file was written.

    A stale PDF from an earlier build may exist on disk, so the log (not the file) is
    the authority on whether *this* pass produced output.
    """
    m = _OUTPUT_RE.search(log_text)
    if m is None:
        return None
    return int(m.group(2))


# ---------------------------------------------------------------------------
# Warnings


@dataclasses.dataclass(frozen=True)
class Issue:
    """One warning/error found in a log.

    kind: one of KINDS (below).
    key: what to deduplicate on (e.g. label name, or (file, line)).
    text: human-readable one-line description (without location).
    where: "file:line" if known, else "".
    """

    kind: str
    text: str
    where: str = ""


# Display order and headings.
KINDS = [
    ("error", "LaTeX errors"),
    ("undefined", "Undefined references/citations"),
    ("multiply", "Multiply-defined labels"),
    ("overfull", "Overfull boxes"),
    ("underfull", "Underfull boxes"),
    ("missing_char", "Missing characters in font"),
    ("spork", "spork style warnings"),
    ("other", "Other LaTeX/package warnings"),
]
KIND_TITLES = dict(KINDS)

# How many distinct items to show per kind in the summary.
SHOW_LIMIT = {
    "error": 10,
    "undefined": 10,
    "multiply": 10,
    "overfull": 6,
    "underfull": 3,
    "missing_char": 6,
    "spork": 10,
    "other": 4,
}

_UNDEF_RE = re.compile(
    r"(?:LaTeX|Package natbib) Warning: (Reference|Citation|Hyper reference) `([^']*)' "
    r"on page (\S+) undefined(?: on input line (\d+))?"
)
_THERE_WERE_UNDEF_RE = re.compile(
    r"LaTeX Warning: There were undefined (references|citations)"
)
_MULTIPLY_RE = re.compile(r"LaTeX Warning: Label `([^']*)' multiply defined")
_BOX_RE = re.compile(
    r"^(Overfull|Underfull) \\([hv]box) \(([^)]*)\) "
    r"(?:in paragraph at lines (\d+)--(\d+)|detected at line (\d+)"
    r"|has occurred while \\output is active|in alignment at lines (\d+)--(\d+))"
)
_MISSING_CHAR_RE = re.compile(r"^Missing character: There is no (.+?) in font (.*?)!?$")
_PKG_WARNING_RE = re.compile(r"^(?:Package|Class) (\S+) Warning: (.*)$")
_LATEX_WARNING_RE = re.compile(r"^LaTeX(?: Font)? Warning: (.*)$")
_INPUT_LINE_RE = re.compile(r" on input line (\d+)\.?$")
_ERROR_LINE_RE = re.compile(r"^l\.(\d+)")

# Package warnings not worth surfacing (pure noise for this toolchain).
_NOISE_RE = re.compile(
    r"LaTeX Font Warning"
    r"|Font shape .* undefined"
    r"|Some font shapes were not available"
    r"|There were undefined references"
    r"|There were multiply-defined labels"
    r"|Label\(s\) may have changed"
    r"|Rerun to get"
    r"|Package rerunfilecheck"
    r"|Unused global option"
)


def _display_path(path: str | None, doc_dir: str | None) -> str:
    if not path:
        return "?"
    if path.startswith("./"):
        path = path[2:]
    if doc_dir and os.path.isabs(path):
        try:
            rel = os.path.relpath(path, doc_dir)
        except ValueError:
            rel = path
        if not rel.startswith(".."):
            return rel
    return path


_FILE_START_RE = re.compile(r"\(([^\s()]+)")
_LIKELY_FILE_EXT = re.compile(r"\.(tex|sty|cls|def|cfg|fd|aux|clo|ldf|code\.tex|out|nav|toc|snm|vrb|ltx|dict|lua|map|enc|bbl|mkii|mkiv|pgf|tikz|txt)$")


class _FileTracker:
    """Heuristic tracking of which input file TeX is reading, from the parentheses in the log.

    TeX writes ``(path`` when it opens a file and ``)`` when it closes it.
    Other parenthesized text in the log is (usually) balanced, so each "(" is pushed
    (as None if it does not look like a file) and each ")" pops.
    """

    def __init__(self, doc_dir: str | None):
        self.stack: list[str | None] = []
        self.doc_dir = doc_dir

    def current(self) -> str | None:
        for item in reversed(self.stack):
            if item is not None:
                return item
        return None

    def _looks_like_file(self, candidate: str) -> bool:
        if candidate.startswith("./") or candidate.startswith("/"):
            return True
        if _LIKELY_FILE_EXT.search(candidate):
            return True
        if self.doc_dir and os.path.isfile(os.path.join(self.doc_dir, candidate)):
            return True
        return False

    def feed(self, line: str) -> None:
        i = 0
        n = len(line)
        while i < n:
            c = line[i]
            if c == "(":
                m = _FILE_START_RE.match(line, i)
                if m and self._looks_like_file(m.group(1)):
                    self.stack.append(m.group(1))
                    i = m.end()
                    continue
                self.stack.append(None)
            elif c == ")":
                if self.stack:
                    self.stack.pop()
            i += 1


def _join_continuations(lines: list[str]) -> list[str]:
    """Join ``(pkg)   more text`` continuation lines onto their package warning."""
    out: list[str] = []
    pkg = None
    for line in lines:
        m = _PKG_WARNING_RE.match(line)
        if m:
            pkg = m.group(1)
            out.append(line)
            continue
        if pkg is not None and out and line.startswith("(" + pkg + ")"):
            out[-1] = out[-1] + " " + line[len(pkg) + 2:].strip()
            continue
        pkg = None
        out.append(line)
    return out


def parse_log(log_text: str, doc_dir: str | None = None) -> list[Issue]:
    """Extract all interesting issues from a log, in log order (duplicates kept)."""
    issues: list[Issue] = []
    tracker = _FileTracker(doc_dir)
    lines = _join_continuations(log_text.splitlines())
    skip_parens_until_blank = False
    pending_error: tuple[str, str | None] | None = None  # (message, file)

    def loc(file: str | None, line: str | int | None) -> str:
        if line is None:
            return _display_path(file, doc_dir) if file else ""
        return f"{_display_path(file, doc_dir)}:{line}"

    for line in lines:
        if pending_error is not None:
            m = _ERROR_LINE_RE.match(line)
            if m:
                issues.append(Issue("error", pending_error[0], loc(pending_error[1], m.group(1))))
                pending_error = None
            elif line == "":
                issues.append(Issue("error", pending_error[0], loc(pending_error[1], None)))
                pending_error = None
        if skip_parens_until_blank:
            if line == "":
                skip_parens_until_blank = False
                continue
            if not line.startswith("! "):
                continue
            skip_parens_until_blank = False

        cur = tracker.current()

        if line.startswith("! "):
            if pending_error is not None:
                issues.append(Issue("error", pending_error[0], loc(pending_error[1], None)))
            pending_error = (line[2:].strip(), cur)
            # The error context (source excerpt, help text) may contain unbalanced parens.
            skip_parens_until_blank = True
            continue

        m = _BOX_RE.match(line)
        if m:
            kind = "overfull" if m.group(1) == "Overfull" else "underfull"
            lineno = m.group(4) or m.group(6) or m.group(7)
            text = f"{m.group(1)} \\{m.group(2)} ({m.group(3)})"
            if m.group(4) and m.group(5) and m.group(5) != m.group(4):
                where = loc(cur, f"{m.group(4)}-{m.group(5)}")
            else:
                where = loc(cur, lineno) if lineno else loc(cur, None)
            issues.append(Issue(kind, text, where))
            # Box contents follow until an empty line; they contain arbitrary text.
            skip_parens_until_blank = True
            continue

        m = _UNDEF_RE.search(line)
        if m:
            what = m.group(1).lower()
            issues.append(Issue("undefined", f"{what} `{m.group(2)}' undefined", loc(cur, m.group(4))))
            continue
        m = _THERE_WERE_UNDEF_RE.search(line)
        if m:
            issues.append(Issue("undefined", f"there were undefined {m.group(1)}", ""))
            continue
        m = _MULTIPLY_RE.search(line)
        if m:
            issues.append(Issue("multiply", f"label `{m.group(1)}' multiply defined", ""))
            continue
        m = _MISSING_CHAR_RE.match(line)
        if m:
            issues.append(Issue("missing_char", f"no {m.group(1)} in font {m.group(2)}", ""))
            continue
        m = _PKG_WARNING_RE.match(line)
        if m:
            if not _NOISE_RE.search(line):
                kind = "spork" if m.group(1).lower().startswith("spork") else "other"
                msg = m.group(2)
                lm = _INPUT_LINE_RE.search(msg)
                where = ""
                if lm:
                    msg = msg[: lm.start()]
                    where = loc(cur, lm.group(1))
                issues.append(Issue(kind, f"{m.group(1)}: {msg.strip()}", where))
            tracker.feed(line)
            continue
        m = _LATEX_WARNING_RE.match(line)
        if m:
            if not _NOISE_RE.search(line):
                msg = m.group(1)
                lm = _INPUT_LINE_RE.search(msg)
                where = ""
                if lm:
                    msg = msg[: lm.start()]
                    where = loc(cur, lm.group(1))
                issues.append(Issue("other", msg.strip(), where))
            tracker.feed(line)
            continue

        tracker.feed(line)

    if pending_error is not None:
        issues.append(Issue("error", pending_error[0], loc(pending_error[1], None)))
    return issues


def has_undefined_refs(issues: list[Issue]) -> bool:
    return any(i.kind == "undefined" for i in issues)


def has_errors(issues: list[Issue]) -> bool:
    return any(i.kind == "error" for i in issues)


def group_issues(issues: list[Issue]) -> dict[str, list[tuple[Issue, int]]]:
    """Deduplicate issues (Beamer repeats a box warning once per overlay slide).

    Returns {kind: [(issue, count), ...]} in first-seen order, kinds in KINDS order.
    The generic "there were undefined references" line is dropped when specific
    undefined-reference warnings are present.
    """
    counts: dict[Issue, int] = {}
    for issue in issues:
        counts[issue] = counts.get(issue, 0) + 1
    grouped: dict[str, list[tuple[Issue, int]]] = {}
    for kind, _title in KINDS:
        items = [(i, c) for i, c in counts.items() if i.kind == kind]
        if kind == "undefined" and any(not i.text.startswith("there were") for i, _ in items):
            items = [(i, c) for i, c in items if not i.text.startswith("there were")]
        if items:
            grouped[kind] = items
    return grouped


# ---------------------------------------------------------------------------
# Overlay-spec lint


_LINT_RE = re.compile(
    r"\\(only|uncover|visible|invisible|onslide|alt|temporal|node|draw|path|fill|filldraw)\s*<"
    r"|\\pause\b"
)


def _strip_comment(line: str) -> str:
    i = 0
    while True:
        j = line.find("%", i)
        if j < 0:
            return line
        # count preceding backslashes
        k = j - 1
        bs = 0
        while k >= 0 and line[k] == "\\":
            bs += 1
            k -= 1
        if bs % 2 == 0:
            return line[:j]
        i = j + 1


def lint_overlay_text(text: str, filename: str) -> list[Issue]:
    """Warn about overlay specs not routed through the spork intent wrappers.

    Every overlay must state its paper intent (\\sporkReveal, \\sporkSlideOnly,
    \\sporkPaperOnly, TikZ keys `spork reveal=...` etc.); bare \\only<...>,
    \\node<...>, \\pause etc. are ambiguous in the paper (beamerarticle shows them all).
    """
    issues = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw)
        for m in _LINT_RE.finditer(line):
            what = m.group(0).rstrip("<").rstrip()
            if what == "\\pause":
                msg = "bare \\pause (use \\sporkReveal<n->{...} so the paper intent is explicit)"
            else:
                msg = f"bare overlay spec {what}<...> (use \\sporkReveal / \\sporkSlideOnly / `spork reveal=` keys)"
            issues.append(Issue("lint", msg, f"{filename}:{lineno}"))
    return issues


# ---------------------------------------------------------------------------
# Figure lint: host-relative sizes in figure files
#
# A figure file (\documentclass[...]{standalone}) is \input by both the paper and the decks, whose
# body sizes and line widths differ. Anything measured relative to the host then differs between the
# two renderings, while the slides' \scalebox assumes identical geometry.

_FIGURE_RE = re.compile(r"\\documentclass\s*(\[[^\]]*\])?\s*\{standalone\}")
_FIG_SIZE_RE = re.compile(
    r"\\(tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|Huge)(?![A-Za-z])"
)
_FIG_WIDTH_RE = re.compile(r"\\(linewidth|textwidth|columnwidth|hsize|paperwidth|textheight)(?![A-Za-z])")
_FIG_LISTING_RE = re.compile(r"\\sporkListing(?![A-Za-z])")
_FIG_ABS_FONT_RE = re.compile(r"\\(fontsize|sporkFigFont|sporkFigSmall|sporkFigLarge)(?![A-Za-z])")


def is_figure_file(text: str) -> bool:
    return _FIGURE_RE.search(text) is not None


def lint_figure_text(text: str, filename: str) -> list[Issue]:
    """In figure files, warn about sizes relative to the host document (they differ between paper and slides)."""
    if not is_figure_file(text):
        return []
    issues = []
    absolute_font_seen = False
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw)
        where = f"{filename}:{lineno}"
        for m in _FIG_SIZE_RE.finditer(line):
            issues.append(Issue("lint", f"figure uses \\{m.group(1)} (host-relative size; use \\sporkFigSmall / \\sporkFigLarge / \\fontsize)", where))
        for m in _FIG_WIDTH_RE.finditer(line):
            issues.append(Issue("lint", f"figure uses \\{m.group(1)} (differs between paper and slides; use mm)", where))
        for m in _FIG_ABS_FONT_RE.finditer(line):
            # A node's `font=\sporkFigLarge` doesn't set the size of a later listing.
            if not re.search(r"font\s*=\s*[^,\]]*$", line[:m.start()]):
                absolute_font_seen = True
        if _FIG_LISTING_RE.search(line) and not absolute_font_seen:
            issues.append(Issue("lint", "figure has \\sporkListing without an absolute font before it (it would follow the host body size; set \\sporkFigFont or \\fontsize first)", where))
    return issues
