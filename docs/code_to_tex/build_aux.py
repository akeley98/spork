"""Reading LaTeX/Beamer auxiliary files for code_to_tex.build.

* ``aux_state``: a fingerprint of the auxiliary files of a job; if it changes
  across a pass, labels/tikzmark positions/navigation moved and another pass is needed.
* ``where``: find the page(s) and Beamer frame(s) containing a label (``--where``).

Beamer .aux format (verified by compiling a test deck with TeX Live on this machine):
a frame with ``[label=intro]`` and 2 overlay slides writes, in order,
``\\newlabel{intro<1>}{{1}{1}...}``, ``\\newlabel{intro}{{1}{1}...}``,
any ``\\label``s in the frame (Beamer writes a ``\\label`` on the frame's *first* slide only,
unless the label has its own overlay spec), ``\\newlabel{intro<2>}{{1}{2}...}``, and finally
``\\@writefile{nav}{\\headcommand {\\beamer@framepages {1}{2}}}`` closing the frame.
The first group of ``\\newlabel``'s value is the frame number, the second the page.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re

AUX_STATE_EXTS = (".aux", ".toc", ".nav", ".snm", ".out", ".lof", ".lot", ".bbl")


def aux_state(job: str) -> str | None:
    """Hash of all auxiliary files of ``job`` in the cwd (None if none exist)."""
    h = hashlib.sha256()
    found = False
    for ext in AUX_STATE_EXTS:
        path = job + ext
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            continue
        found = True
        h.update(ext.encode() + b"\0" + str(len(data)).encode() + b"\0" + data)
    return h.hexdigest() if found else None


def _read_group(text: str, i: int) -> tuple[str, int] | None:
    """Read a balanced {...} group starting at text[i] == '{'; return (content, index after)."""
    if i >= len(text) or text[i] != "{":
        return None
    depth = 0
    j = i
    while j < len(text):
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1: j], j + 1
        j += 1
    return None


def _groups(text: str, max_groups: int) -> list[str]:
    out = []
    i = 0
    while len(out) < max_groups:
        while i < len(text) and text[i] in " \t":
            i += 1
        g = _read_group(text, i)
        if g is None:
            break
        out.append(g[0])
        i = g[1]
    return out


@dataclasses.dataclass
class Label:
    name: str
    number: str  # value of the first group (frame number in Beamer, section/figure number in a paper)
    page: str
    frame_index: int | None = None  # index into AuxInfo.frames (decks only)


@dataclasses.dataclass
class Frame:
    number: int  # 1-based frame ordinal in the document
    first_page: int
    last_page: int
    labels: list[str]  # frame labels ([label=...]), possibly empty


@dataclasses.dataclass
class AuxInfo:
    path: str
    labels: list[Label]
    frames: list[Frame]

    @property
    def is_deck(self) -> bool:
        return bool(self.frames)


_NEWLABEL_RE = re.compile(r"\\newlabel\{")
_FRAMEPAGES_RE = re.compile(r"\\beamer@framepages\s*\{(\d+)\}\{(\d+)\}")
_INPUT_RE = re.compile(r"^\\@input\{([^}]*)\}")
_FRAME_SLIDE_LABEL_RE = re.compile(r"^(.*)<(\d+)>$")


def read_aux(path: str) -> AuxInfo:
    """Parse labels and Beamer frame page ranges from an .aux file (following \\@input)."""
    info = AuxInfo(path, [], [])
    pending: list[Label] = []
    pending_frame_labels: list[str] = []

    def walk(p: str, depth: int) -> None:
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
        except OSError:
            return
        for line in lines:
            m = _INPUT_RE.match(line)
            if m and depth < 10:
                walk(os.path.join(os.path.dirname(p), m.group(1)), depth + 1)
                continue
            if line.startswith("\\newlabel{"):
                g = _groups(line[len("\\newlabel"):], 2)
                if len(g) < 2:
                    continue
                name, value = g
                vals = _groups(value, 2)
                number = vals[0] if vals else ""
                page = vals[1] if len(vals) > 1 else ""
                fm = _FRAME_SLIDE_LABEL_RE.match(name)
                if fm and fm.group(2) == "1" and fm.group(1) not in pending_frame_labels:
                    pending_frame_labels.append(fm.group(1))
                lab = Label(name, number, page)
                info.labels.append(lab)
                pending.append(lab)
                continue
            m = _FRAMEPAGES_RE.search(line)
            if m:
                idx = len(info.frames)
                info.frames.append(
                    Frame(idx + 1, int(m.group(1)), int(m.group(2)), list(pending_frame_labels))
                )
                for lab in pending:
                    lab.frame_index = idx
                pending.clear()
                pending_frame_labels.clear()

    walk(path, 0)
    return info


@dataclasses.dataclass
class WhereHit:
    aux: str
    label: Label
    frame: Frame | None


def where(label: str, aux_paths: list[str]) -> tuple[list[WhereHit], bool]:
    """Find ``label`` in the given aux files.

    Returns (hits, exact). If there are no exact matches anywhere, substring
    matches are returned instead and ``exact`` is False.
    Overlay-slide pseudo-labels (``frame<2>``) are only matched exactly.
    """
    infos = [read_aux(p) for p in aux_paths if os.path.isfile(p)]

    def collect(pred) -> list[WhereHit]:
        hits = []
        for info in infos:
            for lab in info.labels:
                if pred(lab.name):
                    frame = info.frames[lab.frame_index] if lab.frame_index is not None else None
                    hits.append(WhereHit(info.path, lab, frame))
        return hits

    hits = collect(lambda n: n == label)
    if hits:
        return hits, True
    hits = collect(lambda n: label in n and not _FRAME_SLIDE_LABEL_RE.match(n))
    return hits, False


def format_where(label: str, hits: list[WhereHit], exact: bool, doc_dir_arg: str) -> list[str]:
    out = []
    if not hits:
        return [f"--where {label}: not found in any .aux file (build first?)"]
    if not exact:
        out.append(f"--where {label}: no exact match; labels containing it:")
    else:
        out.append(f"--where {label}:")
    for hit in hits[:30]:
        aux = os.path.basename(hit.aux)
        if hit.frame is not None:
            f = hit.frame
            pages = f"{f.first_page}" if f.first_page == f.last_page else f"{f.first_page}-{f.last_page}"
            if f.labels:
                flabel = f"[label={','.join(f.labels)}]"
            else:
                flabel = "(frame has no label; add [label=...] to use --frame)"
            out.append(
                f"  {aux}: `{hit.label.name}' page {hit.label.page}; frame #{f.number} {flabel}, pages {pages}"
            )
            if f.labels:
                deck = os.path.splitext(aux)[0]
                out.append(
                    f"      -> python3 -m code_to_tex.build {doc_dir_arg} --slides {deck} "
                    f"--frame {f.labels[0]} --skip-code --png"
                )
        else:
            num = f" (number {hit.label.number})" if hit.label.number else ""
            out.append(f"  {aux}: `{hit.label.name}' page {hit.label.page}{num}")
    if len(hits) > 30:
        out.append(f"  ... and {len(hits) - 30} more")
    return out


def frame_labels(aux_path: str) -> set[str]:
    """All frame labels ([label=...]) defined in a deck's aux file."""
    info = read_aux(aux_path)
    out: set[str] = set()
    for f in info.frames:
        out.update(f.labels)
    return out
