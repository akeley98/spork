"""Listing generator: annotated source file -> per-listing .tex/.txt/.dbg files.

Public API (also used in-process by ``code_to_tex.build``):

* `generate(src_path, out_dir)` -- write all outputs, return the paths written;
* `render_outputs(text, filename)` -- the same, in memory (``{basename: content}``);
* `ListingError` -- user error, ``str(e)`` is ``file:line: message``.

Pipeline: `listing_parse` (directives -> items) -> `listing_layout`
(per-version selection, validation, dedent) -> `listing_render` (output text).
Directive reference and LaTeX usage: README.md in this package.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from .listing_layout import layout_listings
from .listing_parse import ListingError, parse_source
from .listing_render import dbg_file, tex_file, txt_file

__all__ = ["ListingError", "generate", "render_outputs", "main"]


def render_outputs(text: str, filename: str, warn: Optional[Callable[[str], None]] = None) -> dict[str, str]:
    """Render every listing in ``text``; returns ``{output basename: content}``.

    ``filename`` is used in error messages and the generated-file header.
    ``warn`` receives warning lines (default: print to stderr).
    Raises `ListingError` before producing anything if the source has an error.
    """
    parsed = parse_source(text, filename, **({"warn": warn} if warn else {}))
    outputs: dict[str, str] = {}
    for name, layout in layout_listings(parsed).items():
        outputs[f"{name}.tex"] = tex_file(layout, os.path.basename(filename))
        for v, lines in enumerate(layout.versions):
            outputs[f"{name}.{v}.txt"] = txt_file(lines)
            outputs[f"{name}.{v}.dbg"] = dbg_file(lines)
    return outputs


def generate(src_path: str, out_dir: str, warn: Optional[Callable[[str], None]] = None) -> list[str]:
    """Generate all listing outputs of ``src_path`` into ``out_dir``.

    Returns the paths of all outputs (written or already up to date).  Files
    whose content is unchanged are not rewritten (mtimes are kept); other
    files in ``out_dir`` are never touched.  Raises `ListingError`; on error
    nothing is written.
    """
    with open(src_path, encoding="utf-8") as f:
        text = f.read()
    outputs = render_outputs(text, src_path, warn)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for basename, content in outputs.items():
        path = os.path.join(out_dir, basename)
        _write_if_changed(path, content)
        paths.append(path)
    return paths


def _write_if_changed(path: str, content: str) -> None:
    try:
        with open(path, encoding="utf-8") as f:
            if f.read() == content:
                return
    except (FileNotFoundError, UnicodeDecodeError):
        pass
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python3 -m code_to_tex",
        description="Generate LaTeX code listings (NAME.tex, NAME.N.txt, NAME.N.dbg) from '# TeX: ' directives.",
    )
    parser.add_argument("src", help="annotated source file, e.g. samples.py")
    parser.add_argument("out_dir", help="output directory, e.g. code/")
    args = parser.parse_args(argv)
    try:
        paths = generate(args.src, args.out_dir)
    except ListingError as e:
        print(e, file=sys.stderr)
        return 1
    except OSError as e:
        print(f"code_to_tex: {e}", file=sys.stderr)
        return 1
    print(f"code_to_tex: {len(paths)} files up to date in {args.out_dir}")
    return 0
