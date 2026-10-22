"""Per-version layout of listings (second stage of the listing generator).

For every (listing, version) this decides which items are shown and how:

* ``begin``/``end`` regions are reference counted per (listing, version);
* remarks need a matching filter and an enclosing region;
* summaries show as ordinary lines inside a region and as italic
  placeholders outside (leading/trailing placeholders are dropped);
* ``color``/``mark`` annotations are checked for liveness, conflicts and
  duplicate names, then turned into per-character colors and mark columns;
* the shared indentation is removed (dedent).

The result is a list of `RenderLine` per version, consumed by `listing_render`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union

from .listing_parse import (
    ELIDE,
    ColorDirective,
    ContentLine,
    Filbreak,
    FilterList,
    ListingDecl,
    ListingError,
    MarkDirective,
    ParsedSource,
    RegionDirective,
)

#: Character classes for `RenderLine.styles`.
CODE, COMMENT, MATH = "code", "comment", "math"


@dataclass(frozen=True)
class RenderLine:
    """One displayed line of one listing version (after dedent).

    ``kind`` is ``source``, ``remark``, ``summary`` (a summary inside a region),
    ``placeholder`` (a summary outside any region) or ``filbreak``.
    ``colors[i]`` / ``styles[i]`` describe character ``text[i]``; ``marks``
    are ``(column, directive)`` pairs sorted by column (stable in file order).
    """

    item: Union[ContentLine, Filbreak]
    kind: str
    bold: bool
    text: str
    colors: tuple[Optional[str], ...]
    styles: tuple[str, ...]
    marks: tuple[tuple[int, MarkDirective], ...]


@dataclass
class ListingLayout:
    name: str
    count: int
    versions: list[list[RenderLine]]  # index = version number


@dataclass
class _Selection:
    entries: list[tuple[Union[ContentLine, Filbreak], bool]]  # (item, is_placeholder)
    in_region: set[int]  # ids of ContentLines inside a region (before remark filtering)


def layout_listings(parsed: ParsedSource) -> dict[str, ListingLayout]:
    """Lay out every version of every listing.  Raises `ListingError`."""
    return _Layouter(parsed).run()


class _Layouter:
    def __init__(self, parsed: ParsedSource):
        self.parsed = parsed
        self.filename = parsed.filename

    def error(self, lineno: int, message: str) -> ListingError:
        return ListingError(self.filename, lineno, message)

    def all_versions(self):
        for name, count in self.parsed.listings.items():
            for v in range(count):
                yield name, v

    def run(self) -> dict[str, ListingLayout]:
        selections = {key: self.select(*key) for key in self.all_versions()}
        self.check_liveness(selections)
        result = {}
        for name, count in self.parsed.listings.items():
            versions = []
            for v in range(count):
                lines = self.render_lines(name, v, selections[name, v])
                if not any(line.kind != "filbreak" for line in lines):
                    raise self.error(
                        self.parsed.decl_lines[name], f"listing version {name}[{v}] is empty (no begin/end region includes any line)"
                    )
                versions.append(lines)
            result[name] = ListingLayout(name, count, versions)
        return result

    # -- selection -----------------------------------------------------------

    def select(self, name: str, version: int) -> _Selection:
        open_begins: list[int] = []
        entries: list[tuple[Union[ContentLine, Filbreak], bool]] = []
        in_region: set[int] = set()
        for item in self.parsed.items:
            if isinstance(item, ListingDecl):
                continue
            if isinstance(item, RegionDirective):
                if not item.filters.matches(name, version):
                    continue
                if item.is_begin:
                    open_begins.append(item.lineno)
                elif not open_begins:
                    raise self.error(item.lineno, f"'end' without a matching 'begin' for {name}[{version}]")
                else:
                    open_begins.pop()
            elif isinstance(item, Filbreak):
                if open_begins:
                    entries.append((item, False))
            elif item.kind == "summary":
                entries.append((item, not open_begins))
            elif open_begins:
                in_region.add(id(item))
                if item.kind == "source" or item.filters.matches(name, version):
                    entries.append((item, False))
        if open_begins:
            raise self.error(open_begins[-1], f"'begin' for {name}[{version}] is never closed by an 'end'")
        # Summary placeholders only make sense between included lines.
        while entries and entries[-1][1]:
            entries.pop()
        first = next((i for i, (_, placeholder) in enumerate(entries) if not placeholder), len(entries))
        return _Selection(entries[first:], in_region)

    # -- liveness ------------------------------------------------------------

    def check_liveness(self, selections: dict[tuple[str, int], _Selection]) -> None:
        shown: dict[int, set[tuple[str, int]]] = defaultdict(set)  # non-placeholder only
        in_region: dict[int, set[tuple[str, int]]] = defaultdict(set)
        for key, sel in selections.items():
            for item, placeholder in sel.entries:
                if not placeholder:
                    shown[id(item)].add(key)
            for item_id in sel.in_region:
                in_region[item_id].add(key)
        for item in self.parsed.items:
            if not isinstance(item, ContentLine):
                continue
            if item.kind == "remark":
                self.check_live(
                    item.filters, in_region[id(item)], item.directive_lineno,
                    f"remark (text on line {item.lineno}) is outside every begin/end region of",
                )
            for a in item.annotations:
                what = "color" if isinstance(a, ColorDirective) else "mark"
                self.check_live(
                    a.filters, shown[id(item)], a.lineno,
                    f"dead '{what}' directive: its target line {item.lineno} is not shown in",
                )

    def check_live(self, filters: FilterList, live: set[tuple[str, int]], lineno: int, dead_message: str) -> None:
        any_live = False
        for key in self.all_versions():
            f = filters.strictly_matches(*key)
            if f is not None and key not in live:
                raise self.error(lineno, f"{dead_message} {key[0]}[{key[1]}] (selected by filter {f.text!r})")
            any_live = any_live or (key in live and filters.matches(*key))
        if not any_live:
            raise self.error(lineno, f"{dead_message} any listing version selected by '{filters}'")

    # -- per-version lines ---------------------------------------------------

    def render_lines(self, name: str, version: int, sel: _Selection) -> list[RenderLine]:
        raw: list[tuple[Union[ContentLine, Filbreak], str, bool, str, list, list]] = []
        for item, placeholder in sel.entries:
            if isinstance(item, Filbreak):
                raw.append((item, "filbreak", False, "", [], []))
                continue
            if "\t" in item.text:
                raise self.error(item.lineno, "tab character in a listing line (expand tabs to spaces)")
            if placeholder:  # annotations never apply to placeholders (see check_liveness)
                text = item.text.replace("#", "# ...", 1)
                raw.append((item, "placeholder", False, text, [None] * len(text), []))
                continue
            colors, marks = self.annotate(item, name, version)
            raw.append((item, item.kind, item.bold, item.text, colors, marks))

        self.check_duplicate_marks(name, version, raw)
        dedent = min(
            (len(text) - len(text.lstrip()) for _, kind, _, text, _, _ in raw if kind != "filbreak" and text.strip()),
            default=0,
        )
        lines = []
        for item, kind, bold, text, colors, marks in raw:
            text, colors = text[dedent:], tuple(colors[dedent:])
            marks = tuple(sorted(((max(0, col - dedent), m) for col, m in marks), key=lambda cm: cm[0]))
            styles = self.char_styles(item, text, colors, marks)
            lines.append(RenderLine(item, kind, bold, text, colors, styles, marks))
        return lines

    def check_duplicate_marks(self, name: str, version: int, raw) -> None:
        seen: dict[str, MarkDirective] = {}
        for *_, marks in raw:
            for _, m in marks:
                if m.name in seen:
                    raise self.error(
                        m.lineno, f"duplicate mark {m.name!r} in {name}[{version}] (also on line {seen[m.name].lineno})"
                    )
                seen[m.name] = m

    def annotate(self, item: ContentLine, name: str, version: int) -> tuple[list[Optional[str]], list[tuple[int, MarkDirective]]]:
        """Per-character colors (raw columns) and marks applying to this version."""
        colors: list[Optional[str]] = [None] * len(item.text)
        owners: list[Optional[ColorDirective]] = [None] * len(item.text)
        marks: list[tuple[int, MarkDirective]] = []
        for a in item.annotations:
            if not a.filters.matches(name, version):
                continue
            if isinstance(a, MarkDirective):
                marks.append((a.column, a))
                continue
            for start, end in a.spans:
                for i in range(start, end):
                    if colors[i] is None:
                        colors[i], owners[i] = a.color, a
                    elif colors[i] != a.color:
                        raise self.error(
                            a.lineno,
                            f"conflicting colors in {name}[{version}]: character {i + 1} "
                            f"({item.text[i]!r}) of line {item.lineno} is {a.color} here ({a.text!r}) "
                            f"but {colors[i]} ({owners[i].text!r}) by the directive on line {owners[i].lineno}",
                        )
        for col, m in marks:
            if 0 < col < len(colors) and colors[col - 1] == ELIDE and colors[col] == ELIDE:
                raise self.error(m.lineno, f"mark {m.name!r} is inside elided text in {name}[{version}]")
        return colors, marks

    def char_styles(self, item, text: str, colors: tuple, marks: tuple) -> tuple[str, ...]:
        """Classify characters as code / comment / math; check math boundaries.

        Comments start at the first ``#`` (naive, as in the legacy script);
        inside comments ``$...$`` is LaTeX math, passed through unescaped.
        """
        styles = []
        math_regions = []
        in_comment = False
        math_start = None
        for i, c in enumerate(text):
            in_comment = in_comment or c == "#"
            if in_comment and c == "$":
                if math_start is None:
                    math_start = i
                else:
                    math_regions.append((math_start, i + 1))
                    math_start = None
                styles.append(MATH)
            elif math_start is not None:
                styles.append(MATH)
            else:
                styles.append(COMMENT if in_comment else CODE)
        if math_start is not None:
            raise self.error(
                item.lineno,
                "unbalanced '$' in comment (text between $...$ in comments is LaTeX math, "
                "so comments cannot contain a literal '$')",
            )
        for lo, hi in math_regions:
            for p in range(lo + 1, hi):
                if colors[p - 1] != colors[p]:
                    raise self.error(item.lineno, "a color span starts or ends inside $...$ math")
            for col, m in marks:
                if lo < col < hi:
                    raise self.error(m.lineno, f"mark {m.name!r} is inside $...$ math")
        return tuple(styles)
