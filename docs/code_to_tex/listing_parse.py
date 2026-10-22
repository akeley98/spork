"""Parsing of ``# TeX: `` listing directives (first stage of the listing generator).

The parser turns a source file into a flat list of *items* in file order:

* `ListingDecl`   -- ``listing NAME COUNT``
* `RegionDirective` -- ``begin FILTERS`` / ``end FILTERS``
* `Filbreak`      -- ``filbreak``
* `ContentLine`   -- an ordinary source line, or the payload line of a
  ``remark``/``summary`` directive.  ``color``/``mark`` directives are
  attached to the next content line as *annotations*, with their matches
  already resolved against that line's raw text.

Everything that can be checked without knowing which listing versions include
which lines is checked here; version-dependent checks live in `listing_layout`.
See README.md for the user-facing directive reference.
"""

from __future__ import annotations

import re
import shlex
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

DIRECTIVE_PREFIX = "# TeX: "

#: Directive color name -> box macro (provided by spork_style.tex).
COLOR_MACROS = {
    "red": r"\sporkRed",
    "yellow": r"\sporkYellow",
    "green": r"\sporkGreen",
    "blue": r"\sporkBlue",
    "violet": r"\sporkViolet",
    "invRed": r"\sporkInvRed",
    "invGold": r"\sporkInvGold",
    "invGreen": r"\sporkInvGreen",
    "invBlue": r"\sporkInvBlue",
    "invPurple": r"\sporkInvPurple",
}
#: Pseudo-color: replace the span with an ellipsis.
ELIDE = "elide"
COLOR_NAMES = tuple(COLOR_MACROS) + (ELIDE,)

MARK_POSITIONS = ("before", "after", "start", "end")

#: Names a listing may not have, because they would make directives ambiguous.
RESERVED_LISTING_NAMES = frozenset(COLOR_NAMES) | frozenset(MARK_POSITIONS)

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_MARK_NAME_RE = re.compile(r"[A-Za-z0-9_]+\Z")
_FILTER_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\[(?P<lo>\d*)(?P<colon>:)?(?P<hi>\d*)\])?\Z"
)
_COUNT_RE = re.compile(r"(?P<n>\d+):(?P<i>\d+|\*)\Z")
_DIMEN_RE = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*(?:pt|mm|cm|in|ex|em|bp|pc|sp|dd|cc)\Z"
)
_MISTYPED_RE = re.compile(r"#\s*tex\b|\btex\s*:", re.IGNORECASE)


class ListingError(Exception):
    """User error in a listing source; ``str(e)`` is ``file:line: message``."""

    def __init__(self, filename: str, lineno: int, message: str):
        super().__init__(f"{filename}:{lineno}: {message}")
        self.filename = filename
        self.lineno = lineno
        self.message = message


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Filter:
    """One filter token.  ``listing is None`` means ``*`` (everything)."""

    text: str
    listing: Optional[str]
    lo: int = 0
    hi: int = 0  # exclusive

    @property
    def strict(self) -> bool:
        """Named filters are strict: every version they match must be live."""
        return self.listing is not None

    def matches(self, listing: str, version: int) -> bool:
        if self.listing is None:
            return True
        return listing == self.listing and self.lo <= version < self.hi


@dataclass(frozen=True)
class FilterList:
    filters: tuple[Filter, ...]

    def matches(self, listing: str, version: int) -> bool:
        return any(f.matches(listing, version) for f in self.filters)

    def strictly_matches(self, listing: str, version: int) -> Optional[Filter]:
        """The first strict filter matching (listing, version), if any."""
        for f in self.filters:
            if f.strict and f.matches(listing, version):
                return f
        return None

    def __str__(self) -> str:
        return " ".join(f.text for f in self.filters)


def parse_filter(token: str, listings: dict[str, int]) -> Filter:
    """Parse one filter token; raises ValueError with a user-facing message."""
    if token == "*":
        return Filter(token, None)
    m = _FILTER_RE.match(token)
    if not m:
        raise ValueError(
            f"bad filter {token!r} (expected *, NAME, NAME[i], NAME[lo:hi], NAME[lo:] or NAME[:hi])"
        )
    name = m["name"]
    if name not in listings:
        raise ValueError(
            f"unknown listing {name!r} in filter {token!r} "
            "(declare it with 'listing NAME COUNT' earlier in the file)"
        )
    count = listings[name]
    lo_txt, colon, hi_txt = m["lo"], m["colon"], m["hi"]
    if lo_txt is None:  # bare NAME
        return Filter(token, name, 0, count)
    if not colon:
        if not lo_txt:
            raise ValueError(f"bad filter {token!r}: empty index")
        i = int(lo_txt)
        if i >= count:
            raise ValueError(
                f"filter {token!r}: version {i} out of range (listing {name!r} has {count} versions)"
            )
        return Filter(token, name, i, i + 1)
    lo = int(lo_txt) if lo_txt else 0
    hi = int(hi_txt) if hi_txt else count
    if hi > count or lo >= hi:
        raise ValueError(
            f"filter {token!r}: empty or out-of-range slice "
            f"(listing {name!r} has versions 0..{count - 1})"
        )
    return Filter(token, name, lo, hi)


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


@dataclass
class ListingDecl:
    lineno: int
    name: str
    count: int


@dataclass
class RegionDirective:
    lineno: int
    is_begin: bool
    filters: FilterList


@dataclass
class Filbreak:
    lineno: int


@dataclass
class ColorDirective:
    lineno: int
    filters: FilterList
    color: str
    text: str
    count: int = 1  # number of matches required on the target line
    index: Optional[int] = 0  # which match to color; None = all
    spans: tuple[tuple[int, int], ...] = ()  # resolved, raw-line columns


@dataclass
class MarkDirective:
    lineno: int
    filters: FilterList
    name: str
    text: Optional[str]
    position: str  # one of MARK_POSITIONS
    xshift: Optional[str]
    yshift: Optional[str]
    count: int = 1  # number of matches of text required on the target line
    index: int = 0  # which match to mark
    column: int = 0  # resolved, raw-line column


Annotation = Union[ColorDirective, MarkDirective]


@dataclass
class ContentLine:
    """A line that may appear in listings.

    ``kind`` is ``"source"``, ``"remark"`` or ``"summary"``.  For remarks and
    summaries, ``lineno`` is the payload line and ``directive_lineno`` the
    directive line.  ``filters`` is only set for remarks.
    """

    lineno: int
    text: str
    kind: str = "source"
    bold: bool = False
    filters: Optional[FilterList] = None
    directive_lineno: Optional[int] = None
    annotations: list[Annotation] = field(default_factory=list)


Item = Union[ListingDecl, RegionDirective, Filbreak, ContentLine]


@dataclass
class ParsedSource:
    filename: str
    listings: dict[str, int]  # name -> version count, in declaration order
    decl_lines: dict[str, int]
    items: list[Item]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    value: str
    quoted: bool


def tokenize(body: str) -> list[Token]:
    """Split a directive body into shell-style tokens, remembering quoting.

    Token boundaries are found here; each token's value is then produced by
    `shlex.split`, so quoting/escaping follows POSIX shell rules.
    """
    tokens = []
    i, n = 0, len(body)
    while i < n:
        if body[i].isspace():
            i += 1
            continue
        start, quoted = i, False
        while i < n and not body[i].isspace():
            c = body[i]
            if c in "'\"":
                quoted = True
                j = i + 1
                while j < n and body[j] != c:
                    j += 2 if (c == '"' and body[j] == "\\") else 1
                if j >= n:
                    raise ValueError(f"unterminated {c} quote")
                i = j + 1
            elif c == "\\":
                i += 2
            else:
                i += 1
        raw = body[start:i]
        parts = shlex.split(raw)
        assert len(parts) <= 1, raw
        tokens.append(Token(parts[0] if parts else "", quoted))
    return tokens


def find_matches(line: str, needle: str) -> list[int]:
    """Start columns of non-overlapping matches of needle, left to right."""
    result = []
    pos = line.find(needle)
    while pos >= 0:
        result.append(pos)
        pos = line.find(needle, pos + len(needle))
    return result


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

WarnFn = Callable[[str], None]


def _default_warn(message: str) -> None:
    print(message, file=sys.stderr)


class _Parser:
    def __init__(self, text: str, filename: str, warn: WarnFn):
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        self.lines = [line.rstrip("\r") for line in lines]
        self.filename = filename
        self.warn = warn
        self.listings: dict[str, int] = {}
        self.decl_lines: dict[str, int] = {}
        self.items: list[Item] = []
        self.pending: list[Annotation] = []
        self.next_index = 0  # index into self.lines of the next unread line

    def error(self, lineno: int, message: str) -> ListingError:
        return ListingError(self.filename, lineno, message)

    def parse(self) -> ParsedSource:
        while self.next_index < len(self.lines):
            lineno = self.next_index + 1
            line = self.lines[self.next_index]
            self.next_index += 1
            stripped = line.lstrip()
            if stripped.startswith(DIRECTIVE_PREFIX):
                try:
                    self.directive(lineno, stripped[len(DIRECTIVE_PREFIX):])
                except ValueError as e:
                    raise self.error(lineno, str(e)) from None
            else:
                self.check_mistyped(lineno, line)
                self.add_content(ContentLine(lineno, line.rstrip()))
        if self.pending:
            d = self.pending[0]
            raise self.error(
                d.lineno, f"'{_directive_name(d)}' directive at end of file: no line follows it"
            )
        return ParsedSource(self.filename, self.listings, self.decl_lines, self.items)

    def check_mistyped(self, lineno: int, line: str) -> None:
        if "#" in line and _MISTYPED_RE.search(line):
            self.warn(
                f"{self.filename}:{lineno}: warning: looks like a mistyped directive "
                f"(directives must start with exactly {DIRECTIVE_PREFIX!r}): {line.strip()!r}"
            )

    # -- content lines and annotations -------------------------------------

    def add_content(self, content: ContentLine) -> None:
        for annotation in self.pending:
            if isinstance(annotation, ColorDirective):
                self.resolve_color(annotation, content)
            else:
                self.resolve_mark(annotation, content)
        content.annotations = self.pending
        self.pending = []
        self.items.append(content)

    def select_matches(self, d: Annotation, content: ContentLine) -> list[int]:
        """Check the match count and return the selected match columns."""
        count, index = d.count, d.index
        matches = find_matches(content.text, d.text)
        if len(matches) != count:
            hint = "" if count != 1 or len(matches) < 2 else (
                f"; add N:I (e.g. {len(matches)}:0) to pick one match or N:* to use all"
            )
            raise self.error(
                d.lineno,
                f"{d.text!r} matches {len(matches)} time(s) on line {content.lineno}, "
                f"expected {count}{hint}",
            )
        return matches if index is None else [matches[index]]

    def resolve_color(self, d: ColorDirective, content: ContentLine) -> None:
        starts = self.select_matches(d, content)
        d.spans = tuple((s, s + len(d.text)) for s in starts)

    def resolve_mark(self, d: MarkDirective, content: ContentLine) -> None:
        text = content.text
        if d.text is None:
            if d.position == "start":
                d.column = len(text) - len(text.lstrip())
            else:
                d.column = len(text.rstrip())
            return
        (start,) = self.select_matches(d, content)
        d.column = start if d.position == "before" else start + len(d.text)

    def take_payload(self, lineno: int, what: str) -> ContentLine:
        if self.next_index >= len(self.lines):
            raise self.error(lineno, f"'{what}' directive at end of file: expected a comment line after it")
        payload_lineno = self.next_index + 1
        payload = self.lines[self.next_index].rstrip()
        self.next_index += 1
        stripped = payload.lstrip()
        if stripped.startswith(DIRECTIVE_PREFIX):
            raise self.error(payload_lineno, f"expected the text of the '{what}' on line {lineno}, got a directive")
        if not stripped.startswith("#"):
            raise self.error(
                payload_lineno,
                f"the text of a '{what}' must be a '#' comment line (keeps the source valid Python)",
            )
        return ContentLine(payload_lineno, payload, directive_lineno=lineno)

    # -- directives ----------------------------------------------------------

    def directive(self, lineno: int, body: str) -> None:
        tokens = tokenize(body)
        if not tokens or tokens[0].quoted:
            raise ValueError("missing directive name")
        name, args = tokens[0].value, tokens[1:]
        if name == "listing":
            self.parse_listing(lineno, args)
        elif name in ("begin", "end"):
            self.items.append(RegionDirective(lineno, name == "begin", self.parse_filters(args, name)))
        elif name in ("remark", "remark!"):
            filters = self.parse_filters(args, name)
            content = self.take_payload(lineno, name)
            content.kind, content.bold, content.filters = "remark", name == "remark!", filters
            self.add_content(content)
        elif name in ("summary", "summary!"):
            self.expect_no_args(args, name)
            content = self.take_payload(lineno, name)
            content.kind, content.bold = "summary", name == "summary!"
            self.add_content(content)
        elif name == "filbreak":
            self.expect_no_args(args, name)
            self.items.append(Filbreak(lineno))
        elif name == "color":
            self.pending.append(self.parse_color(lineno, args))
        elif name == "mark":
            self.pending.append(self.parse_mark(lineno, args))
        elif name == "version":
            raise ValueError("unknown directive 'version' (renamed: use 'listing NAME COUNT')")
        else:
            raise ValueError(
                f"unknown directive {name!r} (expected listing, begin, end, remark[!], "
                "summary[!], filbreak, color, mark)"
            )

    @staticmethod
    def expect_no_args(args: list[Token], name: str) -> None:
        if args:
            raise ValueError(f"'{name}' takes no arguments")

    def parse_listing(self, lineno: int, args: list[Token]) -> None:
        if len(args) != 2 or any(t.quoted for t in args):
            raise ValueError("expected 'listing NAME COUNT'")
        name, count_txt = args[0].value, args[1].value
        if not _NAME_RE.match(name):
            raise ValueError(f"bad listing name {name!r} (use letters, digits, underscores)")
        if name in RESERVED_LISTING_NAMES:
            raise ValueError(f"listing name {name!r} is reserved (it is a color or mark position)")
        if name in self.listings:
            raise ValueError(f"listing {name!r} already declared on line {self.decl_lines[name]}")
        if not count_txt.isdigit() or int(count_txt) < 1:
            raise ValueError(f"bad version count {count_txt!r} (expected a positive integer)")
        self.listings[name] = int(count_txt)
        self.decl_lines[name] = lineno
        self.items.append(ListingDecl(lineno, name, int(count_txt)))

    def parse_filters(self, tokens: list[Token], what: str) -> FilterList:
        if not tokens:
            raise ValueError(f"'{what}' needs at least one filter (*, NAME, NAME[i], NAME[lo:hi])")
        filters = []
        for t in tokens:
            if t.quoted:
                raise ValueError(f"unexpected quoted text {t.value!r} among the filters of '{what}'")
            filters.append(parse_filter(t.value, self.listings))
        return FilterList(tuple(filters))

    @staticmethod
    def parse_count(token: Token, allow_all: bool) -> tuple[int, Optional[int]]:
        m = _COUNT_RE.match(token.value) if not token.quoted else None
        if not m:
            raise ValueError(f"expected match count N:I{' or N:*' if allow_all else ''}, got {token.value!r}")
        n = int(m["n"])
        if m["i"] == "*":
            if not allow_all:
                raise ValueError("a mark needs exactly one position: use N:I, not N:*")
            return n, None
        i = int(m["i"])
        if n < 1 or i >= n:
            raise ValueError(f"bad match count {token.value!r}: need N >= 1 and 0 <= I < N")
        return n, i

    def parse_color(self, lineno: int, args: list[Token]) -> ColorDirective:
        if args and args[0].value == "line" and not args[0].quoted:
            raise ValueError("legacy 'color line' syntax is not supported; use: color FILTERS COLOR \"text\" [N:I|N:*]")
        split = next((i for i, t in enumerate(args) if not t.quoted and t.value in COLOR_NAMES), None)
        if split is None:
            raise ValueError(f"'color' needs a color name (one of: {', '.join(COLOR_NAMES)})")
        filters = self.parse_filters(args[:split], "color")
        color = args[split].value
        rest = args[split + 1:]
        if not rest or not rest[0].quoted:
            raise ValueError("'color' needs the text to color, in quotes, after the color name")
        text = rest[0].value
        if not text:
            raise ValueError("empty text to color")
        count, index = 1, 0
        if len(rest) >= 2:
            count, index = self.parse_count(rest[1], allow_all=True)
        if len(rest) > 2:
            raise ValueError(f"unexpected extra argument {rest[2].value!r}")
        return ColorDirective(lineno, filters, color, text, count, index)

    def parse_mark(self, lineno: int, args: list[Token]) -> MarkDirective:
        def is_tail(t: Token) -> bool:
            return t.quoted or t.value in MARK_POSITIONS or t.value.startswith(("xshift=", "yshift="))

        split = next((i for i, t in enumerate(args) if is_tail(t)), len(args))
        head, tail = args[:split], args[split:]
        if len(head) < 2:
            raise ValueError("expected 'mark FILTERS NAME [\"text\" [N:I]] [before|after|start|end] [xshift=..] [yshift=..]'")
        filters = self.parse_filters(head[:-1], "mark")
        name = head[-1].value
        if not _MARK_NAME_RE.match(name):
            raise ValueError(f"bad mark name {name!r} (use letters, digits, underscores)")
        text, count, index = None, 1, 0
        if tail and tail[0].quoted:
            text = tail.pop(0).value
            if not text:
                raise ValueError("empty mark text")
            if tail and _COUNT_RE.match(tail[0].value) and not tail[0].quoted:
                count, index = self.parse_count(tail.pop(0), allow_all=False)
        position = xshift = yshift = None
        for t in tail:
            v = t.value
            if t.quoted:
                raise ValueError(f"unexpected quoted text {v!r} (the mark text must directly follow the mark name)")
            if v in MARK_POSITIONS:
                if position is not None:
                    raise ValueError(f"two mark positions given ({position}, {v})")
                position = v
            elif v.startswith(("xshift=", "yshift=")):
                key, dim = v.split("=", 1)
                if not _DIMEN_RE.match(dim):
                    raise ValueError(f"bad {key} {dim!r} (expected a TeX length like 2pt, -0.5ex, 1mm)")
                if (xshift if key == "xshift" else yshift) is not None:
                    raise ValueError(f"{key} given twice")
                if key == "xshift":
                    xshift = dim
                else:
                    yshift = dim
            elif _COUNT_RE.match(v):
                raise ValueError(f"match count {v!r} must directly follow the quoted mark text")
            else:
                raise ValueError(f"unexpected mark argument {v!r}")
        if text is None:
            position = position or "start"
            if position not in ("start", "end"):
                raise ValueError(f"mark position '{position}' needs quoted text to match")
        else:
            position = position or "before"
            if position not in ("before", "after"):
                raise ValueError(f"mark position '{position}' cannot be combined with text (use before/after)")
        return MarkDirective(lineno, filters, name, text, position, xshift, yshift, count, index)


def _directive_name(d: Annotation) -> str:
    return "color" if isinstance(d, ColorDirective) else "mark"


def parse_source(text: str, filename: str, warn: WarnFn = _default_warn) -> ParsedSource:
    """Parse a listing source file's text.  Raises `ListingError`."""
    return _Parser(text, filename, warn).parse()
