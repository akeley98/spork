import sys
import os
import math
from typing import Optional, List
from dataclasses import dataclass

color_dict = {
    "r": "redBox",
    "g": "greenBox",
    "b": "blueBox",
    "y": "yellowBox",
    "v": "violetBox",
    "#": "codecomment",  # Internal use
    ".": "codecomment",  # Special meaning, delete chars
}

char_dict = {
    " ": r"~",
    "#": "\\#",
    "$": "\\$",
    "%": "\\%",
    "&": "\\&",
    "_": "\\_",
    "{": "\\{",
    "}": "\\}",
    "~": r"{\textasciitilde}",
    "^": r"{\textasciicircum}",
    "\\": r"{\textbackslash}",
}


@dataclass(slots=True)
class TexLine:
    lineno: int
    is_summary: bool
    is_remark: bool
    raw_text: str
    color_letters: str
    bold: bool = False
    raw_text_is_latex: bool = False

    def leading_spaces(self):
        if self.raw_text_is_latex:
            return float("inf")
        for i, c in enumerate(self.raw_text):
            if not c.isspace():
                return i
        return float("inf")

    def gen_tex(self, dedent):
        if self.raw_text_is_latex:
            return self.raw_text

        py_text = self.raw_text[dedent:]
        if self.is_summary:
            py_text = py_text.replace("#", "# ...", 1)
            colors = ""
        else:
            colors = self.color_letters[dedent:]
        prev_color_char = None
        snippets = [r"\blacktt{"]
        in_comment = False
        in_math = False

        for i, c in enumerate(py_text):
            # Comment detection (not very smart e.g. fooled by strings)
            if c == "#":
                in_comment = True

            # Colorize based on matching charater in color chars
            color_char = "#" if in_comment else None
            if i < len(colors):
                tmp = colors[i]
                if not tmp.isspace() and tmp != '#':
                    color_char = tmp

            # End colorBox if needed.
            if color_char != prev_color_char and prev_color_char is not None:
                snippets.append("}")

            # Begin colorBox if needed.
            if color_char is not None and color_char != prev_color_char:
                colorBox = color_dict.get(color_char)
                if colorBox is None:
                    raise ValueError(f"Line {self.lineno}: unknown color character {color_char!r} in {colors!r}")
                snippets.append(fr"\{colorBox}{{")
                if color_char == ".":
                    snippets.append("...")

            # Sanitize characters, except for stuff surrounded by $ in comments.
            # Remove characters to be replaced with ...
            if color_char == ".":
                pass
            elif in_comment:
                if c == "$":
                    in_math = not in_math
                elif not in_math:
                    c = char_dict.get(c, c)
                snippets.append(c)
            else:
                c = char_dict.get(c, c)
                snippets.append(c)

            prev_color_char = color_char

        if prev_color_char:
            snippets.append("}")
        snippets.append(r"}")

        if self.is_summary:
            snippets = [r"\textit{"] + snippets + ["}"]
        if self.bold:
            snippets = [r"\textbf{"] + snippets + ["}"]

        return "".join(snippets)


@dataclass(slots=True)
class AnyFilter:
    def accepts(self, name, version_number):
        return True

@dataclass(slots=True)
class VersionFilter:
    name: str
    lo_version: Optional[int]
    hi_version: Optional[int]

    def accepts(self, name, version_number):
        if name != self.name:
            return False
        if self.lo_version is not None and version_number < self.lo_version:
            return False
        if self.hi_version is not None and version_number >= self.hi_version:
            return False
        return True

class BaseFilterImpl:
    __slots__ = []
    def passes_filter(self, name, version_number):
        return any(f.accepts(name, version_number) for f in self.filters)

@dataclass(slots=True)
class VersionDirective:
    name: str
    version_count: int

@dataclass(slots=True)
class BeginDirective(BaseFilterImpl):
    filters: List[VersionFilter]
    lineno: int

@dataclass(slots=True)
class EndDirective(BaseFilterImpl):
    filters: List[VersionFilter]
    lineno: int

@dataclass(slots=True)
class SummaryDirective:
    text: str
    color_letters: str
    bold: bool
    lineno: int

@dataclass(slots=True)
class RemarkDirective(BaseFilterImpl):
    filters: List[VersionFilter]
    text: str
    color_letters: str
    bold: bool
    lineno: int

@dataclass(slots=True)
class ColorDirective(BaseFilterImpl):
    filters: List[VersionFilter]
    color_letters: str

@dataclass(slots=True)
class FilbreakDirective:
    lineno: int

@dataclass(slots=True)
class SourceLine:
    text: str
    lineno: int

def parse_filter(s):
    try:
        if s == "*":
            return AnyFilter()
        frags = s.split("[")
        if len(frags) == 1:
            return VersionFilter(frags[0], None, None)
        elif len(frags) == 2:
            name, trail = frags
            if trail and trail[-1] == "]":
                lo = None
                hi = None
                lo_hi_txt = trail[:-1].split(":")
                if len(lo_hi_txt) == 1:
                    lo = int(lo_hi_txt[0])
                    return VersionFilter(name, lo, lo+1)
                elif len(lo_hi_txt) == 2:
                    lo_txt, hi_txt = lo_hi_txt
                    if lo_txt and not lo_txt.isspace():
                        lo = int(lo_txt)
                    if hi_txt and not hi_txt.isspace():
                        hi = int(hi_txt)
                    return VersionFilter(name, lo, hi)
        raise ValueError("Expect 'name[{lo}:{hi}]'")
    except Exception as e:
        raise ValueError(f"Could not parse filter {s}: {e}") from e

def parse_filters(version_names, frags):
    if not frags:
        raise ValueError("Missing filter arguments")
    result = []
    for s in frags:
        filter = parse_filter(s)
        if not isinstance(filter, AnyFilter) and filter.name not in version_names:
            raise ValueError(f"Unknown version name {filter.name!r} in {s!r}")
        result.append(filter)
    return result

def file_to_lines_directives(f):
    """Read lines from file object and parse into list of directives + SourceLine"""
    result = []
    lineno = 0
    lines = list(f)
    version_names = set()

    while lineno < len(lines):
        text = lines[lineno]
        if text.endswith("\n"):
            text = text[:-1]
        lineno = lineno + 1
        directive = None
        try:
            def eat_next_line():
                nonlocal lineno
                if lineno >= len(lines):
                    raise ValueError(f"Unexpected end-of-input")
                lineno = lineno + 1
                return lines[lineno - 1]

            if text.strip().startswith("# TeX: "):
                frags = text.split()
                if len(frags) <= 2:
                    raise ValueError("Missing directive name")
                directive = frags[2]
                if directive == "color":
                    color_letters = eat_next_line()
                    if len(frags) <= 3:
                        raise ValueError("Missing directive name after 'color' (e.g. say 'color line'")
                    directive = frags[3]
                    args = frags[4:]
                else:
                    color_letters = ""
                    args = frags[3:]
                if directive == "version":
                    if len(args) != 2:
                        raise ValueError("Expect [name] [version_count] for 'version' directive")
                    name, version_count_txt = args
                    result.append(VersionDirective(name, int(version_count_txt)))
                    version_names.add(name)
                elif directive == "begin":
                    result.append(BeginDirective(parse_filters(version_names, args), lineno))
                elif directive == "end":
                    result.append(EndDirective(parse_filters(version_names, args), lineno))
                elif directive == "summary":
                    text = eat_next_line()
                    if len(args) != 0:
                        raise ValueError("Summary expects no filters")
                    result.append(SummaryDirective(text, color_letters, False, lineno))
                elif directive == "summary!":
                    text = eat_next_line()
                    if len(args) != 0:
                        raise ValueError("Summary expects no filters")
                    result.append(SummaryDirective(text, color_letters, True, lineno))
                elif directive == "remark":
                    text = eat_next_line()
                    result.append(RemarkDirective(parse_filters(version_names, args), text, color_letters, False, lineno))
                elif directive == "remark!":
                    text = eat_next_line()
                    result.append(RemarkDirective(parse_filters(version_names, args), text, color_letters, True, lineno))
                elif directive == "line":
                    if not color_letters:
                        raise ValueError("line directive with no/empty color info")
                    result.append(ColorDirective(parse_filters(version_names, args), color_letters))
                elif directive == "filbreak":
                    if len(args) != 0:
                        raise ValueError("filbreak expects no filters")
                    result.append(FilbreakDirective(lineno))
                else:
                    raise ValueError(f"Unknown directive {directive!r}")
            else:
                if '#' in text and 'TEX' in text.upper():
                    print(f"Line {lineno}: mistyped TeX directive?")
                result.append(SourceLine(text, lineno))
        except Exception as e:
            directive_str = ""
            if directive is not None:
                directive_str = f"(for {directive} directive) "
            raise ValueError(f"Line {lineno}: {directive_str}{e}") from e

    return result

def lines_directives_to_tex(lines_directives, name, version_number):
    tex_lines = []
    refcnt = 0
    source_color_letters = ""
    for directive in lines_directives:
        if isinstance(directive, SourceLine):
            if refcnt > 0:
                tex_lines.append(TexLine(directive.lineno, False, False, directive.text, source_color_letters))
            source_color_letters = ""
        elif isinstance(directive, VersionDirective):
            pass
        elif isinstance(directive, BeginDirective):
            if directive.passes_filter(name, version_number):
                refcnt += 1
        elif isinstance(directive, EndDirective):
            if directive.passes_filter(name, version_number):
                refcnt -= 1
                if refcnt < 0:
                    raise ValueError(f"Line {directive.lineno}: end without begin for {name}.{version_number}")
        elif isinstance(directive, SummaryDirective):
            tex_lines.append(TexLine(directive.lineno, refcnt == 0, False, directive.text, directive.color_letters, refcnt > 0 and directive.bold))
        elif isinstance(directive, RemarkDirective):
            if refcnt > 0:
                if directive.passes_filter(name, version_number):
                    tex_lines.append(TexLine(directive.lineno, False, True, directive.text, directive.color_letters, directive.bold))
        elif isinstance(directive, ColorDirective):
            if directive.passes_filter(name, version_number):
                source_color_letters = directive.color_letters
        elif isinstance(directive, FilbreakDirective):
            if refcnt > 0:
                line = TexLine(directive.lineno, False, False, "\\filbreak", "")
                line.raw_text_is_latex = True
                tex_lines.append(line)
        else:
            assert 0, str(type(directive))

    # Strip leading and trailing summaries
    while tex_lines and tex_lines[-1].is_summary:
        tex_lines.pop()
    leading_count = 0
    for i, tex_line in enumerate(tex_lines):
        if tex_line.is_summary:
            leading_count = i + 1
        else:
            break
    tex_lines = tex_lines[leading_count:]

    # Determine leading whitespace to strip
    dedent = min((line.leading_spaces() for line in tex_lines), default=0)
    if math.isinf(dedent):
        dedent = 0

    # Join lines
    output_lines = [line.gen_tex(dedent) for line in tex_lines]
    snippets = []
    for i, line in enumerate(output_lines):
        # Add LaTeX endlines to all lines except the last one, \filbreak, and those followed by \filbreak
        snippets.append(line)
        if line == "\\filbreak":
            snippets.append("\n")
        elif i + 1 == len(output_lines) or output_lines[i + 1] == "\\filbreak":
            snippets.append("\n")
        else:
            snippets.append("\\\\\n")
    return "".join(snippets)
    return "\n".join(line.gen_tex(dedent) for line in tex_lines)

def main(argv):
    if len(argv) != 3:
        print("Args: [input.py] [output dir]", file=sys.stderr)
        sys.exit(1)
    _, input_name, output_dir_name = argv

    os.makedirs(output_dir_name, exist_ok=True)

    with open(input_name) as f:
        parsed_lines = file_to_lines_directives(f)

    # Census of all named versions and version counts.
    version_count_dict = {}
    for directive in parsed_lines:
        if isinstance(directive, VersionDirective):
            tmp = version_count_dict.get(directive.name, 0)
            version_count_dict[directive.name] = max(directive.version_count, tmp)

    # Generate all versions
    for name, version_count in version_count_dict.items():
        for version_number in range(0, version_count):
            tex = lines_directives_to_tex(parsed_lines, name, version_number)
            filename = f"{output_dir_name}/{name}.{version_number}.tex"
            print(f"Generating {filename}...")
            with open(filename, "w") as f:
                f.write(tex)


if __name__ == "__main__":
    main(sys.argv)
