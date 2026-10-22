"""Unit tests of the listing generator (no LaTeX needed)."""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_PARENT))

from code_to_tex.listing import ListingError, generate, render_outputs  # noqa: E402
from code_to_tex.listing_parse import tokenize  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "listing_fixtures"
LEGACY_NYC25 = PACKAGE_PARENT / "nyc25_tex"


def render(src, warnings=None):
    return render_outputs(textwrap.dedent(src).lstrip("\n"), "t.py", warn=(warnings.append if warnings is not None else lambda m: None))


def txt(src, name="a", v=0):
    return render(src)[f"{name}.{v}.txt"]


def dbg(src, name="a", v=0):
    return render(src)[f"{name}.{v}.dbg"]


def tex(src, name="a"):
    return render(src)[f"{name}.tex"]


def error(src):
    with pytest.raises(ListingError) as info:
        render(src)
    return info.value


def assert_error(src, lineno, pattern):
    e = error(src)
    assert e.lineno == lineno, str(e)
    assert re.search(pattern, e.message), str(e)
    assert str(e) == f"t.py:{lineno}: {e.message}"


SIMPLE = """
# TeX: listing a 1
# TeX: begin a
x = 1
# TeX: end a
"""


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenize_quoting():
    toks = tokenize('color a red "A[m, k]" 2:1')
    assert [t.value for t in toks] == ["color", "a", "red", "A[m, k]", "2:1"]
    assert [t.quoted for t in toks] == [False, False, False, True, False]
    toks = tokenize(r"""x 'it"s' "a\"b" c\ d ''""")
    assert [t.value for t in toks] == ["x", 'it"s', 'a"b', "c d", ""]
    assert [t.quoted for t in toks] == [False, True, True, False, True]


def test_tokenize_unterminated_quote():
    assert_error(
        """
        # TeX: listing a 1
        # TeX: color a red "oops
        x = 1
        """,
        2, "unterminated",
    )


# ---------------------------------------------------------------------------
# Listings, regions, filters
# ---------------------------------------------------------------------------


def test_simple_outputs():
    out = render(SIMPLE)
    assert set(out) == {"a.tex", "a.0.txt", "a.0.dbg"}
    assert out["a.0.txt"] == "x = 1\n"
    assert out["a.0.dbg"] == "x = 1\n"
    assert out["a.tex"].startswith("%%")
    assert "a.N.txt" in out["a.tex"] and ".dbg" in out["a.tex"]
    assert r"\sporkListingBegin{a}{1}" in out["a.tex"]
    assert out["a.tex"].rstrip().endswith(r"\sporkListingEnd")


def test_versions_and_filters():
    src = """
    # TeX: listing a 5
    # TeX: begin a
    all
    # TeX: end a
    # TeX: begin a[1]
    one
    # TeX: end a[1]
    # TeX: begin a[1:3]
    one_two
    # TeX: end a[1:3]
    # TeX: begin a[3:]
    three_up
    # TeX: end a[3:]
    # TeX: begin a[:2]
    below_two
    # TeX: end a[:2]
    # TeX: begin a[:]
    every
    # TeX: end a[:]
    """
    out = render(src)
    assert out["a.0.txt"].split() == ["all", "below_two", "every"]
    assert out["a.1.txt"].split() == ["all", "one", "one_two", "below_two", "every"]
    assert out["a.2.txt"].split() == ["all", "one_two", "every"]
    assert out["a.3.txt"].split() == ["all", "three_up", "every"]
    assert out["a.4.txt"].split() == ["all", "three_up", "every"]


def test_star_filter_and_multiple_listings():
    src = """
    # TeX: listing a 1
    # TeX: listing b 2
    # TeX: begin *
    shared
    # TeX: end b[0],a
    # TeX: begin b[1]
    b1
    # TeX: end b[1]
    # TeX: end b[1]
    """
    out = render(src)
    assert out["a.0.txt"] == "shared\n"
    assert out["b.0.txt"] == "shared\n"
    assert out["b.1.txt"] == "shared\nb1\n"


def test_nested_regions_refcount():
    src = """
    # TeX: listing a 1
    # TeX: begin a
    one
    # TeX: begin a
    two
    # TeX: end a
    three
    # TeX: end a
    four
    """
    assert txt(src).split() == ["one", "two", "three"]


@pytest.mark.parametrize(
    "filt, pattern",
    [
        ("b", "unknown listing 'b'"),
        ("a[2]", "out of range"),
        ("a[1:1]", "empty or out-of-range"),
        ("a[0:3]", "empty or out-of-range"),
        ("a[x]", "bad filter"),
        ("a[]", "empty index"),
        ("a[0", "bad filter"),
        ("a[-1]", "bad filter"),
    ],
)
def test_bad_filters(filt, pattern):
    assert_error(
        f"""
        # TeX: listing a 2
        # TeX: begin {filt}
        x
        # TeX: end a
        """,
        2, pattern,
    )


def test_listing_must_be_declared_before_use():
    assert_error(
        """
        # TeX: begin a
        x
        # TeX: end a
        # TeX: listing a 1
        """,
        1, "unknown listing 'a'.*earlier",
    )


@pytest.mark.parametrize(
    "decl, pattern",
    [
        ("listing a", "expected 'listing NAME COUNT'"),
        ("listing a 0", "bad version count"),
        ("listing a two", "bad version count"),
        ("listing 1a 1", "bad listing name"),
        ("version a 1", "renamed: use 'listing NAME COUNT'"),
        ("frobnicate", "unknown directive 'frobnicate'"),
        ("", "missing directive name"),
        ("summary x", "takes no arguments"),
        ("filbreak x", "takes no arguments"),
        ("begin", "needs filters as its first argument"),
        ('begin "a"', "needs filters.*got quoted text"),
        ("end red", "needs filters.*got 'red'"),
    ],
)
def test_bad_directives(decl, pattern):
    assert_error(f"# TeX: {decl}\nx\n", 1, pattern)


def test_duplicate_listing():
    assert_error("# TeX: listing a 1\n# TeX: listing a 2\n", 2, "already declared on line 1")


def test_end_without_begin():
    assert_error(
        """
        # TeX: listing a 2
        # TeX: begin a[0]
        x
        # TeX: end a
        """,
        4, r"'end' without a matching 'begin' for a\[1\]",
    )


def test_unclosed_begin():
    assert_error(
        """
        # TeX: listing a 1
        # TeX: begin a
        x
        """,
        2, "never closed",
    )


def test_empty_version():
    assert_error(
        """
        # TeX: listing a 2
        # TeX: begin a[0]
        x
        # TeX: end a[0]
        """,
        1, r"a\[1\] is empty",
    )


def test_blank_lines_kept_and_dedent():
    src = """
    # TeX: listing a 2
        # TeX: begin a
        def f():

            # TeX: begin a[1]
            pass
            # TeX: end a[1]
        # TeX: end a
    # TeX: begin a[1]
            tail
    # TeX: end a[1]
    """
    assert txt(src, v=0) == "def f():\n\n    pass\n"
    assert txt(src, v=1) == "def f():\n\n    pass\n    tail\n"


def test_dedent_differs_per_version_gives_variants():
    src = """
    # TeX: listing a 2
    # TeX: begin a[0]
    x
    # TeX: end a[0]
    # TeX: begin a
        y
    # TeX: end a
    """
    assert txt(src, v=0) == "x\n    y\n"
    assert txt(src, v=1) == "y\n"
    t = tex(src)
    assert r"\sporkListingVariant{0}{~~~~y}" in t
    assert r"\sporkListingVariant{1}{y}" in t
    assert r"\sporkListingRow{0,1}{" in t


def test_identical_versions_share_a_variant():
    src = """
    # TeX: listing a 3
    # TeX: begin a
    x = 1
    # TeX: end a
    """
    assert r"\sporkListingVariant{0,1,2}{x~=~1}" in tex(src)


def test_tabs_rejected_in_shown_lines_only():
    assert_error("# TeX: listing a 1\n# TeX: begin a\n\tx\n# TeX: end a\n", 3, "tab character")
    assert txt("# TeX: listing a 1\n\tignored\n# TeX: begin a\nx\n# TeX: end a\n") == "x\n"


# ---------------------------------------------------------------------------
# remark / summary / filbreak
# ---------------------------------------------------------------------------


def test_remark():
    src = """
    # TeX: listing a 2
    # TeX: begin a
        x = 1
        # TeX: remark a[1]
        # plain remark
        # TeX: remark! *
        # bold remark
    # TeX: end a
    """
    out = render(src)
    assert out["a.0.txt"] == "x = 1\n# bold remark\n"
    assert out["a.1.txt"] == "x = 1\n# plain remark\n# bold remark\n"
    assert out["a.1.dbg"] == "x = 1\n⟦remark:# plain remark⟧\n⟦remark!:# bold remark⟧\n"
    t = out["a.tex"]
    assert r"\sporkListingVariant{1}{\sporkComment{\#~plain~remark}}" in t
    assert r"\sporkListingVariant{0,1}{\textbf{\sporkComment{\#~bold~remark}}}" in t


def test_remark_payload_must_be_comment():
    assert_error("# TeX: listing a 1\n# TeX: begin a\n# TeX: remark a\nx = 1\n# TeX: end a\n", 4, "must be a '#' comment")


def test_remark_payload_may_not_be_directive():
    assert_error("# TeX: listing a 1\n# TeX: begin a\n# TeX: remark a\n# TeX: end a\n", 4, "got a directive")


def test_remark_at_eof():
    assert_error("# TeX: listing a 1\n# TeX: remark a\n", 2, "end of file")


def test_dead_remark():
    assert_error(
        """
        # TeX: listing a 2
        # TeX: begin a[0]
        x
        # TeX: end a[0]
        # TeX: begin a
        # TeX: end a
        # TeX: remark a[1]
        # outside
        # TeX: begin a[1]
        y
        # TeX: end a[1]
        """,
        7, r"remark .*outside every begin/end region of a\[1\]",
    )


def test_star_remark_needs_some_region():
    assert_error(
        """
        # TeX: listing a 1
        # TeX: remark *
        # nowhere
        # TeX: begin a
        x
        # TeX: end a
        """,
        2, "any listing version",
    )


SUMMARY = """
# TeX: listing a 2
# TeX: begin a
head
# TeX: end a
# TeX: begin a[1]
# TeX: summary
# collapsed loop
for i in x:
    body
# TeX: end a[1]
# TeX: begin a
tail
# TeX: end a
# TeX: summary!
# trailing summary
"""


def test_summary_placeholder_and_inline():
    out = render(SUMMARY)
    assert out["a.0.txt"] == "head\n# ... collapsed loop\ntail\n"
    assert out["a.0.dbg"] == "head\n⟦placeholder:# ... collapsed loop⟧\ntail\n"
    # Inside a region a summary is an ordinary line.
    assert out["a.1.txt"] == "head\n# collapsed loop\nfor i in x:\n    body\ntail\n"
    assert out["a.1.dbg"].splitlines()[1] == "⟦summary:# collapsed loop⟧"
    t = out["a.tex"]
    assert r"\sporkListingVariant{0}{\textit{\sporkComment{\#~...~collapsed~loop}}}" in t
    assert "trailing" not in t  # trailing placeholder stripped


def test_summary_bold_in_region_and_leading_placeholder_stripped():
    src = """
    # TeX: listing a 1
    # TeX: summary!
    # leading
    # TeX: begin a
    x
    # TeX: summary!
    # inside
    # TeX: end a
    """
    out = render(src)
    assert out["a.0.txt"] == "x\n# inside\n"
    assert out["a.0.dbg"] == "x\n⟦summary!:# inside⟧\n"
    assert r"\textbf{\sporkComment{\#~inside}}" in out["a.tex"]


def test_filbreak():
    src = """
    # TeX: listing a 2
    # TeX: begin a
    x
    # TeX: filbreak
    y
    # TeX: end a
    """
    out = render(src)
    assert out["a.0.txt"] == "x\ny\n"
    assert out["a.0.dbg"] == "x\n⟦filbreak⟧\ny\n"
    assert r"\sporkListingFilbreak{0,1}" in out["a.tex"]


def test_filbreak_outside_region_ignored():
    out = render("# TeX: listing a 1\n# TeX: filbreak\n# TeX: begin a\nx\n# TeX: end a\n")
    assert "Filbreak" not in out["a.tex"]


# ---------------------------------------------------------------------------
# color
# ---------------------------------------------------------------------------


def colored(directives, line="accum += A[m, k] * B[k, n]  # note", count=1):
    body = "".join(f"# TeX: {d}\n" for d in directives)
    return f"# TeX: listing a {count}\n# TeX: begin a\n{body}{line}\n# TeX: end a\n"


def test_color_single_match():
    src = colored(['color a blue "accum"'])
    out = render(src)
    assert out["a.0.dbg"] == "⟦blue:accum⟧ += A[m, k] * B[k, n]  # note\n"
    assert out["a.0.txt"] == "accum += A[m, k] * B[k, n]  # note\n"
    assert r"\sporkListingBox{\sporkBlue}{accum}~+=" in out["a.tex"]


def test_color_match_count_required():
    e = error(colored(['color a green "m"']))
    assert e.lineno == 3
    assert "matches 2 time(s) on line 4, expected 1" in e.message
    assert "2:0" in e.message  # hint


def test_color_wrong_explicit_count():
    assert_error(colored(['color a green "m" 3:0']), 3, "matches 2 time.*expected 3")
    assert_error(colored(['color a green "zzz"']), 3, "matches 0 time")


def test_color_index_and_all():
    assert dbg(colored(['color a green "m" 2:1'])) == "accu⟦green:m⟧ += A[m, k] * B[k, n]  # note\n".replace(
        "accu⟦green:m⟧ += A[m", "accum += A[⟦green:m⟧"
    )
    assert dbg(colored(['color a green "m" 2:0'])) == "accu⟦green:m⟧ += A[m, k] * B[k, n]  # note\n"
    assert dbg(colored(['color a red "k" 2:*'])) == "accum += A[m, ⟦red:k⟧] * B[⟦red:k⟧, n]  # note\n"


@pytest.mark.parametrize(
    "spec, pattern",
    [
        ("2:2", "need N >= 1 and 0 <= I < N"),
        ("0:0", "need N >= 1"),
        ("2:x", "expected match count"),
        ("2:1 extra", "unknown word 'extra'.*must be quoted"),
        ("2:1 3:0", "must directly follow a quoted string.*follows '2:1'"),
    ],
)
def test_color_bad_count(spec, pattern):
    assert_error(colored([f'color a red "k" {spec}']), 3, pattern)


def test_color_matches_are_non_overlapping():
    # "aa" in "aaaa" matches twice (columns 0 and 2), not three times.
    assert dbg(colored(['color a red "aa" 2:1'], line="aaaa")) == "aa⟦red:aa⟧\n"


def test_color_syntax_errors():
    assert_error(colored(["color a blue accum"]), 3, "unknown word 'accum'.*must be quoted: \"accum\"")
    assert_error(colored(['color a blue ""']), 3, "empty text")
    assert_error(colored(['color a purple "accum"']), 3, "unknown word 'purple'.*colors: red, ")
    assert_error(colored(['color a "accum"']), 3, "before any color name")
    assert_error(colored(["color a"]), 3, "needs a color name after the filters")
    assert_error(colored(['color blue "accum"']), 3, "needs filters as its first argument.*got 'blue'")
    assert_error(colored(["color line a"]), 3, "legacy 'color line'")
    assert_error(colored(['color "a" blue "accum"']), 3, "needs filters.*got quoted text")


def test_all_color_names():
    names = ["red", "yellow", "green", "blue", "violet", "invRed", "invGold", "invGreen", "invBlue", "invPurple"]
    line = " ".join(f"w{i}" for i in range(len(names)))
    t = tex(colored([f'color a {n} "w{i}"' for i, n in enumerate(names)], line=line))
    for n in names:
        assert rf"\sporkListingBox{{\spork{n[0].upper() + n[1:]}}}" in t


def test_same_color_twice_is_noop_and_spans_merge():
    out = render(colored(['color a red "accum"', 'color a red "accum"', 'color a red " +="'], line="accum += 1"))
    assert out["a.0.dbg"] == "⟦red:accum +=⟧ 1\n"
    assert out["a.tex"].count(r"\sporkListingBox") == 1
    # overlapping same-color spans merge too
    out = render(colored(['color a red "accu"', 'color a red "cum +"'], line="accum += 1"))
    assert out["a.0.dbg"] == "⟦red:accum +⟧= 1\n"


def test_adjacent_different_colors_stay_separate():
    assert dbg(colored(['color a red "ab"', 'color a blue "cd"'], line="abcd")) == "⟦red:ab⟧⟦blue:cd⟧\n"


def test_conflicting_colors():
    e = error(colored(['color a red "accum"', 'color a blue "cum +"']))
    assert e.lineno == 4
    assert "conflicting colors in a[0]" in e.message
    assert "line 3" in e.message


def test_conflict_only_checked_per_version():
    src = colored(['color a[0] red "accum"', 'color a[1] blue "accum"'], count=2)
    out = render(src)
    assert out["a.0.dbg"].startswith("⟦red:accum⟧")
    assert out["a.1.dbg"].startswith("⟦blue:accum⟧")
    t = out["a.tex"]
    assert r"\sporkListingVariant{0}{\sporkListingBox{\sporkRed}" in t
    assert r"\sporkListingVariant{1}{\sporkListingBox{\sporkBlue}" in t


def test_elide():
    out = render(colored(['color a elide "m, k"']))
    assert out["a.0.txt"] == "accum += A[...] * B[k, n]  # note\n"
    assert out["a.0.dbg"] == "accum += A[⟦elide:m, k⟧] * B[k, n]  # note\n"
    assert r"A[\sporkComment{...}]" in out["a.tex"]


def test_adjacent_elides_merge():
    out = render(colored(['color a elide "ab"', 'color a elide "cd"'], line="xabcdx"))
    assert out["a.0.txt"] == "x...x\n"


def test_elide_conflicts_with_color():
    assert_error(
        colored(['color a elide "m, k"', 'color a red "k]"']), 4,
        r"conflicting colors.*red here \('k\]'\) but elide \('m, k'\) by the directive on line 3",
    )


def test_color_dead_directive():
    src = """
    # TeX: listing a 2
    # TeX: begin a
    x
    # TeX: end a
    # TeX: begin a[1]
    # TeX: color a red "y"
    y
    # TeX: end a[1]
    """
    assert_error(src, 6, r"dead 'color' directive: its target line 7 is not shown in a\[0\] \(selected by filter 'a'\)")


def test_color_dead_after_end():
    src = """
    # TeX: listing a 1
    # TeX: begin a
    x
    # TeX: color a red "y"
    # TeX: end a
    y
    """
    assert_error(src, 4, r"dead 'color'.*target line 6")


def test_star_color_is_lenient_but_needs_some_version():
    src = """
    # TeX: listing a 1
    # TeX: listing b 1
    # TeX: begin a
    x
    # TeX: end a
    # TeX: begin b
    # TeX: color * red "y"
    y
    # TeX: end b
    """
    assert dbg(src, "b") == "⟦red:y⟧\n"
    src = """
    # TeX: listing a 1
    # TeX: begin a
    x
    # TeX: end a
    # TeX: color * red "y"
    y
    """
    assert_error(src, 5, "any listing version")


def test_color_dead_on_summary_placeholder():
    src = """
    # TeX: listing a 2
    # TeX: begin a
    x
    # TeX: end a
    # TeX: begin a[1]
    # TeX: color a red "loop"
    # TeX: summary
    # loop
    # TeX: end a[1]
    # TeX: begin a
    z
    # TeX: end a
    """
    assert_error(src, 6, r"dead 'color'.*a\[0\]")
    assert dbg(src.replace("color a red", "color a[1] red"), v=1) == "x\n⟦summary:# ⟦red:loop⟧⟧\nz\n"


def test_color_at_eof():
    assert_error("# TeX: listing a 1\n# TeX: begin a\nx\n# TeX: end a\n# TeX: color a red \"x\"\n", 5, "end of file")


def test_color_skips_directives_and_targets_remark():
    src = """
    # TeX: listing a 1
    # TeX: color a red "x"
    # TeX: begin a
    x = 1
    # TeX: color a blue "note"
    # TeX: remark! a
    # note
    # TeX: end a
    """
    out = render(src)
    assert out["a.0.dbg"] == "⟦red:x⟧ = 1\n⟦remark!:# ⟦blue:note⟧⟧\n"
    # inside a box, comment text keeps the box colors and only switches font
    assert r"\textbf{\sporkComment{\#~}\sporkListingBox{\sporkBlue}{\textsf{note}}}" in out["a.tex"]


def test_color_dead_on_filtered_remark():
    src = """
    # TeX: listing a 2
    # TeX: begin a
    x
    # TeX: color a blue "note"
    # TeX: remark a[1]
    # note
    # TeX: end a
    """
    assert_error(src, 4, r"dead 'color'.*a\[0\]")


def test_colors_stack_on_one_line():
    src = colored(['color a red "accum"', 'color a green "A"', 'color a violet "B"', 'color a yellow "note"'])
    assert dbg(src) == "⟦red:accum⟧ += ⟦green:A⟧[m, k] * ⟦violet:B⟧[k, n]  # ⟦yellow:note⟧\n"


def test_multi_color_directive_sticky_colors():
    # color words are sticky until the next color word; N:I applies to the string just before it
    src = colored(['color a red "accum" "A" green "m" 2:1 "B" elide "  # note"'])
    assert dbg(src) == "⟦red:accum⟧ += ⟦red:A⟧[⟦green:m⟧, k] * ⟦green:B⟧[k, n]⟦elide:  # note⟧\n"
    assert dbg(src) == dbg(colored(
        ['color a red "accum"', 'color a red "A"', 'color a green "m" 2:1', 'color a green "B"', 'color a elide "  # note"']
    ))


def test_multi_color_directive_counts_and_merging():
    src = colored(['color a violet "k" 2:* "n]" blue "accum" " +="'])
    assert dbg(src) == "⟦blue:accum +=⟧ A[m, ⟦violet:k⟧] * B[⟦violet:k⟧, ⟦violet:n]⟧  # note\n"
    # each quoted string is checked on its own: "m" alone is ambiguous
    e = error(colored(['color a red "accum" "m"']))
    assert e.lineno == 3 and "'m' matches 2 time(s)" in e.message


def test_multi_color_directive_conflict_within_one_directive():
    assert_error(
        colored(['color a red "accum" blue "cum"']), 3,
        r"red here \('accum'\) but blue \('cum'\) by the directive on line 3|blue here \('cum'\) but red \('accum'\)",
    )
    # the same color twice in one directive is a no-op
    assert dbg(colored(['color a red "accum" "accum"'])).startswith("⟦red:accum⟧ +=")


@pytest.mark.parametrize(
    "directive, pattern",
    [
        ('color a red', "color 'red' has no quoted text"),
        ('color a red blue "accum"', "color 'red' has no quoted text"),
        ('color a red "accum" green', "color 'green' has no quoted text"),
        ('color a 2:0 red "m"', "must directly follow a quoted string \\(it follows the filters\\)"),
        ('color a red 2:0 "m"', "must directly follow a quoted string \\(it follows 'red'\\)"),
        ('color a red "m" green 2:0', "must directly follow a quoted string \\(it follows 'green'\\)"),
        ('color a red "accum" frob', "unknown word 'frob'"),
        ('color a red "m" 2:', "expected match count"),
        ('color a red "accum" before', "unknown word 'before'"),
    ],
)
def test_color_grammar_errors(directive, pattern):
    assert_error(colored([directive]), 3, pattern)


def test_comma_filters_on_every_directive():
    src = """
    # TeX: listing a 3
    # TeX: listing b 2
    # TeX: begin a[0],a[2],b[1]
    # TeX: color a[2],b[1] red "x"
    # TeX: mark a[0],b[1] m end
    x = 1
    # TeX: remark! a[2],b[1]
    # bold
    # TeX: end a[0],a[2],b[1]
    # TeX: begin *
    y
    # TeX: end *
    """
    out = render(src)
    assert out["a.0.dbg"] == "x = 1⟦@m⟧\ny\n"
    assert out["a.1.dbg"] == "y\n"
    assert out["a.2.dbg"] == "⟦red:x⟧ = 1\n⟦remark!:# bold⟧\ny\n"
    assert out["b.0.dbg"] == "y\n"
    assert out["b.1.dbg"] == "⟦red:x⟧ = 1⟦@m⟧\n⟦remark!:# bold⟧\ny\n"


def test_dead_directive_message_shows_comma_filters():
    src = """
    # TeX: listing a 1
    # TeX: listing b 1
    # TeX: begin a
    x
    # TeX: end a
    # TeX: color *,* red "y"
    y
    """
    assert_error(src, 6, r"any listing version selected by '\*,\*'")


def test_listing_named_like_a_color_or_position():
    src = """
    # TeX: listing red 2
    # TeX: listing end 1
    # TeX: begin red,end
    # TeX: color red[1] red "x" blue "y"
    # TeX: mark end e end
    x = y
    # TeX: end red,end
    """
    out = render(src)
    assert out["red.0.dbg"] == "x = y\n"
    assert out["red.1.dbg"] == "⟦red:x⟧ = ⟦blue:y⟧\n"
    assert out["end.0.dbg"] == "x = y⟦@e⟧\n"


@pytest.mark.parametrize(
    "directive, pattern",
    [
        ("begin a[0] b", "filters must be one comma-joined word, without spaces: a\\[0\\],b$"),
        ("end a b[1:]", "comma-joined word, without spaces: a,b\\[1:\\]$"),
        ("begin a, b", "comma-joined word, without spaces: a,b$"),
        ("begin a ,b", "comma-joined word, without spaces: a,b$"),
        ("begin a,", "comma-joined word, without spaces: a$"),
        ("begin a,,b", "empty filter in 'a,,b'"),
        ("remark! a b", "comma-joined word, without spaces: a,b$"),
        ("begin a x", "unexpected argument 'x' after the filters of 'begin'"),
        ('color a b red "x"', "unknown word 'b'.*comma-joined word, without spaces: a,b$"),
        ('color a b[0] * red "x"', "unknown word 'b\\[0\\]'.*comma-joined word, without spaces: a,b\\[0\\]$"),
        ("mark a b[1] m end", "bad mark name 'b\\[1\\]'.*comma-joined word, without spaces: a,b\\[1\\]$"),
        ("mark a b m end", "unexpected mark argument 'm' \\(if 'b' is a filter: .*: a,b, then the mark name\\)"),
    ],
)
def test_old_space_separated_filters_get_a_hint(directive, pattern):
    src = f"""
    # TeX: listing a 1
    # TeX: listing b 2
    # TeX: begin *
    # TeX: {directive}
    # x
    # TeX: end *
    """
    assert_error(src, 4, pattern)


# ---------------------------------------------------------------------------
# mark
# ---------------------------------------------------------------------------


def test_mark_positions():
    src = colored(
        [
            'mark a s start',
            'mark a e end',
            'mark a b "A" before',
            'mark a f "B" after',
            'mark a d "k" 2:1',
        ],
        line="    accum += A[m, k] * B[k, n]  ",
    )
    assert dbg(src) == "⟦@s⟧accum += ⟦@b⟧A[m, k] * B⟦@f⟧[⟦@d⟧k, n]⟦@e⟧\n"


def test_mark_tex_and_shifts():
    src = colored(['mark a m1 "A" xshift=1pt yshift=-0.5ex', 'mark a m2 end'])
    out = render(src)
    t = out["a.tex"]
    assert r"\sporkListingMark{m1}{1pt}{-0.5ex}A[m" in t
    assert t.count(r"\sporkListingMark{m2}{}{}") == 1
    assert "⟦@m1 xshift=1pt yshift=-0.5ex⟧A" in out["a.0.dbg"]


def test_mark_at_box_boundary_is_outside_box_and_inside_otherwise():
    t = tex(colored(['color a blue "accum"', 'mark a x "accum"', 'mark a y "accum" after', 'mark a z "cum"'], line="accum = 0"))
    assert (
        r"\sporkListingMark{x}{}{}\sporkListingBox{\sporkBlue}{ac\sporkListingMark{z}{}{}cum}\sporkListingMark{y}{}{}~=~0"
        in t
    )


def test_mark_on_empty_line():
    src = "# TeX: listing a 1\n# TeX: begin a\nx\n# TeX: mark a e end\n\n# TeX: end a\n"
    assert dbg(src) == "x\n⟦@e⟧\n"


@pytest.mark.parametrize(
    "directive, pattern",
    [
        ('mark a', "missing mark name; expected 'mark FILTERS NAME"),
        ('mark a before', "missing mark name \\('before' is not a name\\)"),
        ('mark a "A"', "missing mark name"),
        ('mark x m start', "unknown listing 'x'"),
        ('mark a m before', "needs quoted text"),
        ('mark a m after', "needs quoted text"),
        ('mark a m "A" start', "cannot be combined with text"),
        ('mark a m "A" before after', "two mark positions"),
        ('mark a m "k" 2:*', "N:I, not N:\\*"),
        ('mark a m "A" xshift=1', "bad xshift"),
        ('mark a m "A" yshift=abc', "bad yshift"),
        ('mark a m "A" xshift=1pt xshift=2pt', "given twice"),
        ('mark a m "A" bogus', "unexpected mark argument"),
        ('mark a m end "A"', "must directly follow the mark name"),
        ('mark a m-x end', "bad mark name"),
        ('mark a m start 2:1', "must directly follow the quoted mark text"),
        ('mark a m "k"', "matches 2 time"),
    ],
)
def test_mark_errors(directive, pattern):
    assert_error(colored([directive]), 3, pattern)


def test_duplicate_mark_in_one_version():
    src = """
    # TeX: listing a 2
    # TeX: begin a
    # TeX: mark a m start
    x
    # TeX: mark a[1] m start
    y
    # TeX: end a
    """
    assert_error(src, 5, r"duplicate mark 'm' in a\[1\] \(also on line 3\)")
    # Same name in different versions (and lines) is fine.
    assert dbg(src.replace("mark a m start", "mark a[0] m start"), v=1) == "x\n⟦@m⟧y\n"


def test_mark_inside_elide():
    assert_error(colored(['color a elide "accum"', 'mark a m "cum"']), 4, "inside elided text")
    # at the boundary it is fine
    assert dbg(colored(['color a elide "accum"', 'mark a m "accum" after'])).startswith("⟦elide:accum⟧⟦@m⟧")


def test_mark_dead_directive():
    src = """
    # TeX: listing a 2
    # TeX: begin a[0]
    # TeX: mark a m start
    x
    # TeX: end a[0]
    # TeX: begin a[1]
    y
    # TeX: end a[1]
    """
    assert_error(src, 3, r"dead 'mark'.*a\[1\]")


def test_mark_name_may_equal_listing_name():
    assert dbg(colored(["mark a a start"])).startswith("⟦@a⟧")


# ---------------------------------------------------------------------------
# Escaping, comments, math
# ---------------------------------------------------------------------------


def test_escaping():
    t = tex(colored([], line=r"a_b = {c} % d & e ~ ^ \ $"))
    assert r"a\_b~=~\{c\}~\%~d~\&~e~{\textasciitilde}~{\textasciicircum}~{\textbackslash}~\$" in t


def test_comment_and_math():
    t = tex(colored([], line=r"x = 1  # cost $O(n^2)$ #_%"))
    assert r"x~=~1~~\sporkComment{\#~cost~$O(n^2)$~\#\_\%}" in t


def test_hash_in_string_starts_comment_naively():
    assert r'f("\sporkComment{\#")}' in tex(colored([], line='f("#")'))


def test_unbalanced_dollar_in_comment():
    assert_error(colored([], line="x  # costs $5"), 3, "unbalanced '\\$'")


def test_dollar_in_code_is_escaped():
    assert r"x~=~\$y" in tex(colored([], line="x = $y"))


def test_color_boundary_inside_math():
    assert_error(colored(['color a red "n^"'], line="x  # $O(n^2)$"), 4, "inside \\$...\\$ math")
    # coloring the whole math region is fine
    assert dbg(colored(['color a red "$O(n^2)$"'], line="x  # $O(n^2)$")) == "x  # ⟦red:$O(n^2)$⟧\n"


def test_mark_inside_math():
    assert_error(colored(['mark a m "n"'], line="x  # $O(n)$"), 3, "inside \\$...\\$ math")


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("line", ["#TeX: begin a", "# tex: begin a", "  # TeX:begin a", "x = 1  # TeX: begin a", "# TeX : x"])
def test_mistyped_directive_warns(line):
    warnings = []
    render(f"# TeX: listing a 1\n{line}\n# TeX: begin a\nx\n# TeX: end a\n", warnings)
    assert len(warnings) == 1
    assert warnings[0].startswith("t.py:2: warning: looks like a mistyped directive")


@pytest.mark.parametrize("line", ["# compute the texture", "x = latex_thing", "# TeXture"])
def test_no_false_positive_warning(line):
    warnings = []
    render(f"# TeX: listing a 1\n{line}\n# TeX: begin a\nx\n# TeX: end a\n", warnings)
    assert warnings == []


# ---------------------------------------------------------------------------
# Files, CLI, realistic fixture
# ---------------------------------------------------------------------------


def test_generate_writes_files_and_keeps_unchanged(tmp_path):
    src = tmp_path / "s.py"
    src.write_text(SIMPLE)
    out = tmp_path / "code"
    (out).mkdir()
    (out / "unrelated.tex").write_text("keep")
    paths = generate(str(src), str(out))
    assert sorted(os.path.basename(p) for p in paths) == ["a.0.dbg", "a.0.txt", "a.tex"]
    assert (out / "unrelated.tex").read_text() == "keep"
    assert "from s.py" in (out / "a.tex").read_text()
    os.utime(out / "a.tex", (1, 1))
    generate(str(src), str(out))
    assert os.stat(out / "a.tex").st_mtime == 1  # unchanged content -> not rewritten
    src.write_text(SIMPLE.replace("x = 1", "x = 2"))
    generate(str(src), str(out))
    assert os.stat(out / "a.tex").st_mtime != 1


def test_generate_error_writes_nothing(tmp_path):
    src = tmp_path / "s.py"
    src.write_text(SIMPLE + "# TeX: listing b 1\n# TeX: begin b\nx\n")
    with pytest.raises(ListingError) as info:
        generate(str(src), str(tmp_path / "code"))
    assert str(info.value).startswith(f"{src}:")
    assert not (tmp_path / "code").exists()


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "code_to_tex", *args], cwd=PACKAGE_PARENT, capture_output=True, text=True
    )


def test_cli(tmp_path):
    src = tmp_path / "s.py"
    src.write_text(SIMPLE)
    r = run_cli(str(src), str(tmp_path / "code"))
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "code" / "a.0.txt").read_text() == "x = 1\n"
    src.write_text("# TeX: bogus\n")
    r = run_cli(str(src), str(tmp_path / "code"))
    assert r.returncode == 1
    assert r.stderr.strip() == f"{src}:1: unknown directive 'bogus' (expected listing, begin, end, remark[!], summary[!], filbreak, color, mark)"


def test_nyc25_cpu_fixture_matches_legacy_text():
    out = render_outputs((FIXTURES / "nyc25_cpu.py").read_text(), "nyc25_cpu.py")
    assert sorted(out) == sorted(["cpu.tex"] + [f"cpu.{v}.{ext}" for v in range(5) for ext in ("txt", "dbg")])
    if LEGACY_NYC25.is_dir():
        for v in range(5):
            assert out[f"cpu.{v}.txt"] == (LEGACY_NYC25 / f"cpu.{v}.txt").read_text() + "\n"
    assert out["cpu.4.dbg"] == textwrap.dedent(
        """\
        for ⟦green:m⟧ in ⟦green:seq⟧(0, M):
          for ⟦violet:n⟧ in ⟦violet:seq⟧(0, N):
            accum: f32 @ DRAM
            accum = 0
            for ⟦blue:k⟧ in seq(0, K):
              accum += A[⟦green:m⟧, k] * B[k, ⟦violet:n⟧]
            C[⟦green:m⟧, ⟦violet:n⟧] = accum
        """
    )
    assert out["cpu.1.dbg"].splitlines()[1] == "def nyc25_gemm_cpu(M: ⟦blue:size⟧, N: ⟦blue:size⟧, K: ⟦blue:size⟧,"


def test_nyc25_cpu_box_count_matches_legacy():
    """Same number of colored boxes per version as the legacy .tex output."""
    if not LEGACY_NYC25.is_dir():
        pytest.skip("legacy nyc25_tex output not present")
    out = render_outputs((FIXTURES / "nyc25_cpu.py").read_text(), "nyc25_cpu.py")
    for v in range(5):
        legacy = (LEGACY_NYC25 / f"cpu.{v}.tex").read_text()
        assert out[f"cpu.{v}.dbg"].count("⟦") == len(re.findall(r"\\smash\{", legacy)), v
