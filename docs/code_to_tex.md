# `code_to_tex.py`: turning annotated Python/Exo source into LaTeX code listings

> **Note:** This document was written by a generative AI (Claude, Anthropic) by
> reading `code_to_tex.py`, its callers, and its inputs/outputs in `spork/docs`.
> It has not been fully reviewed by a human; if it disagrees with the script,
> trust the script.

## Cheat sheet (single-version listings, as in spork_b)

* Put the example code in the samples file (e.g. `b_samples.py`). If it
  shouldn't be compiled by `exocc`, put it under `if False:` (it must still
  parse as Python).
* Declare the listing, then wrap the lines to show:

  ```python
  # TeX: version my_fig 1
  # TeX: begin my_fig[0]
  ...code...
  # TeX: end my_fig[0]
  ```

  Rebuilding makes `spork_b/my_fig.0.tex` and `spork_b/my_fig.0.txt`. Include
  the `.tex` in a figure with `\input{spork_b/my_fig.0.tex}` and
  `\label{fig:my_fig}`.
* Directives must be written exactly as `# TeX: ` (with that capitalization
  and one space), with optional indentation before it. They never show up in
  the output. The listing's shared indentation is removed automatically.
* **Color part of a line:** put a `color line` directive right before the line,
  followed by a line of letters aligned by column with the code:

  ```python
      # TeX: color line *
      #   rrrrrr        ggg
      x = foo(a) + bar(b, c)
  ```

  * `r g b y v`: red/green/blue/yellow/violet box around those characters.
  * `.`: replace those characters with `...`.
  * space or `#`: no color. Place the letter line's leading `#` in the
    indentation (or anywhere a space would go) so the letters line up.
  * It applies only to the next source line.
* **Add a comment line that isn't in the real code** (for headings or notes):

  ```python
  # TeX: remark! *
  # Correct version
  ```

  Use `remark!` for bold and `remark` for plain text. The line after the
  directive is the text, and it is written as a comment so the file stays
  valid Python.
* Everything after `#` is shown as a green sans-serif comment. Text inside `$...$` in a
  comment is rendered as LaTeX math, so a literal `$` can't be used in comments.
* Rebuild with `./spork_b_tex.sh` from `spork/docs`.

## Overview

`code_to_tex.py` is a plain text-processing script (it does not import `exo`).
It reads a single `.py` file containing ordinary source code with special
`# TeX: ...` comment directives, and writes out any number of code "listings",
each as a pair of files:

* `<output_dir>/<name>.<version>.tex`: LaTeX formatted code, meant to be
  `\input{...}` inside a `figure` environment.
* `<output_dir>/<name>.<version>.txt`: the same code as plain text (for
  humans/AI assistants who want to read the listing without the LaTeX noise).

Usage:

```bash
python3 code_to_tex.py input.py output_dir
```

The output directory is created if needed. Every listing declared in the input
is (re)generated on every run.

### How spork_b uses it

`spork_b_tex.sh`:

```bash
exocc b_samples.py && python3 code_to_tex.py b_samples.py spork_b && xelatex spork_b.tex </dev/null
```

1. `exocc b_samples.py` compiles the samples. This is only a sanity check that
   the example code is valid Exo; `code_to_tex.py` never looks at `exocc`
   output. (Fragments that are not meant to be compiled, like the deliberately
   broken `why_dist` example, are hidden inside `if False:` blocks; they still
   need to parse as Python.)
2. `code_to_tex.py` writes `spork_b/<name>.0.tex` and `spork_b/<name>.0.txt`
   directly into the `spork_b/` source directory, next to the hand-written
   `.tex` files.
3. Hand-written files pull listings in, e.g. `DistributedMemoryOverview.tex`:

   ```latex
   \begin{figure*}[!h]
   \codehrule
   \input{spork_b/why_dist.0.tex}
   \caption{...}
   \label{fig:why_dist}
   ```

   By the `spork_b/README.md` convention, `fig:foo` for a code listing
   corresponds to `foo.0.txt`/`foo.0.tex`.

Other documents in `spork/docs` use the same script the same way; their build
commands are in the first-line comment of each top-level `.tex` file
(e.g. `spork_gemm.tex`, `nexo.tex`, `exo_intro.tex`, `spork_guide.tex`).

The generated `.tex` depends on these LaTeX macros, which the including
document must define:

| Macro | Defined in | Use |
| --- | --- | --- |
| `\blacktt{...}` | `slides_common.tex` / `whitepaper_common.tex` | wraps each code line |
| `\codecomment{...}` | `colorbox_common.tex` | Python `#` comments (green sans-serif) and `...` elisions |
| `\redBox`, `\greenBox`, `\blueBox`, `\yellowBox`, `\violetBox` | `colorbox_common.tex` | highlighted spans |

## Input format

The input is a normal source file. Lines whose stripped text starts with
exactly `# TeX: ` (capital T, lowercase e, capital X, colon, one space) are
**directives**; every other line is a **source line**. A directive line is
split on whitespace: the third token is the directive name and the rest are
its arguments.

Some directives consume the *next* physical line of the file as their payload
(that line is not treated as a source line, whatever it contains).

If a non-directive line contains `#` and, case-insensitively, `TEX`, the
script prints a `mistyped TeX directive?` warning but otherwise treats it as
source.

### Listings, versions, and filters

A listing is identified by a **name** and a **version number**. Versions let
one region of code produce several related figures (e.g. a GEMM kernel at
successive optimization steps) with lines added/removed/highlighted per step.

```python
# TeX: version NAME COUNT
```

Declares listing `NAME` with versions `0 .. COUNT-1`. If a name is declared
more than once, the max count wins. A name must be declared (earlier in the
file) before any filter mentions it. spork_b only uses `COUNT = 1`, hence the
`.0` in every file name.

Several directives take a list of **filters** (at least one). A line/directive
applies to a given `(name, version)` if **any** filter matches:

| Filter | Matches |
| --- | --- |
| `*` | every listing, every version |
| `NAME` | every version of `NAME` |
| `NAME[i]` | version `i` only |
| `NAME[lo:hi]` | versions `lo <= v < hi` (half-open, like Python) |
| `NAME[lo:]` / `NAME[:hi]` | open-ended ranges |

Filters are separated by whitespace, e.g. `# TeX: end ann[0] ann[1]`.

### Directives

#### `begin` / `end`

```python
# TeX: begin FILTERS...
...source lines...
# TeX: end FILTERS...
```

Each listing version keeps a counter. `begin` increments it if a filter
matches; `end` decrements it (error if it goes negative). Source lines are
included in a listing version whenever its counter is `> 0`. Because it is a
counter, overlapping/nested regions for the same listing work, and different
listings' regions can overlap freely. An unmatched `begin` at end of file is
not an error (the region simply runs to EOF).

Blank source lines inside a region are kept (as empty lines).

#### `color line`

```python
# TeX: color line FILTERS...
#     rrrr   gggg
x = foo(a, b) + bar
```

The next physical line is a string of **color letters**, aligned column-for-
column with the next source line. It applies only to the very next source line
(and only in listings matching the filters). If several `color line`
directives precede the same source line, the last matching one wins (they are
not merged). The pending colors are cleared after any source line, even one not
included in the listing.

Color letters:

| Letter | Effect |
| --- | --- |
| `r` `g` `b` `y` `v` | `\redBox` / `\greenBox` / `\blueBox` / `\yellowBox` / `\violetBox` around the span |
| `.` | the span is **deleted** and replaced with a single `...` (in comment style) |
| space or `#` | no color (default formatting) |
| anything else | error |

Because `#` counts as "no color", the letter line is conventionally written as
a Python comment whose `#` sits somewhere in the leading whitespace so that the
letters line up with the code below, e.g. from `gemms.py`:

```python
                                # TeX: color line *
                               #yyyyyyy             y
                                A_smem[m1*8 + m0, k0] = A[m2*128 + m1*8 + m0, k1*32 + k0]
```

Columns are absolute file columns (before dedent). Letters beyond the end of
the code line are ignored. Colored spans are wrapped in `\smash{}` with small
negative `\hspace` so the boxes don't change line height/spacing.

#### `remark` / `remark!`

```python
# TeX: remark FILTERS...
# Text of the remark, shown as a comment line
```

The next physical line is inserted into the listing as its own line, but only
for matching listings and only inside an active `begin`/`end` region. It is
otherwise invisible (not even treated as source). `remark!` renders it in
bold. Useful for per-version commentary or headings (see `why_dist` in
`b_samples.py`).

`color remark FILTERS...` / `color remark! FILTERS...` takes *two* following
lines: first the color letters, then the remark text.

#### `summary` / `summary!`

```python
# TeX: summary
# Distribute work over threads
for m1 in cuda_threads(...):
    ...
```

Takes no filters. The next physical line is the summary text. Its behavior
depends on whether the listing version is currently inside a region:

* **Inside a region** (counter `> 0`): it's emitted as an ordinary line
  (bold for `summary!`, colored if given `color summary`).
* **Outside** (counter `== 0`): it's emitted as a *collapsed placeholder* for
  omitted code: the first `#` is replaced with `# ...`, the line is italic, and
  color letters are dropped.

This applies to *every* declared listing version, which is how one file can
show a detailed and a zoomed-out view of the same code. Placeholder lines at
the very start or end of a listing are stripped, so only placeholders
*between* included code survive. Note that consecutive placeholders are not
merged.

`color summary` / `color summary!` works like `color remark` (colors line, then
text line).

#### `filbreak`

```python
# TeX: filbreak
```

Takes no filters. Inside an active region, emits a raw LaTeX `\filbreak`
(a hint that allows a page/column break at that point). The line preceding a
`\filbreak` does not get a trailing `\\`.

## Output details

For each `(name, version)`:

1. Walk the directive list as above, collecting lines.
2. Strip leading/trailing summary placeholders.
3. **Dedent**: remove the minimum leading-whitespace count over all collected
   lines (whitespace-only lines are ignored for this computation), so a listing
   taken from deeply indented code starts at column 0.
4. Render.

`.tex` rendering, per line:

* Wrapped in `\blacktt{...}`; lines separated by `\\` + newline.
* Everything from the first `#` onward is treated as a comment and wrapped in
  `\codecomment{...}`. Detection is naive: a `#` inside a string literal also
  starts a "comment".
* Special characters are escaped (`_ # $ % & { } ~ ^ \`), and spaces become `~`
  so indentation survives. Tabs are not handled.
* **Inside comments only**, text between `$...$` is passed through
  unescaped as LaTeX math (so `# $\tau_s$` renders math). Consequently a
  literal `$` cannot appear in a code comment.
* `remark!`/`summary!` lines are wrapped in `\textbf{}`; placeholders in
  `\textit{}`.
* The file begins with `%%` comments telling AI assistants to read the `.txt`
  instead.

`.txt` rendering: the dedented raw text of each collected line, joined by
newlines, without any coloring.

## Quirks observed

These seem unintentional but are harmless in practice:

* The payload line consumed by `remark`/`summary` (and their color line)
  keeps its trailing newline. In the `.tex` this is just whitespace, but in the
  `.txt` each remark/summary is followed by a spurious blank line (visible in
  `spork_b/why_dist.0.txt`).
* `filbreak` lines produce an empty line in the `.txt`.
* `summary` directives cannot be filtered, so a summary affects every listing
  declared anywhere in the file (usually invisible because of the
  leading/trailing placeholder stripping and because other listings rarely
  have a region around it).
* The filter parser rejects `name[i]` only when the name is undeclared; a
  version index out of range simply never matches.
