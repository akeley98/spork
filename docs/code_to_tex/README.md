# `code_to_tex`: code listings for papers and Beamer slides

Quick orientation (layout, build commands, LaTeX skeleton, overlay intent): [REFERENCE_CARD.md](REFERENCE_CARD.md).

The `code_to_tex` package is a self-contained toolchain (Python 3.10, standard library only) for documents whose paper (`main.tex`) and Beamer decks share style, figures and code listings.
This README covers the **listing generator** and its LaTeX side, `\sporkListing`.
The build driver is `python3 -m code_to_tex.build` (`build.py`), and the shared look lives in `latex/spork_style.tex`.

| File | Purpose |
|---|---|
| `listing.py` | API and CLI: `generate`, `render_outputs`, `ListingError` |
| `listing_parse.py` | turns `# TeX: ` directives into items; checks that do not depend on the version |
| `listing_layout.py` | per-version selection, liveness and conflict checks, dedent |
| `listing_render.py` | writes the `.tex`, `.txt` and `.dbg` outputs |
| `latex/code_to_tex_support.tex` | defines `\sporkListing` and the macros that generated files use |
| `tests/test_listing*.py` | pytest suites; the LaTeX ones are skipped when `xelatex` is missing |

## Running

```bash
python3 -m code_to_tex samples.py code/      # run from the directory that contains code_to_tex/
```

From Python: `code_to_tex.listing.generate(src_path, out_dir) -> list[str]` returns the output paths.
It raises `code_to_tex.listing.ListingError`, and `str(e)` has the form `file:line: message`.
The CLI prints that message and exits with status 1.
Warnings go to stderr as `file:line: warning: ...`.

* Nothing is written if the source has an error.
* Outputs whose content did not change are not rewritten, so their mtimes are kept.
* Other files in `out_dir` are never touched, so a renamed listing leaves its old outputs behind.

Run the tests with `cd spork/docs && python3 -m pytest code_to_tex/tests`.

## Source format

Listings are cut out of an ordinary source file (normally `samples.py`, which also has to pass `exocc`).
A line is a **directive** when it starts with exactly `# TeX: `, possibly after indentation.
The prefix is case-sensitive and has one space after the colon.
Directive lines never appear in any output.
The arguments are split like a shell command line (`shlex`): `"quoted text"` is one argument, and backslash escapes work.
Arguments are recognized by **position and shape**:
the filters are always the first word after the directive name (except for `listing NAME COUNT`),
and the remaining arguments are told apart by their shape (quoted string, `N:I` count, bare word).
The only bare words with a meaning of their own are the color names (in `color`), the mark positions and shifts (in `mark`),
and the match mode `chars`, which is reserved directly before a quoted match text (see [Matching](#matching)).

A line that contains `#` together with `#tex`, `# tex` or `tex:` (any case), but is not a valid directive, gets a warning such as `looks like a mistyped directive`.
Examples are `#TeX: begin a`, `# tex: begin a` and code ending in `# TeX: ...`.

### Listings, versions, filters

```python
# TeX: listing cpu 5          # declares listing "cpu" with versions 0..4
```

A listing must be declared before any filter mentions it, and each listing name can be declared only once.
Names use `[A-Za-z_][A-Za-z0-9_]*`. Any such name is allowed, even `red` or `end`, because filters are found by position.

Every directive except `listing`, `summary` and `filbreak` takes **filters** as its first argument: one word,
several filters joined by commas with no spaces (`cpu[:4],reorder_loops[0]`).
A directive applies to (listing, version) if *any* of its filters matches:

| Filter | Matches |
|---|---|
| `*` | every version of every listing |
| `cpu` | every version of `cpu` |
| `cpu[2]` | version 2 |
| `cpu[1:3]` | versions 1 and 2 (half-open, like Python slices) |
| `cpu[2:]`, `cpu[:2]` | open-ended ranges |

Indices must be in range, and a slice must not be empty: `cpu[5]` is an error for a listing with 5 versions.
Space-separated filters (the old syntax, `begin cpu[0] cpu[1]`) are an error with a hint: `filters must be one comma-joined word, without spaces: cpu[0],cpu[1]`.

### Directives

| Directive | Meaning |
|---|---|
| `listing NAME COUNT` | declares a listing |
| `begin FILTERS` / `end FILTERS` | include the lines in between (see below) |
| `remark FILTERS` / `remark! FILTERS` | the next line is an extra comment line, bold with `!` |
| `summary` / `summary!` | the next line is a summary line (see below) |
| `filbreak` | allows a page break here (`\filbreak`) |
| `color FILTERS COLOR [chars] "text" [N:I \| N:*] [[chars] "text" ...] [COLOR ...]` | colors text on the next line |
| `mark FILTERS NAME [[chars] "text" [N:I]] [before\|after\|start\|end] [xshift=DIM] [yshift=DIM]` | puts a TikZ mark on the next line |

Examples: `begin cpu[:4],reorder_loops[0]`, `end cpu[:4]`, `remark! split[2]`, `mark split raw_await end`,
`mark cpu[1],cpu[3] accum "accum" after`, `color cpu[3:] violet "n" "seq" blue "k"`.

**`begin`/`end`.** Each (listing, version) keeps a counter.
A matching `begin` increments it and a matching `end` decrements it.
A line is included while the counter is above 0, so regions nest and may overlap across listings.
These are errors:

* an `end` without a matching `begin`;
* a `begin` that is never closed;
* a listing version that ends up with no lines.

Blank lines inside a region are kept.

**`remark`.** The next physical line is the remark text.
It must be a `#` comment, so the source stays valid Python.
The remark is shown in the versions selected by FILTERS, and only inside a region.
A remark whose named filter selects a version where it is outside every region is an error.

**`summary`** (legacy semantics) has no filters, applies to every listing, and its next line is a `#` comment:

* Inside a region it is an ordinary line, bold with `summary!`.
* Outside a region it becomes an italic *placeholder*: `# text` turns into `# ... text`.

Placeholders at the start or end of a version are dropped, so only placeholders between shown lines remain.
A typical use is collapsing a loop body in a zoomed-out version.

```python
# TeX: begin full
# TeX: summary
# Load A tile using cp.async
for ...:                     # shown in "full"; "overview" shows "# ... Load A tile using cp.async"
# TeX: end full
```

**`color`** colors text on the *next non-directive line*: a source line, or the text of a `remark`/`summary` when the directive comes before that `remark`/`summary`.

```python
# TeX: color cpu[1] blue "k" 2:*        # exactly 2 matches (whole words), color all
# TeX: color cpu[4] green "m" violet "n" # the m of A[m, k] and the n of B[k, n] (not the m of accum)
# TeX: color cpu[2] red chars "cum"     # chars: substring match, here inside the word accum
# TeX: color cpu[3] blue "k" 2:1        # exactly 2 matches, color the second (0-based)
# TeX: color cpu[0] elide "k] * B[k"    # replace the text with "..."
accum += A[m, k] * B[k, n]
```

One directive can apply several colorings that share its filters, which saves vertical space:

```
colorDirective → "# TeX: color" filters coloring+
filters        → filter ("," filter)*        (one word, no spaces)
coloring       → colorName match+             (the color word is sticky until the next color word)
match          → mode? quotedString index?
mode           → "chars"                      (omitted: whole-word matching)
index          → N ":" I  |  N ":*"           (omitted: exactly 1 match)
```

After the filters, each argument is classified by its shape:
a quoted string is text to color in the current color, `N:I`/`N:*` refines the quoted string *directly* before it,
`chars` sets the match mode of the quoted string *directly* after it, and any other bare word must be a color name.
Errors: an unknown bare word (for example unquoted text), a color name with no quoted string after it,
quoted text before any color name, a count that does not directly follow a quoted string,
and `chars` that is not directly followed by a quoted string.
Each quoted string is matched and checked on its own, exactly as if it had its own `color` directive.

* Colors: `red yellow green blue violet invRed invGold invGreen invBlue invPurple` map to `\sporkRed` ... `\sporkInvPurple`.
  `elide` replaces the span with `...` in comment style.
* The text must be quoted. Matches are whole-word by default (see [Matching](#matching)), non-overlapping, left to right, on the raw source line.
* Match counts: without a count, the text must match **exactly once**. `N:I` requires exactly N matches and colors match I. `N:*` requires exactly N matches and colors all of them.
  A wrong count is an error that tells you how many matches there were (and how many more occur inside longer words).
* Any number of `color` directives (and colorings within one directive) may stack on one line:
  * The same character in two different colors (including `elide`) is an error.
  * The same color applied twice does nothing extra.
  * Adjacent or overlapping spans of the same color become **one** box.
* A comment inside a colored box keeps the box colors and switches only to the comment font.

**`mark`** places a TikZ mark (`\pgfmark` from the `tikzmark` library) on the next non-directive line.

```python
# TeX: mark cpu accum_def "accum" after     # just after the (single) match of "accum"
# TeX: mark cpu loop start                  # before the first non-space character
# TeX: mark cpu[2:] call "f(" 2:1 yshift=0pt
```

* The arguments are positional: filters, then the mark NAME, then the optional quoted text (with an optional `chars` directly before it
  and its optional `N:I` directly after it); the position word and `xshift=`/`yshift=` follow in any order.
  A mark may be named `chars` (`mark cpu chars chars "cum"` is a mark named `chars` at a substring match).
  A mark cannot be named `before`/`after`/`start`/`end` (that is reported as a missing mark name).
* With text, the position is `before` (the default) or `after` the match. Match counts work as for `color`, but `N:*` is not allowed.
* Without text, the position is `start` (the default: before the first non-space character) or `end` (after the last character).
* `xshift=` and `yshift=` take TeX lengths (`pt mm cm in ex em bp pc sp dd cc`). They **replace** the defaults for this mark (see Marks below).
* Mark names use `[A-Za-z0-9_]+` and must be unique within each listing version. The same name may sit on different lines in different versions.
* Errors: a mark inside elided text or inside `$...$` math. A mark between backticks in a comment is fine.

### Matching

The quoted text of `color` and `mark` is found in the raw source line, exactly as written in `samples.py` (backticks and `$` included).

* **Whole-word (default).** At each end of the text that is a word character (ASCII `[A-Za-z0-9_]`),
  the neighbouring source character must not be a word character (or the text must be at the line edge).
  An end that is not a word character imposes nothing.
  So `"i"` matches the `i` of `i = 1`, `x[i]` and `` `i` ``, but not the one in `if`, `in`, `i1` or `_i`;
  `"@proc"` only checks the character after the `c`; `"f32[M, K]"` only the one before the `f`;
  `" + "` is a plain substring match. `$` and the backtick are not word characters.
* **`chars`** (written directly before one quoted text, for that text only) matches any substring:
  `color cpu[1] green chars "m" 2:0 "seq"` colors the first `m` of the line even inside a word, and `"seq"` is whole-word again.
  Use it to color part of an identifier (`green chars "smem"` in `A_smem`).
* Counts (`N:I`, `N:*`) count matches of the selected mode. A whole-word count error also reports how many occurrences sit inside longer words.
* The grammar leaves room for more named modes in the same place (`match → mode? quotedString index?`);
  `chars` is the only one so far.

### Anti-ambiguity errors

The generator refuses anything that is probably a mistake, rather than guessing:

* **Dead directives.** If a *named* filter of a `color`/`mark` selects a version that does not show the target line, it is an error.
  This includes a version that only shows the line as a summary placeholder.
  The same rule applies to a `remark` outside every region.
  A `*` filter is lenient: it applies to whichever versions show the line, but it must apply to at least one of them.
* A `color`/`mark`/`remark`/`summary` with no following line (end of file).
* Unknown colors, directives, listings or arguments; unquoted or empty match text; `N:I` out of range or not directly after a quoted string; conflicting colors; duplicate marks.
* Tab characters in shown lines (expand them to spaces).
* An unbalanced `$` in a comment, or a color boundary or mark inside `$...$`.
* An odd number of backticks in a comment, or a color span that contains only one of the two backticks of a pair
  (see [Comments: math and code](#comments-math-and-code)).
  Like tabs, these are only checked on lines that some listing version shows (including remark and summary text, and summary placeholders).
* The legacy syntax (`version`, `color line`) and space-separated filters (`begin a b`, `color a b red "x"`, `mark a b m end`) are rejected with a hint about the new form.

## Outputs (per listing NAME, in OUTDIR)

| File | Content |
|---|---|
| `NAME.tex` | all versions in one file, for `\sporkListing{NAME}`. Starts with a `%%` header telling AI agents to read the `.txt`/`.dbg` instead |
| `NAME.N.txt` | the displayed plain text of version N: dedented, summary placeholders as `# ... text`, elided spans as `...`, no filbreaks |
| `NAME.N.dbg` | the `.txt` text plus `⟦...⟧` annotations (format below) |

**`.dbg` format** is debug output for checking highlights without reading TeX; it is not an interchange format:

| Annotation | Meaning |
|---|---|
| `⟦blue:accum⟧` | colored span |
| `⟦elide:k1*32 + ⟧` | elided span, showing the original text (rendered as `...`) |
| `⟦@name⟧`, `⟦@name xshift=1pt yshift=0pt⟧` | a mark at this position, with any per-mark shifts |
| `⟦remark:…⟧`, `⟦remark!:…⟧` | a whole remark line, plain or bold |
| `⟦summary:…⟧`, `⟦summary!:…⟧` | a summary inside a region |
| `⟦placeholder:# ... text⟧` | a summary outside a region |
| `⟦filbreak⟧` | a filbreak (a line of its own) |

Annotations nest, for example `⟦remark!:# ⟦blue:note⟧⟧`.

**Rendering** follows the legacy script:

* Each line is `\texttt` in black.
* Escaped characters: `_ # $ % & { } ~ ^ \`. Spaces become `~`, so indentation survives.
* Everything from the first `#` on is a comment, set in `\sporkComment` (green sans). Detection is naive: a `#` inside a string also starts a comment.
* Inside comments, `$...$` passes through as LaTeX math and `` `...` `` is set in the code font (see below).
* Colored spans are `\smash`ed so they do not change line spacing, and their `\fboxsep` padding is cancelled horizontally (with `\kern`) so the text does not move.
* Each line is an `\hbox`, so it never wraps: an overlong line sticks out into the margin and LaTeX reports `Overfull \hbox` (the build driver lists these warnings).
  Shorten the line in `samples.py`, elide part of it, or use a smaller font.
* `remark!`/`summary!` lines are bold, and placeholders are italic.
* Each version is dedented by the smallest indentation among its non-blank lines.

### Comments: math and code

In the comment part of a line (from the first `#`, detected naively as above), and in remark and summary text:

* `$...$` is LaTeX math, passed through unescaped.
* Backticks pair left to right. The text between a pair is set in `\texttt` (the listing's code font) in the surrounding color:
  the comment color in plain comment text, the box colors inside a colored box. The backticks themselves are not rendered.
  Example: ``# accumulate into `accum` `` renders as "# accumulate into" in the comment font followed by `accum` in the code font.
* Whichever delimiter opens first wins: inside backticks `$` is a literal character (`` `$` `` is how to write a literal `$` in a comment,
  `` `$5` `` renders `$5` in the code font), and inside `$...$` a backtick is a math character.
* In code (before the `#`) backticks and `$` are ordinary, escaped characters.
* Errors (only on lines that are shown): an unbalanced `$`, or an odd number of backticks outside math.
* Colors and marks match the raw text, delimiters included:
  * a color span may cover a whole `` `...` `` region (both backticks) or a whole `$...$` region (both dollars);
  * a color span may start or end *between* backticks (the monospace text is split around the box), but not inside `$...$`;
  * a color span that contains exactly one backtick of a pair is an error;
  * a mark may sit between backticks, but not inside `$...$`.
* The `.txt` and `.dbg` outputs show the source text, backticks and dollars included.

## LaTeX: `\sporkListing`

`copied_styles/code_to_tex_support.tex` is loaded by the main file after `spork_style.tex`.
`\sporkListing[options]{NAME}` reads `code/NAME.tex`, with paths relative to the document directory.
Inputting `code/NAME.tex` directly is a LaTeX error.

| Option | Slides (`\ifsporkSlides` true) | Paper |
|---|---|---|
| (none) | version v on overlay step v+1; the last version stays on all later steps | the **last** version |
| `version=N` | version N only, on every step | version N |
| `offset=K` (K ≥ 0) | version v on step v+1+K; steps 1..K show no code | (no effect) |
| `reserve` | rows missing from the current version leave a blank line (`\visible`-like) instead of disappearing (`\only`, code grows) | the same, for the chosen version |
| `prefix=P` | mark names become `P-MARK` (default `NAME-MARK`); use when a listing appears twice in one document | same |

Notes:

* Slides use top-aligned frames (`\begin{frame}[t]`) when a listing grows. With Beamer's default vertical centering, every line moves when lines are added.
* Each code line is its own paragraph holding one unbreakable `\hbox` (`\parskip` is 0, so spacing equals `\baselineskip`). Page breaks may fall between lines, and `filbreak` works.
  So `\sporkListing` needs vertical-mode context: running text, a frame, a `minipage`, or a TikZ node with `text width`.
* Font size comes from the surrounding text, for example `{\small\sporkListing{cpu}}`.
* Beamer pitfall: in `\begin{frame}{Title}{\small\sporkListing{cpu}}`, Beamer reads the braced group as the frame *subtitle*, which the spork frame title template does not show.
  Write `\begin{frame}[t]{Title}\small\sporkListing{cpu}` instead (the frame body is already a group).
* An out-of-range `version=N` is a LaTeX error.

Generated-file macros (internal; they appear only in `code/NAME.tex`):

* `\sporkListingBegin{NAME}{COUNT}` and `\sporkListingEnd`.
* `\sporkListingRow{versions}{variants}`: one source row, shown in the listed versions.
* `\sporkListingVariant{versions}{tex}`: the row's rendering for those versions. Rendering differs when coloring or dedent differs.
* `\sporkListingFilbreak{versions}`, `\sporkListingBox{\sporkBlue}{text}` and `\sporkListingMark{name}{xshift}{yshift}`.

Version lists are comma lists such as `0,2,3`.
On slides they become overlay specs (`1,3,4-`); on paper they are tested against the chosen version.

### Marks

```latex
\sporkListing{cpu}
\begin{tikzpicture}[remember picture, overlay]
  \draw[spork arrow] (pic cs:cpu-accum_def) -- ++(2cm, 0);
\end{tikzpicture}
```

* A mark sits at the character boundary, shifted by `\sporkMarkXShift` (default `0pt`) and raised by `\sporkMarkYShift` (default `0.5ex`, the vertical middle of a lowercase letter).
  Change the defaults with `\renewcommand\sporkMarkYShift{0pt}`, or override them per mark with `xshift=`/`yshift=` in the directive.
  The defaults were calibrated with Ubuntu Mono by drawing dots at marks before/after characters, at line start/end, inside boxes and comments, and at `\Large`.
* `ex`/`em` units follow the current code font size.
* Marks need **two LaTeX passes**; the build driver reruns automatically.
* On slides, the `tikzmark` library adds the slide number to mark names behind the scenes, so `pic cs:cpu-m` on each overlay step refers to that step's position, even when the code grows above the mark.
* **Known limitation (xelatex):** marks inside the *text of a TikZ node* are recorded at the wrong position.
  xdvipdfmx places node text with a PDF transformation that TeX's position recording does not see; this was the cause of the old `\bugYFix` hack.
  Put the listing in normal text (or a `minipage`) and draw the arrows in a separate `remember picture, overlay` picture, as shown above.
  For the same reason (positions inside a scaled box), neither the listing nor the arrows go inside a `sporkfigure`;
  [beamer_nyc25_test/split_marks.tex](beamer_nyc25_test/split_marks.tex) is a plain `\input` snippet.

## Tips for frames and figures

* **Figures are `sporkfigure` environments** (`latex/spork_style.tex` section 9; walkthrough in [PRIMER.md](PRIMER.md) section 5).
  `\begin{sporkfigure}[scale=S, <tikzpicture options>]` is a `tikzpicture` with the fixed sans figure font `\sporkFigFont`,
  typeset into a box and scaled as a whole by `S` (default `\sporkFigScale`, 1 in both modes for now).
  A figure used in several places keeps its TikZ in `fig_<name>_body.tex` (no preamble), `\input` inside a `sporkfigure`
  at each place, and a slide can add elements placed relative to the body's nodes. There is no standalone figure build:
  look at figures in context (`--frame LABEL --png`, or the paper page `--where fig:x` names, with `--png`).
  * **Node names are local.** Stock TikZ keeps every node name for the rest of the document, so a picture referring to
    a name it doesn't define silently uses whatever picture defined it last (and `name prefix` silently falls back
    to the unprefixed name). In a `sporkfigure`, (1) every name it creates (nodes, coordinates, `alias=`, pic and matrix
    nodes, nested pictures) is forgotten at `\end{sporkfigure}`, so any later reference is pgf's
    ``No shape named `C2' is known``; and (2) a reference to a node defined in another picture (for example an earlier plain
    `tikzpicture`) is the error ``No shape named `C2' is known in this sporkfigure (that name belongs to another picture)``.
    So two figures may use the same names freely, and a renamed node in a shared body fails loudly in every figure that
    still uses the old name. Plain tikzpictures (listing-mark overlays, inline `\tikz`) keep stock behavior.
    This relies on PGF internals (pgf 3.1.x; listed in the style's comments): the per-node macros
    `\pgf@sh@{ns,np,nt,pi,ma,sa}@NAME`, the `\pgf@nodecallback` hook, and `\pgf@shape@interpictureshift`.
  * **Width warning.** A `sporkfigure` wider than `\linewidth` after scaling (paper column, frame, Beamer column)
    gives `Package spork Warning: sporkfigure is <w> wide after scaling, > \linewidth <lw>` in the build summary.
  * `remember picture` on a `sporkfigure` warns: recorded positions are wrong inside a scaled box under xelatex,
    so listing marks and decorations are drawn in plain pictures outside it (see [Marks](#marks)).

* **Text whose height changes between steps** (for example an `\only<2>{...}` caption with one line on step 1 and three on step 2)
  makes everything below it jump. Wrap it in Beamer's `overlayarea`, which reserves a fixed height:

  ```latex
  \begin{overlayarea}{\linewidth}{3\baselineskip}
    \only<1>{One line.}\only<2>{Three\\ lines\\ here.}
  \end{overlayarea}
  ```

* **Projections of marks.** `(pic cs:split-raw_await -| X)` does not work: everything after `pic cs:` (including ` -| X`) is read as the mark name,
  which is unknown, so the point silently lands at the picture origin (no error or warning). Name the mark position first, then project it:

  ```latex
  \begin{tikzpicture}[remember picture, overlay]
    \coordinate (await) at (pic cs:split-raw_await);
    \coordinate (arrive) at (pic cs:split-raw_arrive);
    \draw[spork arrow] (arrive) -- (arrive -| await);
  \end{tikzpicture}
  ```

## Differences from the legacy `legacy_code_to_tex.py`

* `version` is renamed `listing`. A single `NAME.tex` holds all versions, replacing `NAME.N.tex` per version.
* Column-letter `color line` directives are replaced by matching text, with the new color names; one `color` directive can apply several colorings.
* Filters are always one comma-joined word in first position (legacy: space-separated).
* New: `mark`, the `.dbg` output, dead-directive and conflict errors, duplicate-declaration errors, and errors for an unclosed `begin` or an empty version.
* Fixed quirks:
  * There are no spurious blank lines after remarks or summaries in the `.txt`.
  * Filbreaks are not written to the `.txt`.
  * Remark and summary text must be a `#` comment line.
  * A filter index out of range is an error.
