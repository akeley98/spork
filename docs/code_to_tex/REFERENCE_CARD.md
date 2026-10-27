# code_to_tex reference card

One page to get oriented. Details, in increasing depth:

* Guided walkthrough (writing a deck, figure, and listing in this setup): [PRIMER.md](PRIMER.md)
* Listing directives, outputs, `\sporkListing`, marks: [README.md](README.md)
* Every style macro / color / TikZ key: header comments + sections of [latex/spork_style.tex](latex/spork_style.tex)
* Build driver: `python3 -m code_to_tex.build --help` (run from `spork/docs`)
* Design rationale and known pitfalls: `endgame_plan/plan_slides.md` ("Implementation notes")
* Worked example of everything: [`beamer_nyc25_test/`](beamer_nyc25_test/) in this package
  (build: `cd spork/docs && python3 -m code_to_tex.build code_to_tex/beamer_nyc25_test --slides beamer_nyc25_test`)

## Directory layout

```
spork/docs/
  code_to_tex/                  # this package (python -m code_to_tex, python -m code_to_tex.build)
    latex/spork_style.tex       # master style: the only source of truth for look
    latex/code_to_tex_support.tex
    beamer_nyc25_test/          # the worked example: a <doc>/ directory like the one below
  <doc>.pdf, <doc>_<deck>.pdf   # viewer copies (open these in evince), gitignored; deck named <doc> -> <doc>_slides.pdf,
                                #   deck <doc>_x -> <doc>_x.pdf; only written when there are no LaTeX errors and no undefined refs
  <doc>/                        # one document = one directory = cwd of every build step
    tex_build.json              # {"check": ["exocc","samples.py"], "listing_sources": ["samples.py"]}
    main.tex                    # the paper (fixed name)
    <deck>.tex                  # any number of Beamer decks (--slides <deck>)
    <doc>_defs.tex              # document-specific packages + \newcommands
    <Name>.tex                  # sections; \label{sec:Name} / \label{fig:Name}
    fig_<name>_body.tex         # optional shared figure bodies: raw TikZ, \input inside a sporkfigure (see Figures)
    samples.py                  # listing source; must pass exocc
    code/  copied_styles/  samples/  preview/    # generated, gitignored
```

## Building

```bash
cd spork/docs
python3 -m code_to_tex.build <doc>                                   # paper
python3 -m code_to_tex.build <doc> --slides <deck>                   # paper + deck
python3 -m code_to_tex.build <doc> --slides <deck> --no-paper
python3 -m code_to_tex.build <doc> --slides <deck> --frame dag --skip-code --png   # one frame, ~1s
python3 -m code_to_tex.build <doc> --png                             # paper pages as PNGs
python3 -m code_to_tex.build <doc> --slides <deck> --where fig:dag   # which page/frame shows a label
```

Order: copy styles -> `check` (exocc) -> listing generator -> xelatex (reruns until stable).
Warnings are summarized at the end and never block the build. The summary names each job's full LaTeX log
(`<doc>/<job>.log`) and points into it as `[main.log:914]`; errors also show TeX's own context excerpt.
Previews land in `<doc>/preview/`; figures are only seen in context (a frame's PNGs, or the paper page found with `--where`).
The viewer copy is the exception: LaTeX errors or undefined references block it, since evince crashes on such PDFs.

## Every main file and deck starts like this

```latex
\documentclass{article}               % paper | [aspectratio=169]{beamer} deck
\newif\ifsporkSlides \sporkSlidesfalse % paper | \sporkSlidestrue for a deck
\input{copied_styles/spork_style.tex}
\input{copied_styles/code_to_tex_support.tex}
\input{<doc>_defs.tex}                 % optional
```

## Figures

Draw TikZ figures in a `sporkfigure` environment, written wherever the figure is used (a figure and its caption can live in one file).
In cases where figures are shared across the paper and slides, or between multiple slides,
create a `sporkfigure` environment in the document and `\input` shared TikZ commands from `fig_<name>_body.tex`
(no preamble, not compilable alone); a slide can add its own elements next to the body's nodes:

```latex
% fig_dag_body.tex: the shared figure; its node names (A1, B4, C2, ...) are its interface
\node (C2) [dag compute] {compute};
...

% paper
\begin{figure}
  \begin{sporkfigure}
    \input{fig_dag_body.tex}
  \end{sporkfigure}
  \caption{...}\label{fig:dag}
\end{figure}

% one slide: the same body plus this slide's additions, placed relative to the body's nodes
\begin{sporkfigure}
  \input{fig_dag_body.tex}
  \node [spork red style, draw, below=2mm of C2] {only on this slide};
\end{sporkfigure}
```

* `sporkfigure[scale=S, <tikzpicture options>]` is a `tikzpicture` scaled as a whole by `S` (default 1 in paper and slides).
  Text inside is sans at 11pt (`\sporkFigFont`, the paper body size; slides are 12pt), so a body has the same geometry
  in paper and slides and needs no compensating scale. Use `scale` only for a real layout reason (e.g. fitting a column).
  (David Zhao Akeley: unless instructed otherwise, please don't try to "compensate" for the 11/12 difference).
  Node `font=\bfseries` / `font=\sporkFigLarge` add to that font.
* Node names exist only inside their `sporkfigure`: referring to a name the figure doesn't define
  (renamed, misspelled, from another figure) is an error (`No shape named C2 is known`), never a silent wrong arrow.
* The build warns when a `sporkfigure`, after scaling, is wider than `\linewidth` (paper and each slide checked separately).
* Overlay keys in a body (`spork reveal=<2->`, ...) animate on slides; the paper shows the final figure.
  Variations needed by one slide only go in that slide's `sporkfigure`, not in the body.
  Step numbers in a body are shared by every frame that uses it; frame-specific timing belongs in the frame.
* No `remember picture` / listing marks inside a `sporkfigure` (scaling breaks their positions under xelatex).

**In figure bodies, don't use** `\small`/`\large`/... or `\linewidth`/`\textwidth`/`\columnwidth`: they follow the host document,
so paper and slides would differ. Use `\sporkFigSmall`/`\sporkFigLarge`, `\fontsize`, and mm (the build lints `fig_*_body.tex`).
Example: [beamer_nyc25_test/fig_dag_body.tex](beamer_nyc25_test/fig_dag_body.tex).

## Overlays: every overlay states its paper intent

`\sporkReveal`, `\sporkSlideOnly`, and `\sporkPaperOnly` wrap content written inline: they decide on which overlay steps,
and in which output (slides or paper), their braced argument is visible.
`\sporkListing{NAME}` is the only input-like command: it reads the generated `code/NAME.tex` (from `samples.py`)
and picks which version appears on each step, or in the paper.

| Intent | Text | TikZ key | Paper |
|---|---|---|---|
| revealed, part of final figure | `\sporkReveal<3->{..}` | `spork reveal=<3->` | shown |
| one-off emphasis / remark | `\sporkSlideOnly<3>{..}` | `spork slide only=<3>`, `spork slide only=<3>{keys}` | hidden / not applied |
| paper only | `\sporkPaperOnly{..}` | `spork paper only`, `spork paper style={keys}` | shown / applied |

Elements without a spec appear in both. Bare `\only<..>`, `\node<..>`, `\pause`, ... get a lint warning.
Frames: `\begin{frame}[label=dag]{Title\sporkTitleTag{\sporkInvBlue{GEMM v1}}}`; `\sporkExtraSlides` for the appendix.
Pitfall: `\begin{frame}{Title}{...}` makes the braced group an (invisible) subtitle.

### Differences from stock Beamer (surprising even if you know Beamer)

1. Stock `beamerarticle` ignores overlay specs in the paper (`\only<2>{X}` shows X). Here every overlay goes through
   a wrapper that states its paper intent; bare specs are linted.
2. Visibility and space are paired: `\sporkReveal` keeps space (like `\uncover`) and stays in the paper;
   `\sporkSlideOnly` takes no space (like `\only`) and is dropped from the paper.
3. TikZ keys hide by drawing at opacity 0: layout and anchors stay put (stock `\node<2->` removes the node and can
   shift the picture), but hidden text remains in the PDF text layer.
4. `\sporkListing` generates its own steps: version v on step v+1(+offset); the last version persists on later steps.
5. Frames are top-aligned by default (stock: centered); body is 12pt (stock: 11pt); the title template shows no subtitle.
6. Figures are raw TikZ bodies drawn inside `sporkfigure`, whose node names are local to it (stock TikZ keeps
   every node name for the rest of the document).
7. Single-frame builds use `\sporkOnlyFrame` (set by `--frame`); don't edit `\includeonlyframes`.
8. Listing marks are `\pgfmark` (not `\tikzmark`) and don't work inside TikZ node text under xelatex.
9. `\textrm`, `\textsf`, `\texttt`, ... are overlay-aware even in the paper, so `x_\textrm{lo}` fails:
   write `x_\mathrm{lo}` / `x_\mathsf{pre}` / `x_\mathtt{k1}`.

## Code listings

In `samples.py`. Filters are always the first word after the directive, comma-joined without spaces (`cpu[:4],ring[0]`);
the other arguments are told apart by shape (quoted text, `N:I`, bare word):

```python
# TeX: listing cpu 5                      # versions 0..4
# TeX: begin cpu[:4],ring[0]              # include lines in cpu versions 0..3 and ring version 0 (half-open, like Python)
# TeX: color cpu[1] blue "size" 3:*       # next line: exactly 3 matches, color all
# TeX: color cpu[3:] violet "n" "seq" green "m"   # several colorings; a color word applies until the next one
# TeX: color cpu[2] green chars "smem" "k" 2:1  # chars: substring match (inside A_smem); "k" is whole-word again
# TeX: color ring[0] elide "k1 * K0 + i1" # replace with ...
# TeX: mark split raw_await end           # \pgfmark at end of next line: pic cs:split-raw_await
# TeX: mark cpu[1],cpu[3] acc "accum" after
# TeX: remark! split[2]                   # next line is an extra (bold) comment line
# WAR: slot k1 % 3 is free once its old readers arrived
# TeX: end cpu[:4],ring[0]
```

Matching is whole-word: an end of the quoted text that is a letter, digit or `_` must not touch another one in the source
(`"i"` skips `if`, `in`, `i1`; `"@proc"` only checks after the `c`). `chars` directly before one quoted text matches substrings.
`N:I` / `N:*` goes directly after its quoted text (no count: exactly one match) and counts matches of that mode.
In code comments, `$...$` is LaTeX math and `` `...` `` is monospace code; `` `$` `` gives a literal `$`; outside `` `...` ``, 2+ backticks in a row pass to TeX as is (` ``quotes'' `); unbalanced `$` or an odd number of backticks is an error.
Colors: `red yellow green blue violet invRed invGold invGreen invBlue invPurple elide`.
Ambiguity is an error: wrong match count, conflicting colors on a character, directives that never apply, unknown words,
a color span holding only one backtick of a `` `...` `` pair or ending inside `$...$`,
space-separated filters (the error message shows the comma-joined form).
Listing lines never wrap: overlong lines give `Overfull \hbox` warnings.
Check what you highlighted without reading TeX: `<doc>/code/NAME.N.dbg` (`⟦blue:size⟧`).

In LaTeX: `\sporkListing{cpu}` (slides: version v on step v+1, code grows), `\sporkListing[reserve]{..}` (space kept),
`\sporkListing[version=2]{cpu}` (paper default: last version), `offset=K`, `prefix=P`.
Marks only work when the listing is in normal text or a minipage, not inside a TikZ node or a `sporkfigure`;
draw arrows in a separate `\begin{tikzpicture}[remember picture, overlay]`.
Project marks via a named coordinate: `\coordinate (a) at (pic cs:split-x);` then `(a -| b)`;
`(pic cs:split-x -| b)` silently lands at the origin.
Step-varying text of different heights makes content below it jump; wrap it in `\begin{overlayarea}{\linewidth}{H}`.
Example: [beamer_nyc25_test/split_marks.tex](beamer_nyc25_test/split_marks.tex) (a plain `\input` snippet, not a figure body).

## Style macros (all `spork`-prefixed)

`\sporkKeyA \sporkKeyB \sporkMainKey \sporkMinorKey` keywords; `\sporkRed ... \sporkViolet`, `\sporkInvRed ... \sporkInvPurple` highlights;
keywords and highlights take `[only=<spec>]` (formatted on those slides only, like `\only<spec>{\sporkRed}{x}`) and `[only=<spec>, paper]` (also formatted in the paper);
`\sporkComment`, `\sporkCodeRule` (`\linewidth` wide, works in Beamer columns); `\sporkSection{Title}{sec:X}`; `\sporkPaperLink{label}{text}`;
environment `sporkfigure`; TikZ `spork arrow`, `spork line`, `spork normal node`, `spork red style`, `spork inv red style`, `spork red node`, ...
Colors `spork<Name>Fg/Bg`, `sporkKeyColorA/B`, `sporkMainColor`, `sporkMinorColor`, `sporkTitleColor`.

## .gitignore (per document dir)

```gitignore
# exocc output for samples.py
samples/
# listing generator output: NAME.tex (all versions), NAME.N.txt (plain), NAME.N.dbg (formatting shown as ⟦...⟧)
code/
# read-only copies of code_to_tex/latex/*.tex; edit the originals
copied_styles/
# --png renders for agents
preview/
# LaTeX auxiliary files (tikzmark positions live in .aux)
*.aux
*.log
*.out
*.fls
*.nav
*.snm
*.toc
*.vrb
*.bbl
*.blg
# build outputs (main.pdf, decks, --frame jobs)
*.pdf
```
