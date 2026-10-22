# code_to_tex reference card

One page to get oriented. Details, in increasing depth:

* Listing directives, outputs, `\sporkListing`, marks: [README.md](README.md)
* Every style macro / color / TikZ key: header comments + sections of [latex/spork_style.tex](latex/spork_style.tex)
* Build driver: `python3 -m code_to_tex.build --help` (run from `spork/docs`)
* Design rationale and known pitfalls: `endgame_plan/plan_slides.md` ("Implementation notes")
* Worked example of everything: [`spork/docs/beamer_nyc25_test/`](../beamer_nyc25_test/)

## Directory layout

```
spork/docs/
  code_to_tex/                  # this package (python -m code_to_tex, python -m code_to_tex.build)
    latex/spork_style.tex       # master style: the only source of truth for look
    latex/code_to_tex_support.tex
  <doc>.pdf, <doc>_<deck>.pdf   # viewer copies (open these in evince), gitignored
  <doc>/                        # one document = one directory = cwd of every build step
    tex_build.json              # {"check": ["exocc","samples.py"], "listing_sources": ["samples.py"]}
    main.tex                    # the paper (fixed name)
    <deck>.tex                  # any number of Beamer decks (--slides <deck>)
    <doc>_defs.tex              # document-specific packages + \newcommands
    <Name>.tex                  # sections and figures; \label{sec:Name} / \label{fig:Name}
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
python3 -m code_to_tex.build <doc> --figure fig_dag [--paper] --png  # one figure alone (slides / paper rendering)
python3 -m code_to_tex.build <doc> --slides <deck> --where fig:dag   # which page/frame shows a label
```

Order: copy styles -> `check` (exocc) -> listing generator -> xelatex (reruns until stable).
Warnings are summarized at the end and never block the build; previews land in `<doc>/preview/`.

## Every main file, deck, and figure starts like this

```latex
\documentclass{article}               % paper | [aspectratio=169]{beamer} deck | [beamer,aspectratio=169]{standalone} figure
\newif\ifsporkSlides \sporkSlidesfalse % paper | \sporkSlidestrue for deck and figure
\input{copied_styles/spork_style.tex}
\input{copied_styles/code_to_tex_support.tex}
\input{<doc>_defs.tex}                 % optional
```

A figure is `\documentclass[beamer]{standalone}` + `\begin{standaloneframe}...\end{standaloneframe}`,
compiles alone (`--figure`), and is `\input{fig_x.tex}` by the paper and decks.
Example: [beamer_nyc25_test/fig_dag.tex](../beamer_nyc25_test/fig_dag.tex).

## Overlays: every overlay states its paper intent

| Intent | Text | TikZ key | Paper |
|---|---|---|---|
| revealed, part of final figure | `\sporkReveal<3->{..}` | `spork reveal=<3->` | shown |
| one-off emphasis / remark | `\sporkSlideOnly<3>{..}` | `spork slide only=<3>`, `spork slide only=<3>{keys}` | hidden / not applied |
| paper only | `\sporkPaperOnly{..}` | `spork paper only`, `spork paper style={keys}` | shown / applied |

Elements without a spec appear in both. Bare `\only<..>`, `\node<..>`, `\pause`, ... get a lint warning.
Frames: `\begin{frame}[label=dag]{Title\sporkTitleTag{\sporkInvBlue{GEMM v1}}}`; `\sporkExtraSlides` for the appendix.
Pitfall: `\begin{frame}{Title}{...}` makes the braced group an (invisible) subtitle.

## Code listings

In `samples.py` (currently the filter list ends at the first color name; this syntax may change, see the plan's open questions):

```python
# TeX: listing cpu 5                      # versions 0..4
# TeX: begin cpu[:4]                      # include lines in versions 0..3 (half-open, like Python)
# TeX: color cpu[1] blue "size" 3:*       # next line: exactly 3 matches, color all
# TeX: color cpu[3:] violet "n" 2:0       # exactly 2 matches, color the first
# TeX: color ring[0] elide "k1 * K0 + i1" # replace with ...
# TeX: mark split raw_await end           # \pgfmark at end of next line: pic cs:split-raw_await
# TeX: remark! split[2]                   # next line is an extra (bold) comment line
# WAR: slot k1 % 3 is free once its old readers arrived
# TeX: end cpu[:4]
```

Colors: `red yellow green blue violet invRed invGold invGreen invBlue invPurple elide`.
Ambiguity is an error: wrong match count, conflicting colors on a character, directives that never apply.
Check what you highlighted without reading TeX: `<doc>/code/NAME.N.dbg` (`⟦blue:size⟧`).

In LaTeX: `\sporkListing{cpu}` (slides: version v on step v+1, code grows), `\sporkListing[reserve]{..}` (space kept),
`\sporkListing[version=2]{cpu}` (paper default: last version), `offset=K`, `prefix=P`.
Marks only work when the listing is in normal text or a minipage, not inside a TikZ node;
draw arrows in a separate `\begin{tikzpicture}[remember picture, overlay]`.
Example: [beamer_nyc25_test/fig_split_marks.tex](../beamer_nyc25_test/fig_split_marks.tex).

## Style macros (all `spork`-prefixed)

`\sporkKeyA \sporkKeyB \sporkMainKey \sporkMinorKey` keywords; `\sporkRed ... \sporkViolet`, `\sporkInvRed ... \sporkInvPurple` highlights;
`\sporkComment \sporkCodeRule`; `\sporkSection{Title}{sec:X}`; `\sporkPaperLink{label}{text}`;
TikZ `spork arrow`, `spork line`, `spork normal node`, `spork red style`, `spork inv red style`, `spork red node`, ...
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
# generated --figure --paper wrappers
*_paperpreview.tex
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
# build outputs (main.pdf, decks, --frame jobs, standalone figures)
*.pdf
```
