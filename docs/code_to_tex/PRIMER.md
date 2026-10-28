# Primer: writing slides (and the paper) in this setup

A walkthrough of the workflow, not of Beamer in general. Terse lookup: [REFERENCE_CARD.md](REFERENCE_CARD.md).
Everything below exists in working form in [`beamer_nyc25_test/`](beamer_nyc25_test/) (in this package); open those files alongside.
Build it with `cd spork/docs && python3 -m code_to_tex.build code_to_tex/beamer_nyc25_test --slides beamer_nyc25_test --png`.

## 0. The mental model

* A **document directory** (e.g. `amspork3/`) holds one paper (`main.tex`), any number of decks, shared figure bodies, and `samples.py`.
  Paper and decks share everything; nothing meaningful is written twice.
* A **figure** is a `sporkfigure` environment (a scaled TikZ picture) written where it is used.
  A figure shown in several places keeps its TikZ in a shared `fig_<name>_body.tex`; it is animated with overlay steps
  on slides and shown once (as the final figure) in the paper. You decide per element what the paper shows.
* A **listing** lives as real code in `samples.py` (which must pass `exocc`), annotated with `# TeX:` directives.
  One listing has several *versions* (different lines shown, different highlights); slides step through them, the paper picks one.
* The **build** (`python3 -m code_to_tex.build`, run from `spork/docs`) runs exocc, generates listings, runs xelatex,
  and copies good PDFs next to the directory for evince.

## 1. Start a deck

Create `<doc>/talk.tex`:

```latex
\documentclass[aspectratio=169]{beamer}
\newif\ifsporkSlides \sporkSlidestrue
\input{copied_styles/spork_style.tex}
\input{copied_styles/code_to_tex_support.tex}
\input{<doc>_defs.tex}            % if the document has one

\title{Exo-GPU}
\author{David Zhao Akeley}
\begin{document}
\begin{frame}[label=title]
\titlepage
\end{frame}
\end{document}
```

Build it: `python3 -m code_to_tex.build <doc> --slides talk` (add `--no-paper` to skip `main.tex`).
Open `spork/docs/<doc>_talk.pdf` in evince; it is only replaced when the build has no LaTeX errors and no undefined references,
so evince never sees a broken PDF. Warnings (overfull boxes etc.) are listed in the summary but never block anything.

## 2. Write a frame

```latex
\begin{frame}[label=smem]{Shared Memory\sporkTitleTag{\sporkInvBlue{GEMM v2}}}
\begin{itemize}
  \item Threads in a block share \sporkKeyA{SMEM}.
  \item \sporkReveal<2->{But SMEM accesses need \sporkKeyB{synchronization}.}
\end{itemize}
\end{frame}
```

* Always give frames a `label=`: it is how you build one frame alone and how `--where` reports locations.
* The title goes in the first braced group only. A second braced group would silently become an invisible subtitle.
* Text is 12pt; you don't need `\LARGE` everywhere like the old decks.
* `\sporkTitleTag{...}` right-aligns a tag in the title; `\sporkExtraSlides` before appendix frames turns titles blue.

## 3. The fast loop: one frame at a time

```bash
python3 -m code_to_tex.build <doc> --slides talk --frame smem --skip-code --png
```

About a second: builds only that frame (all its overlay steps) at the real slide size, and writes
`<doc>/preview/talk_frame-smem-STEP.png`, one per overlay step (STEP from 1). `--skip-code` skips exocc and listing generation when you only touched LaTeX.
Lost track of where something is? `--where fig:dag` prints the frame and pages, plus the `--frame` command to rebuild it.

## 4. Steps and what the paper shows

Every overlay states what the paper does with it:

| You want | Text | In TikZ | Paper |
|---|---|---|---|
| appear on step 2, stay, and be part of the final result | `\sporkReveal<2->{..}` | `spork reveal=<2->` | shown |
| only on step 3 (emphasis, a remark, a caption for that step) | `\sporkSlideOnly<3>{..}` | `spork slide only=<3>` / `spork slide only=<3>{red style...}` | dropped |
| only in the paper (a legend the speaker would say aloud) | `\sporkPaperOnly{..}` | `spork paper only` / `spork paper style={dashed}` | shown |

Things with no spec appear everywhere. `\sporkReveal` keeps its space before it appears (nothing jumps);
`\sporkSlideOnly` takes no space when hidden, which is what you want for per-step captions:

```latex
\sporkSlideOnly<1>{Consume each SMEM tile one iteration after it is loaded.}%
\sporkSlideOnly<2>{One slot may be being filled while another is consumed.}
```

(The `%` stops the line break from adding a space.) If the captions differ in height and the content below them jumps,
put them in `\begin{overlayarea}{\linewidth}{2\baselineskip}...\end{overlayarea}`.
Don't use Beamer's own `\only<..>`, `\pause`, `\node<..>`: the build warns, because their paper behavior is implicit.

## 5. A figure

A figure is a `sporkfigure`: a `tikzpicture` with a fixed 11pt sans font (the paper body size), so it looks the same
in the paper and on slides. `scale=` (default 1) magnifies it as a whole; use it only for a real layout reason.
Write it where it is used; a figure used in one place is just that:

```latex
\begin{figure}
  \centering
  \begin{sporkfigure}[node distance=5mm]       % any tikzpicture options
    \node (A) [spork green style] {Load A};
    \node (C) [draw, right=of A] {compute};
    \draw [spork arrow] (A) -- (C);
  \end{sporkfigure}
  \caption{...}\label{fig:small}
\end{figure}
```

**Shared figures.** When the paper and slides (or several slides) show the same figure, put the TikZ commands in
`<doc>/fig_dag_body.tex` (no preamble, no `tikzpicture`; see [beamer_nyc25_test/fig_dag_body.tex](beamer_nyc25_test/fig_dag_body.tex))
and `\input` it inside a `sporkfigure` at each place:

```latex
% fig_dag_body.tex
\tikzset{dag compute/.style={draw, minimum width=15mm}}   % local to the figure
\node (A1) [spork green style] {Load A};
\node (C2) [dag compute, right=of A1, spork slide only=<3>{spork red style}] {compute};    % red on step 3 only
\draw [spork arrow] (A1) -- (C2);
\draw [spork arrow, spork reveal=<2->, spork paper style={dashed}] (C2) to[bend left] (A1);  % WAR edge
\node [spork paper only, below=of C2] {dashed: WAR};                                     % legend, paper only

% main.tex
\begin{figure}
  \centering
  \begin{sporkfigure}\input{fig_dag_body.tex}\end{sporkfigure}
  \caption{...}\label{fig:dag}
\end{figure}

% talk.tex, frame `dag`: the same body, same size
\begin{center}
\begin{sporkfigure}\input{fig_dag_body.tex}\end{sporkfigure}
\end{center}

% talk.tex, a frame where it shares the slide with a listing: shrunk to fit a half-width column
\begin{sporkfigure}[scale=0.8]\input{fig_dag_body.tex}\end{sporkfigure}
```

**Per-slide additions.** A slide can add its own elements next to the body's nodes, inside its own `sporkfigure`;
the paper and the other slides don't get them. The body's node names (`A1`, `C2`, ...) are its interface:

```latex
\begin{frame}<3>[label=dag_war]{WAR Hazard Close-up}   % <3>: show only the body's step-3 state
\begin{sporkfigure}
  \input{fig_dag_body.tex}
  \node [spork red style, draw, above=2mm of A1] {only on this slide};
\end{sporkfigure}
\end{frame}
```

(The example deck's frame `dag_war` does exactly this.) Step numbers inside a body are shared by every frame that uses it,
so timing that only one frame needs goes in that frame (its own `\sporkReveal`s, or a frame overlay spec like `<3>`).

* Write the body once, at paper size; `scale=` only magnifies it. Inside bodies, size things in mm and use
  `\sporkFigSmall`/`\sporkFigLarge` instead of `\small`/`\large`, and never `\linewidth`: those follow the host document,
  so the paper and slide versions would drift apart (the build lints `fig_*_body.tex`). Bold nodes are just `font=\bfseries`.
* Node names are local to a `sporkfigure`: a reference to a name the figure doesn't define (misspelled, renamed in the body,
  from another figure) is the error `No shape named C2 is known` instead of an arrow silently drawn to some other figure's node.
* The build warns when a `sporkfigure` is wider than the line (`\linewidth`) after scaling, in the paper and on each slide.
* Look at a figure in context: its frame with `--frame dag --png`, or the paper page (`--where fig:dag` tells you which) with `--png`.
* No `remember picture` or listing marks inside a `sporkfigure` (positions in a scaled box are wrong under xelatex);
  arrows between code lines are a plain picture next to the listing (section 7).

## 6. A listing

In `<doc>/samples.py`, around real code:

```python
# TeX: listing cpu 3
# TeX: begin cpu
  # TeX: color cpu[1] blue "size" 3:*
  def gemm(M: size, N: size, K: size, A: f32[M, K] @ DRAM):
    # TeX: color cpu[2] green "m" "seq"
    for m in seq(0, M):
# TeX: end cpu
```

* `listing cpu 3` declares versions 0, 1, 2. Filters say which versions a directive applies to:
  `cpu` (all), `cpu[1]`, `cpu[1:]`, `cpu[:2]`, several joined by commas without spaces (`cpu[0],ring[1]`). Always the first word.
* `color` colors the next code line: a color word, then the quoted text to match. `3:*` means "exactly 3 matches, color all";
  `2:0` "exactly 2, color the first". With no count, exactly one match is required. If anything is ambiguous, the build stops with
  `samples.py:LINE: ...` rather than guessing.
* Matching is whole-word: `"m"` finds the loop variable in `for m in seq(0, M):`, not the `m` inside other words.
  To color part of a word, put `chars` before the text: `green chars "smem"` colors the end of `A_smem`.
* In comments, `` `accum` `` is set in the code font (the backticks disappear) and `$O(n^2)$` is math; `` `$` `` is a literal dollar.
* `begin`/`end` with a filter change which lines a version shows, so code can grow from version to version.
  `remark! cpu[2]` + a comment line adds a bold comment that only that version shows; `elide "long text"` replaces text with `...`.
* Check the result without building TeX: `<doc>/code/cpu.1.dbg` shows version 1 with `⟦blue:size⟧` markers.

In a frame: `\footnotesize\sporkListing{cpu}` steps through the versions (version v on step v+1; the code grows);
`\sporkListing[reserve]{cpu}` keeps space for lines that appear later; `\sporkListing[version=1]{cpu}` shows one version.
In the paper, `\sporkListing[version=1]{cpu}` (default: the last version).
Lines never wrap; a too-long line is reported as an overfull box. Put long listings in `\footnotesize`/`\scriptsize`.

## 7. Arrows between code lines

Add `# TeX: mark split raw_arrive end` before a code line to drop a point named `split-raw_arrive` at its end.
Then, after the listing (which must be in normal text or a minipage, not inside a TikZ node):

```latex
\begin{tikzpicture}[remember picture, overlay]
  \coordinate (a) at (pic cs:split-raw_arrive);
  \coordinate (b) at (pic cs:split-raw_await);
  \sporkReveal<2->{\draw [spork arrow] (a) -- ++(10mm,0) |- (b);}
\end{tikzpicture}
```

Always go through a `\coordinate` first: `(pic cs:split-x -| b)` silently lands at the page origin.
This is a plain picture, not a `sporkfigure` (marks need unscaled positions).
Marks need two LaTeX passes; the build does them. Example: [split_marks.tex](beamer_nyc25_test/split_marks.tex),
an ordinary `\input` snippet used by both the deck and the paper.

## 8. Reading the build summary

* `viewer copy NOT updated: ...`: fix the LaTeX error or undefined reference it names; your evince copy is the last good one.
* `Overfull \hbox` with a file and line: usually a listing line or text too wide for the slide.
* Content running off the bottom of a slide does not always produce a warning: look at the PNG.
* Lint warnings about bare `\only<`, `\pause`, ...: switch to the `\spork...` wrappers so the paper does the right thing.
