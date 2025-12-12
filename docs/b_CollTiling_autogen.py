f = open("b_CollTiling_autogen.tex", "w")

node_name = lambda c0, c1, c2: f"t_{c0}_{c1}_{c2}"

for c1 in (1, 0):
    for c0 in (0, 1, 2, 3):
        for wg in (0, 1, 2):
            for t in (0, 1, 127):
                tid = (c0 * 2 + c1) * 384 + wg * 128 + t
                c2 = wg * 128 + t
                nm = node_name(c0, c1, c2)
                fill = "lightgray" if c1 == 1 else "white"

                if c2 != 0:
                    _at = f"{prev_nm}.north west"
                    xs = 14.0
                    if t == 127:
                        xs += 3.6
                    ys = 0.0
                elif c0 != 0:
                    _at = f"{c0_prev_nm}.north west"
                    xs = 0
                    ys = -24.0
                elif c1 != 1:
                    _at = f"{c1_prev_nm}.north west"
                    xs = 12.0
                    ys = 42.0
                else:
                    _at = "0, 0"
                    xs = 0.0
                    ys = 0.0
                print(
                    r"\node(%s) [CollTilingExampleStyle, fill=%s, anchor=north west, xshift=%fmm, yshift=%fmm] at(%s) {%i\\$c_0$=%i\\$c_1$=%i\\$c_2$=%i};"
                    % (nm, fill, xs, ys, _at, tid, c0, c1, c2),
                    file=f
                )
                if t == 127:
                    print(r"\draw[dotted, thick] (%s.east) -- (%s.west);" % (prev_nm, nm), file=f)
                prev_nm = nm
        ctarank = c1 + c0 * 2
        print(
            r"\node(cta%i) [yellowstyle, anchor=center] at(t_%i_%i_0.north west) {%i$\rightarrow$};"
            % (ctarank, c0, c1, ctarank),
            file=f
        )
        c0_prev_nm = node_name(c0, c1, 0)
    c1_prev_nm = node_name(0, c1, 0)

tick = 7.5
ys = -7.8
print(
    r"\draw[thick, greenstyle] ($(t_0_1_0.north west) + (%fmm, %fmm)$) -- ($(t_0_1_0.north west) + (%fmm, %fmm)$);"
    % (0, ys, -tick, ys),
    file=f
)
print(
    r"\draw[thick, greenstyle] ($(t_3_1_0.north west) + (%fmm, %fmm)$) -- ($(t_3_1_0.north west) + (%fmm, %fmm)$);"
    % (0, ys, -tick, ys),
    file=f
)
print(
    r"\draw[thick, <->, greenstyle] ($(t_0_1_0.north west) + (%fmm, %fmm)$) -- ($(t_3_1_0.north west) + (%fmm, %fmm)$);"
    % (-tick/2, ys, -tick/2, ys),
    file=f
)
print(
    r"\node(D0) [anchor=west] at($(t_0_1_0.north west)!0.5!(t_3_1_0.north west) + (%fmm, %fmm)$) {\greenBox{$\omega.D_0 = 4$}};"
    % (-tick, ys),
    file=f
)

ys = -13.0
print(
    r"\draw[thick, violetstyle] ($(t_0_1_0.north west) + (%fmm, %fmm)$) -- ($(t_0_1_0.north west) + (%fmm, %fmm)$);"
    % (0, ys, -tick, ys),
    file=f
)
print(
    r"\draw[thick, violetstyle] ($(t_0_0_0.north west) + (%fmm, %fmm)$) -- ($(t_0_0_0.north west) + (%fmm, %fmm)$);"
    % (0, ys, -tick, ys),
    file=f
)
print(
    r"\draw[thick, <->, violetstyle] ($(t_0_1_0.north west) + (%fmm, %fmm)$) -- ($(t_0_0_0.north west) + (%fmm, %fmm)$);"
    % (-tick/2, ys, -tick/2, ys),
    file=f
)
print(
    r"\node(D1) [anchor=center] at($(t_0_1_0.north west)!0.5!(t_0_0_0.north west) + (%fmm, %fmm)$) {\violetBox{$\omega.D_1 = 2$}};"
    % (-tick/2, ys),
    file=f
)

print(
    r"\draw[thick, bluestyle] ($(t_3_1_0.south) + (0mm, 0mm)$) -- ($(t_3_1_0.south) + (0mm, %f0mm)$);"
    % (-tick,),
    file=f
)
print(
    r"\draw[thick, bluestyle] ($(t_3_1_383.south) + (0mm, 0mm)$) -- ($(t_3_1_383.south) + (0mm, %f0mm)$);"
    % (-tick,),
    file=f
)
print(
    r"\draw[thick, <->, bluestyle] ($(t_3_1_0.south) + (0mm, %fmm)$) -- ($(t_3_1_383.south) + (0mm, %fmm)$);"
    % (-tick/2, -tick/2),
    file=f
)
print(
    r"\node(D2) [anchor=center] at($(t_3_1_0.south)!0.5!(t_3_1_383.south) + (0mm, %fmm)$) {\blueBox{$\omega.D_2 = 384$}};"
    % (-tick/2,),
    file=f
)

for c1 in (1, 0):
    t0 = "t_3_0_383.south east"
    t1 = "t_3_1_383.south east"
    top = "(%s)!%f!(%s)" % (t0, 0.5 * c1 + 0.45, t1)
    bottom = "(%s)!%f!(%s)" % (t0, 0.5 * c1 + 0.05, t1)
    middle = "(%s)!%f!(%s)" % (t0, 0.5 * c1 + 0.25, t1)
    print(
        r"\draw[thick, violetstyle] ($%s$) -- ($%s + (%fmm, 0mm)$);"
        % (top, top, tick),
        file=f
    )
    print(
        r"\draw[thick, violetstyle] ($%s$) -- ($%s + (%fmm, 0mm)$);"
        % (bottom, bottom, tick),
        file=f
    )
    print(
        r"\draw[thick, violetstyle] ($%s + (%fmm, 0mm)$) -- ($%s + (%fmm, 0mm)$);"
        % (top, tick, bottom, tick),
        file=f
    )
    print(
        r"\node[anchor=center] at($%s + (%fmm, 0mm)$) {\violetBox{\texttt{n\_cta=%i}}};"
        % (middle, tick, c1),
        file=f
    )

for c0 in (0, 1, 2, 3):
    top = "(t_%i_0_383.north east)" % (c0, )
    bottom = "(t_%i_0_383.south east)" % (c0, )
    middle = "%s!0.5!%s" % (top, bottom)
    print(
        r"\draw[thick, greenstyle] ($%s + (%fmm, 0mm)$) -- ($%s + (%fmm, 0mm)$);"
        % (top, tick * 0.25, top, tick * 1.25),
        file=f
    )
    print(
        r"\draw[thick, greenstyle] ($%s + (%fmm, 0mm)$) -- ($%s + (%fmm, 0mm)$);"
        % (bottom, tick * 0.25, bottom, tick * 1.25),
        file=f
    )
    print(
        r"\draw[thick, greenstyle] ($%s + (%fmm, 0mm)$) -- ($%s + (%fmm, 0mm)$);"
        % (bottom, tick * 1.25, top, tick * 1.25),
        file=f
    )
    print(
        r"\node[anchor=center] at($%s + (%fmm, 0mm)$) {\greenBox{\texttt{m\_cta=%i}}};"
        % (middle, tick * 1.25, c0),
        file=f
    )

def print_blue_interval(var, value, c2_min, c2_max, ys):
    xs = 1.75
    left = "(t_0_0_%i.north west)" % (c2_min, )
    right = "(t_0_0_%i.north east)" % (c2_max, )
    print(
        r"\draw[thick, bluestyle] ($%s + (%fmm, %fmm)$) -- ($%s + (%fmm, %fmm)$);"
        % (left, xs, ys, left, xs, ys + tick),
        file=f
    )
    print(
        r"\draw[thick, bluestyle] ($%s + (%fmm, %fmm)$) -- ($%s + (%fmm, %fmm)$);"
        % (right, -xs, ys, right, -xs, ys + tick),
        file=f
    )
    print(
        r"\draw[thick, bluestyle] ($%s + (%fmm, %fmm)$) -- ($%s + (%fmm, %fmm)$);"
        % (left, xs, ys + tick, right, -xs, ys + tick),
        file=f
    )
    print(
        r"\node[anchor=center] at($%s + (%fmm, %fmm)$) {\blueBox{\texttt{%s=%i}}};"
        % (left + "!0.5!" + right, 0, ys + tick, var.replace("_", r"\_"), value),
        file=f
    )

print_blue_interval("CudaWarps_consumer_None_None", 0, 128, 383, tick * 2.75)
for wg in (0, 1):
    print_blue_interval("wg", wg, 128 * (wg + 1), 128 * (wg + 1) + 127, tick * 1.5)
    for t in (0, 1, 127):
        c2 = 128 + wg * 128 + t
        print_blue_interval("t", t, c2, c2, tick * 0.25)

f.close()
