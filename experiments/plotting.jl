# Shared SVG helpers for the Extended Kelly II figures (standard library only; ADR-008).
using Printf

const INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
const PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
const SHAPES = [:circle, :square, :triangle, :diamond]

"""Marker at pixel (x, y) — extracted unchanged from e1_plot.jl."""
function svg_marker(io, shape, x, y, color; r=4.0)
    a = 0.55
    if shape == :circle
        @printf(io, "<circle cx=\"%.1f\" cy=\"%.1f\" r=\"%.1f\" fill=\"%s\" fill-opacity=\"%.2f\"/>\n", x, y, r, color, a)
    elseif shape == :square
        @printf(io, "<rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\" fill=\"%s\" fill-opacity=\"%.2f\"/>\n", x-r, y-r, 2r, 2r, color, a)
    elseif shape == :triangle
        @printf(io, "<polygon points=\"%.1f,%.1f %.1f,%.1f %.1f,%.1f\" fill=\"%s\" fill-opacity=\"%.2f\"/>\n",
                x, y-1.2r, x-1.1r, y+0.8r, x+1.1r, y+0.8r, color, a)
    else
        @printf(io, "<polygon points=\"%.1f,%.1f %.1f,%.1f %.1f,%.1f %.1f,%.1f\" fill=\"%s\" fill-opacity=\"%.2f\"/>\n",
                x, y-1.3r, x+1.1r, y, x, y+1.3r, x-1.1r, y, color, a)
    end
end

"""Split a legend label into at most two lines of about `width` characters — extracted unchanged from e1_plot.jl."""
function wrap_label(label; width = 26)
    line1, line2 = "", ""
    for w in split(label, " ")
        length(line1) + length(w) < width && line2 == "" ? (line1 = strip(line1 * " " * w)) : (line2 = strip(line2 * " " * w))
    end
    return line1, line2
end

fmt_tick(v) = abs(v) ≥ 1000 || (abs(v) < 0.01 && v != 0) ? @sprintf("%.0e", v) : (v == round(v) ? @sprintf("%d", round(Int, v)) : @sprintf("%g", v))

"""
    svg_plot(path; series, xlim, ylim, xticks, yticks, xlabel, ylabel, xlog = false, hlines = [], vlines = [], legend_title = "", W = 760, H = 470)

One line/point chart in the Extended Kelly II house style. Each series is a NamedTuple
`(x, y, label, kind = :line | :points | :both, color_index = k, dash = false)`. `hlines` / `vlines` are `(value, label)`
reference lines (dashed, ink). Log x-axis uses base-10 ticks at the given `xticks`.
"""
function svg_plot(path; series, xlim, ylim, xticks, yticks, xlabel, ylabel, xlog = false, hlines = (), vlines = (),
                  legend_title = "", W = 760, H = 470)
    L, R, T, B = 70, 250, 40, 60
    tx(x) = xlog ? log10(x) : x
    px(x) = L + (tx(x) - tx(xlim[1])) / (tx(xlim[2]) - tx(xlim[1])) * (W - L - R)
    py(y) = T + (ylim[2] - y) / (ylim[2] - ylim[1]) * (H - T - B)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 $W $H\" width=\"$W\" height=\"$H\" font-family=\"Helvetica, Arial, sans-serif\">")
        println(io, "<rect width=\"$W\" height=\"$H\" fill=\"$BG\"/>")
        for x in xticks
            @printf(io, "<line x1=\"%.1f\" y1=\"%d\" x2=\"%.1f\" y2=\"%d\" stroke=\"%s\"/>\n", px(x), T, px(x), H - B, GRID)
            if xlog
                @printf(io, "<text x=\"%.1f\" y=\"%d\" font-size=\"12\" fill=\"%s\" text-anchor=\"middle\">10<tspan dy=\"-5\" font-size=\"9\">%d</tspan></text>\n", px(x), H - B + 18, INK2, round(Int, log10(x)))
            else
                @printf(io, "<text x=\"%.1f\" y=\"%d\" font-size=\"12\" fill=\"%s\" text-anchor=\"middle\">%s</text>\n", px(x), H - B + 18, INK2, fmt_tick(x))
            end
        end
        for y in yticks
            @printf(io, "<line x1=\"%d\" y1=\"%.1f\" x2=\"%d\" y2=\"%.1f\" stroke=\"%s\"/>\n", L, py(y), W - R, py(y), GRID)
            @printf(io, "<text x=\"%d\" y=\"%.1f\" font-size=\"12\" fill=\"%s\" text-anchor=\"end\">%s</text>\n", L - 8, py(y) + 4, INK2, fmt_tick(y))
        end
        for (v, lab) in hlines
            @printf(io, "<line x1=\"%d\" y1=\"%.1f\" x2=\"%d\" y2=\"%.1f\" stroke=\"%s\" stroke-width=\"1.5\" stroke-dasharray=\"6 4\"/>\n", L, py(v), W - R, py(v), INK)
            isempty(lab) || @printf(io, "<text x=\"%d\" y=\"%.1f\" font-size=\"11\" fill=\"%s\">%s</text>\n", L + 6, py(v) - 6, INK, lab)
        end
        for (v, lab) in vlines
            @printf(io, "<line x1=\"%.1f\" y1=\"%d\" x2=\"%.1f\" y2=\"%d\" stroke=\"%s\" stroke-width=\"1.5\" stroke-dasharray=\"6 4\"/>\n", px(v), T, px(v), H - B, INK)
            isempty(lab) || @printf(io, "<text x=\"%.1f\" y=\"%d\" font-size=\"11\" fill=\"%s\">%s</text>\n", px(v) + 4, T + 12, INK, lab)
        end
        for (k, s) in enumerate(series)
            ci = get(s, :color_index, k); col = PALETTE[mod1(ci, 4)]; kind = get(s, :kind, :line)
            keep = [i for i in eachindex(s.x) if xlim[1] ≤ s.x[i] ≤ xlim[2] && isfinite(s.y[i])]
            if kind in (:line, :both)
                pts = join([@sprintf("%.1f,%.1f", px(s.x[i]), py(clamp(s.y[i], ylim...))) for i in keep], " ")
                dash = get(s, :dash, false) ? " stroke-dasharray=\"4 3\"" : ""
                println(io, "<polyline points=\"$pts\" fill=\"none\" stroke=\"$col\" stroke-width=\"2\"$dash/>")
            end
            if kind in (:points, :both)
                for i in keep; svg_marker(io, SHAPES[mod1(ci, 4)], px(s.x[i]), py(clamp(s.y[i], ylim...)), col); end
            end
        end
        @printf(io, "<text x=\"%.1f\" y=\"%d\" font-size=\"13\" fill=\"%s\" text-anchor=\"middle\">%s</text>\n", (L + W - R) / 2, H - 15, INK, xlabel)
        @printf(io, "<text transform=\"translate(18,%.1f) rotate(-90)\" font-size=\"13\" fill=\"%s\" text-anchor=\"middle\">%s</text>\n", (T + H - B) / 2, INK, ylabel)
        lx, ly = W - R + 20, T + 10
        isempty(legend_title) || @printf(io, "<text x=\"%d\" y=\"%d\" font-size=\"12\" font-weight=\"bold\" fill=\"%s\">%s</text>\n", lx, ly, INK, legend_title)
        for (k, s) in enumerate(series)
            ci = get(s, :color_index, k); col = PALETTE[mod1(ci, 4)]; y = ly + 34k
            if get(s, :kind, :line) == :points
                svg_marker(io, SHAPES[mod1(ci, 4)], lx + 6, y - 4, col; r = 5.0)
            else
                dash = get(s, :dash, false) ? " stroke-dasharray=\"4 3\"" : ""
                @printf(io, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"2\"%s/>\n", lx, y - 4, lx + 12, y - 4, col, dash)
            end
            l1, l2 = wrap_label(s.label)
            @printf(io, "<text x=\"%d\" y=\"%d\" font-size=\"11\" fill=\"%s\">%s</text>\n", lx + 18, y, INK, l1)
            l2 != "" && @printf(io, "<text x=\"%d\" y=\"%d\" font-size=\"11\" fill=\"%s\">%s</text>\n", lx + 18, y + 12, INK, l2)
        end
        println(io, "</svg>")
    end
    return path
end

using TOML

"""Write a CSV with a header row; numbers with `%.12e`, strings as given."""
function write_csv(path, header, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(header, ","))
        for r in rows
            println(io, join([x isa AbstractFloat ? @sprintf("%.12e", x) : string(x) for x in r], ","))
        end
    end
    return path
end

"""Write `<name>_meta.toml` (excluded from reference comparison): Julia version, git commit, seed, wall time."""
function write_meta(path; seed, wall_seconds, extra = Dict{String,Any}())
    commit = try readchomp(`git -C $(dirname(path)) rev-parse HEAD`) catch; "unknown" end
    d = Dict{String,Any}("julia_version" => string(VERSION), "git_commit" => commit, "seed" => seed,
                         "wall_seconds" => round(wall_seconds; digits = 3))
    merge!(d, extra)
    open(io -> TOML.print(io, d), path, "w")
    return path
end
