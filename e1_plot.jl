# E1 figure — growth lost to a wrong belief, as a fraction of D(p‖q).
# Standard library only: reads e1_results.csv and writes ../img/e1_kl_growth_loss.svg.
#
#   x-axis: D(p‖q), log scale         y-axis: L / D(p‖q)
# The bound L ≤ D(p‖q) appears as the ceiling y = 1.

using Random, Printf

const GROUPS = [   # (label, cases, color, marker)
    ("Full investment (with or without fees)", ["A_full_investment", "A2_full_inv_fees"], "#2a78d6", :circle),
    ("Cash, fair or superfair book",           ["B_cash", "D_cash_superfair"],           "#eb6834", :square),
    ("Cash, bookmaker margin",                 ["B2_cash_margin"],                       "#1baf7a", :triangle),
    ("Cash, margin + commission",              ["C_cash_margin_fee"],                    "#eda100", :diamond),
]

function read_results(path)
    lines = readlines(path)
    hdr = split(lines[1], ",")
    ic, ik, il = findfirst(==("case"), hdr), findfirst(==("kl"), hdr), findfirst(==("loss"), hdr)
    rows = Tuple{String,Float64,Float64}[]
    for ln in lines[2:end]
        f = split(ln, ",")
        push!(rows, (String(f[ic]), parse(Float64, f[ik]), parse(Float64, f[il])))
    end
    rows
end

function marker(io, shape, x, y, color; r=4.0)
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

function main(; csv=joinpath(@__DIR__, "e1_results.csv"),
               out=joinpath(@__DIR__, "..", "img", "e1_kl_growth_loss.svg"),
               per_group=500, seed=1)
    rows = filter(r -> r[2] > 1e-6, read_results(csv))
    rng = Xoshiro(seed)
    W, H = 760, 470
    L, R, T, B = 70, 250, 40, 60                       # margins (legend on the right)
    xmin, xmax = -6.0, 0.3                             # log10 D
    ymin, ymax = 0.0, 1.08
    px(x) = L + (log10(x) - xmin) / (xmax - xmin) * (W - L - R)
    py(y) = T + (ymax - y) / (ymax - ymin) * (H - T - B)
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e6e5e1"

    mkpath(dirname(out))
    open(out, "w") do io
        println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 $W $H\" width=\"$W\" height=\"$H\" font-family=\"Helvetica, Arial, sans-serif\">")
        println(io, "<rect width=\"$W\" height=\"$H\" fill=\"#fcfcfb\"/>")
        # grid + axes
        for e in Int(xmin):floor(Int, xmax)
            x = px(10.0^e)
            @printf(io, "<line x1=\"%.1f\" y1=\"%d\" x2=\"%.1f\" y2=\"%d\" stroke=\"%s\"/>\n", x, T, x, H-B, grid)
            @printf(io, "<text x=\"%.1f\" y=\"%d\" font-size=\"12\" fill=\"%s\" text-anchor=\"middle\">10<tspan dy=\"-5\" font-size=\"9\">%d</tspan></text>\n", x, H-B+18, ink2, e)
        end
        for y in 0.0:0.25:1.0
            yy = py(y)
            @printf(io, "<line x1=\"%d\" y1=\"%.1f\" x2=\"%d\" y2=\"%.1f\" stroke=\"%s\"/>\n", L, yy, W-R, yy, grid)
            @printf(io, "<text x=\"%d\" y=\"%.1f\" font-size=\"12\" fill=\"%s\" text-anchor=\"end\">%.2f</text>\n", L-8, yy+4, ink2, y)
        end
        # bound line y = 1
        @printf(io, "<line x1=\"%d\" y1=\"%.1f\" x2=\"%d\" y2=\"%.1f\" stroke=\"%s\" stroke-width=\"2\" stroke-dasharray=\"6 4\"/>\n", L, py(1), W-R, py(1), ink)
        # points
        for (_, cases, color, shape) in GROUPS
            g = filter(r -> r[1] in cases, rows)
            g = length(g) > per_group ? g[randperm(rng, length(g))[1:per_group]] : g
            for (_, d, l) in g
                (xmin <= log10(d) <= xmax) && marker(io, shape, px(d), py(clamp(l/d, ymin, ymax)), color)
            end
        end
        @printf(io, "<text x=\"%d\" y=\"%.1f\" font-size=\"12\" fill=\"%s\">L = D(p‖q): exact, and an upper bound</text>\n", L+8, py(1)-8, ink)
        # axis titles
        @printf(io, "<text x=\"%.1f\" y=\"%d\" font-size=\"13\" fill=\"%s\" text-anchor=\"middle\">KL divergence D(p‖q) of the belief from the truth (nats)</text>\n", (L + W - R)/2, H-15, ink)
        @printf(io, "<text transform=\"translate(18,%.1f) rotate(-90)\" font-size=\"13\" fill=\"%s\" text-anchor=\"middle\">Growth lost / D(p‖q)</text>\n", (T + H - B)/2, ink)
        # legend
        lx, ly = W - R + 20, T + 10
        @printf(io, "<text x=\"%d\" y=\"%d\" font-size=\"12\" font-weight=\"bold\" fill=\"%s\">Betting setup</text>\n", lx, ly, ink)
        for (k, (label, _, color, shape)) in enumerate(GROUPS)
            y = ly + 34k
            marker(io, shape, lx + 6, y - 4, color; r=5.0)
            words = split(label, " ")
            # wrap at ~26 chars
            line1, line2 = "", ""
            for w in words
                length(line1) + length(w) < 26 && line2 == "" ? (line1 = strip(line1 * " " * w)) : (line2 = strip(line2 * " " * w))
            end
            @printf(io, "<text x=\"%d\" y=\"%d\" font-size=\"11\" fill=\"%s\">%s</text>\n", lx + 18, y, ink, line1)
            line2 != "" && @printf(io, "<text x=\"%d\" y=\"%d\" font-size=\"11\" fill=\"%s\">%s</text>\n", lx + 18, y + 12, ink, line2)
        end
        println(io, "</svg>")
    end
    println("wrote $out")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
