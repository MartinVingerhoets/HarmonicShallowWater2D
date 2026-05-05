import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Dates
using HarmonicShallowWater2D
using LinearAlgebra
using Logging
using Measures
using Plots
using Printf

function parse_int_list(name::AbstractString, default::Vector{Int})
    value = get(ENV, name, "")
    isempty(strip(value)) && return default
    return [parse(Int, strip(x)) for x in split(value, ",") if !isempty(strip(x))]
end

function parse_grid_list(name::AbstractString, default::Vector{Tuple{Int,Int}})
    value = get(ENV, name, "")
    isempty(strip(value)) && return default

    grids = Tuple{Int,Int}[]
    for item in split(value, ",")
        parts = split(lowercase(strip(item)), "x")
        length(parts) == 2 || error("Invalid grid '$item'. Use Nx x Ny, e.g. 30x10.")
        push!(grids, (parse(Int, strip(parts[1])), parse(Int, strip(parts[2]))))
    end
    return grids
end

function parse_solver_list(name::AbstractString, default::Vector{Symbol})
    value = get(ENV, name, "")
    isempty(strip(value)) && return default
    solvers = Symbol[]
    for item in split(value, ",")
        solver = Symbol(lowercase(strip(item)))
        solver in (:amg, :lu) || error("Invalid solver '$item'. Use amg or lu.")
        push!(solvers, solver)
    end
    return solvers
end

env_bool(name::AbstractString, default::Bool) =
    lowercase(strip(get(ENV, name, string(default)))) in ("true", "1", "yes", "on")

function run_case(; reduced_solver::Symbol, K::Int, Nx::Int, Ny::Int)
    cfg = ModelConfig(
        Nx = Nx,
        Ny = Ny,
        Lx = 3e3,
        Ly = 1e3,
        H = 3.0,
        g = 9.81,
        r = 5.0e-3,
        U = 0.5,
        σ = 2π / 44700,
        K = K,
        latitude_deg = 50.0,
        topography = sinusoidal_topography(3e3; amplitude = 1.0),
    )

    prec = PreconditionerOptions(
        reduced_solver = reduced_solver,
        constant_D0_in_helmholtz = true,
        C_prec = 8.0,
        prec_drag_floor = 0.0,
        reuse_preconditioner = true,
        amg_max_levels = 10,
        amg_max_coarse = 16,
        amg_jacobi_ω = 0.8,
        amg_pre_iters = 2,
        amg_post_iters = 2,
        amg_cycles_per_apply = 1,
        helmholtz_reg_eps = 0.0,
        helmholtz_max_reg_tries = 1,
        verbose = env_bool("HSW_TEST_VERBOSE", false),
    )

    solver = SolverOptions(
        abstol = parse(Float64, get(ENV, "HSW_TEST_ABSTOL", "1e-8")),
        reltol = parse(Float64, get(ENV, "HSW_TEST_RELTOL", "1e-8")),
        maxiters = parse(Int, get(ENV, "HSW_TEST_MAXITERS", "30")),
        gmres_restart = parse(Int, get(ENV, "HSW_TEST_GMRES_RESTART", "50")),
        gmres_verbose = parse(Int, get(ENV, "HSW_TEST_GMRES_VERBOSE", "0")),
        show_trace = env_bool("HSW_TEST_SHOW_TRACE", false),
    )

    verbose = env_bool("HSW_TEST_VERBOSE", false)
    timed = @timed begin
        if verbose
            solve_problem(cfg; solver_options = solver, preconditioner_options = prec)
        else
            Logging.with_logger(Logging.NullLogger()) do
                solve_problem(cfg; solver_options = solver, preconditioner_options = prec)
            end
        end
    end
    run = timed.value

    residual = similar(run.solution.u)
    residuals2D!(residual, run.solution.u, run.params)
    counts = gmres_iters_per_newton(run.gmres_trace)

    return (
        reduced_solver = reduced_solver,
        K = K,
        Nx = Nx,
        Ny = Ny,
        nunknowns = length(run.solution.u),
        elapsed_s = timed.time,
        allocated_bytes = timed.bytes,
        gc_time_s = timed.gctime,
        newton_steps = length(counts),
        total_gmres = length(run.gmres_trace.residuals),
        max_gmres_per_newton = isempty(counts) ? 0 : maximum(counts),
        final_residual_norm = norm(residual),
        retcode = string(run.solution.retcode),
    )
end

function write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "solver,K,Nx,Ny,nunknowns,elapsed_s,allocated_bytes,gc_time_s,newton_steps,total_gmres,max_gmres_per_newton,final_residual_norm,retcode")
        for r in rows
            @printf(
                io,
                "%s,%d,%d,%d,%d,%.6f,%d,%.6f,%d,%d,%d,%.16e,%s\n",
                r.reduced_solver,
                r.K,
                r.Nx,
                r.Ny,
                r.nunknowns,
                r.elapsed_s,
                r.allocated_bytes,
                r.gc_time_s,
                r.newton_steps,
                r.total_gmres,
                r.max_gmres_per_newton,
                r.final_residual_norm,
                r.retcode,
            )
        end
    end
    return path
end

function sorted_unique(values)
    return sort!(collect(unique(values)); by = string)
end

function plot_grouped!(plt, rows, xfield::Symbol, yfield::Symbol, groupfields::Tuple; ylabel::AbstractString)
    groups = sorted_unique(Tuple(getfield(r, f) for f in groupfields) for r in rows)
    for group in groups
        selected = [r for r in rows if Tuple(getfield(r, f) for f in groupfields) == group]
        sort!(selected; by = r -> getfield(r, xfield))
        x = [getfield(r, xfield) for r in selected]
        y = [getfield(r, yfield) for r in selected]
        label = join(["$(f)=$(v)" for (f, v) in zip(groupfields, group)], ", ")
        plot!(plt, x, y; marker = :circle, linewidth = 2, label = label)
    end
    xlabel!(plt, string(xfield))
    ylabel!(plt, ylabel)
    return plt
end

function plot_scaling(path::AbstractString, rows)
    plot_rows = [
        merge(
            r,
            (
                grid_cells = r.Nx * r.Ny,
                allocated_gib = r.allocated_bytes / 1024^3,
            ),
        )
        for r in rows
    ]

    p1 = plot(title = "Runtime vs K", legend = :outertopright)
    plot_grouped!(p1, plot_rows, :K, :elapsed_s, (:reduced_solver, :grid_cells); ylabel = "seconds")

    p2 = plot(title = "Runtime vs spatial size", legend = :outertopright)
    plot_grouped!(p2, plot_rows, :grid_cells, :elapsed_s, (:reduced_solver, :K); ylabel = "seconds")

    p3 = plot(title = "Memory allocated vs K", legend = :outertopright)
    plot_grouped!(p3, plot_rows, :K, :allocated_gib, (:reduced_solver, :grid_cells); ylabel = "GiB allocated")

    p4 = plot(title = "Memory allocated vs spatial size", legend = :outertopright)
    plot_grouped!(p4, plot_rows, :grid_cells, :allocated_gib, (:reduced_solver, :K); ylabel = "GiB allocated")

    fig = plot(p1, p2, p3, p4; layout = (2, 2), size = (1500, 950), margin = 6mm)
    savefig(fig, path)
    return path
end

frequencies = parse_int_list("HSW_TEST_KS", [1, 2])
grids = parse_grid_list("HSW_TEST_GRIDS", [(60, 20), (120, 40)])
solvers = parse_solver_list("HSW_TEST_SOLVERS", [:amg, :lu])
warmup_enabled = env_bool("HSW_TEST_WARMUP", true)
warmup_K = parse(Int, get(ENV, "HSW_TEST_WARMUP_K", "1"))
warmup_grid = first(parse_grid_list("HSW_TEST_WARMUP_GRID", [(8, 3)]))
default_outfile = joinpath(
    @__DIR__,
    "helmholtz_solver_comparison_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv",
)
outfile = get(ENV, "HSW_TEST_OUTFILE", default_outfile)
default_plotfile = replace(outfile, r"\.csv$" => "_scaling.png")
plotfile = get(ENV, "HSW_TEST_PLOTFILE", default_plotfile)

if warmup_enabled
    warmup_Nx, warmup_Ny = warmup_grid
    println("Warming up with K=$warmup_K, grid=$(warmup_Nx)x$(warmup_Ny)")
    for reduced_solver in solvers
        print("  warmup reduced_solver=:$reduced_solver ... ")
        run_case(; reduced_solver = reduced_solver, K = warmup_K, Nx = warmup_Nx, Ny = warmup_Ny)
        println("done")
    end
    GC.gc()
end

rows = []
for (Nx, Ny) in grids, K in frequencies, reduced_solver in solvers
    println("Running reduced_solver=:$reduced_solver, K=$K, grid=$(Nx)x$(Ny)")
    GC.gc()
    row = run_case(; reduced_solver = reduced_solver, K = K, Nx = Nx, Ny = Ny)
    push!(rows, row)
    @printf(
        "  retcode=%s, time=%.3fs, Newton=%d, GMRES=%d, ||F||=%.3e\n",
        row.retcode,
        row.elapsed_s,
        row.newton_steps,
        row.total_gmres,
        row.final_residual_norm,
    )
end

write_csv(outfile, rows)
plot_scaling(plotfile, rows)
println("Wrote $(length(rows)) comparison rows to $outfile")
println("Wrote scaling plot to $plotfile")
println("Override with HSW_TEST_KS=\"1,3,5\" and HSW_TEST_GRIDS=\"30x10,60x20\".")
