import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using HarmonicShallowWater2D

cfg = ModelConfig(
    Nx = 300,
    Ny = 100,
    Lx = 3e3,
    Ly = 1e3,
    H = 3.0,
    g = 9.81,
    r = 2.5e-3,
    U = 0.5,
    σ = 2π / 44700,
    K = 5,
    latitude_deg = 50.0,
    topography = sinusoidal_topography(3e3; amplitude = 1.0),
)

prec = PreconditionerOptions(
    constant_D0_in_helmholtz = true,
    C_prec = 8.0,
    prec_drag_floor = 0.0,
    reuse_preconditioner = false,
    amg_max_levels = 10,
    amg_max_coarse = 16,
    amg_jacobi_ω = 0.8,
    amg_pre_iters = 2,
    amg_post_iters = 2,
    amg_cycles_per_apply = 1,
    helmholtz_reg_eps = 1e-10,
    helmholtz_max_reg_tries = 1,
    verbose = true,
)

solver = SolverOptions(
    abstol = 1e-10,
    reltol = 1e-10,
    maxiters = 50,
    gmres_restart = 50,
    gmres_verbose = 1,
)

run = solve_problem(cfg; solver_options = solver, preconditioner_options = prec)

fig_gmres, gmres_counts = plot_gmres_across_newton(run.gmres_trace; savepath = "gmres_across_newton.png")
fig_xt, _ = plot_midrow_xt(run.solution, run.params; nperiods = 30, nt_per_period = 64)
