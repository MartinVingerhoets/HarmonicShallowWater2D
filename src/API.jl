Base.@kwdef struct ModelConfig
    Nx::Int = 300
    Ny::Int = 100
    Lx::Float64 = 3e3
    Ly::Float64 = 1e3

    H::Float64 = 3.0
    g::Float64 = 9.81
    r::Float64 = 2.5e-3
    U::Float64 = 0.5
    σ::Float64 = 2π / 44700

    Ω::Float64 = 7.29e-5
    latitude_deg::Float64 = 50.0
    Ef::Union{Nothing,Float64} = nothing

    drag_eps::Float64 = 0.0
    K::Int = 5

    topography::Any = nothing
    residual_timing_enabled::Bool = false
end

Base.@kwdef struct PreconditionerOptions
    reduced_solver::Symbol = :amg
    constant_D0_in_helmholtz::Bool = true
    constant_D0_value::Union{Nothing,Float64} = nothing
    C_prec::Float64 = 8.0
    prec_drag_floor::Float64 = 0.0
    reuse_preconditioner::Bool = false
    amg_max_levels::Int = 10
    amg_max_coarse::Int = 16
    amg_setup_reg_eps::Float64 = 0.0
    amg_jacobi_ω::Float64 = 0.8
    amg_pre_iters::Int = 2
    amg_post_iters::Int = 2
    amg_cycles_per_apply::Int = 1
    shifted_amg_gpw_smoothing_threshold::Float64 = 30.0
    shifted_amg_gpw_smoothing_iters::Int = 4
    helmholtz_reg_eps::Float64 = 1e-10
    helmholtz_max_reg_tries::Int = 1
    verbose::Bool = true
end

Base.@kwdef struct SolverOptions
    load_initial_solution::Bool = false
    initial_solution_filename::Union{Nothing,String} = nothing
    save_final_solution::Bool = false
    final_solution_filename::Union{Nothing,String} = nothing

    abstol::Float64 = 1e-8
    reltol::Float64 = 1e-8
    maxiters::Int = 50

    gmres_restart::Int = 50
    gmres_verbose::Int = 1
    gmres_history::Bool = true

    show_trace::Bool = true
    trace_level = TraceMinimal(1)
    autodiff = ADTypes.AutoPolyesterForwardDiff()
    forcing = EisenstatWalkerForcing2()
end

struct SolveArtifacts{S,P,T,B}
    solution::S
    params::P
    gmres_trace::T
    preconditioner_builder::B
end

function effective_coriolis(cfg::ModelConfig)
    return something(cfg.Ef, 2 * cfg.Ω * sin(cfg.latitude_deg * π / 180.0))
end

function default_topography_matrix(cfg::ModelConfig, Δx::Float64, Δy::Float64)
    xC = [(i - 0.5) * Δx for i in 1:cfg.Nx]
    yC = [(j - 0.5) * Δy for j in 1:cfg.Ny]

    topo = cfg.topography
    if topo === nothing
        topo = sinusoidal_topography(cfg.Lx)
    end

    if topo isa AbstractMatrix
        size(topo) == (cfg.Nx, cfg.Ny) ||
            error("Topography matrix has size $(size(topo)), expected ($(cfg.Nx), $(cfg.Ny)).")
        return Matrix{Float64}(topo)
    elseif topo isa Function
        return [Float64(topo(xC[i], yC[j])) for i in 1:cfg.Nx, j in 1:cfg.Ny]
    else
        error("Unsupported topography representation: $(typeof(topo)). Use a function or an Nx×Ny matrix.")
    end
end

function build_params(cfg::ModelConfig)
    Δx = cfg.Lx / cfg.Nx
    Δy = cfg.Ly / cfg.Ny
    hC = default_topography_matrix(cfg, Δx, Δy)

    Kpos = cfg.K
    Kloc = Kpos + 1
    K2pos = 2 * Kpos
    K2loc = K2pos + 1

    _, sin_kt, cos_kt, _ = make_time_basis(Kpos, cfg.σ; Nt = 512)
    U0_t = cfg.U .* vec(sin_kt[2, :])
    bg_t = (cfg.r / cfg.H) .* abs.(U0_t) .* U0_t
    bg_s, bg_c = project_signal_to_basis(bg_t, sin_kt, cos_kt, Kpos)

    nface = cfg.Nx * cfg.Ny
    nfields = 10
    harm_block = nfields * nface

    return Params2DHarm(
        cfg.Nx, cfg.Ny, Δx, Δy,
        cfg.H, cfg.g, cfg.r, cfg.U, cfg.σ, effective_coriolis(cfg), cfg.drag_eps, hC,
        Kpos, Kloc, K2pos, K2loc,
        nface, nfields, harm_block,
        bg_s, bg_c,
        IdDict{DataType, Any}(),
        ReentrantLock(),
    )
end

function make_preconditioner_builder(p::Params2DHarm, opts::PreconditionerOptions)
    return PartialHelmholtzSinCosBuilder(
        p = p,
        reduced_solver = opts.reduced_solver,
        constant_D0_in_helmholtz = opts.constant_D0_in_helmholtz,
        constant_D0_value = opts.constant_D0_value,
        C_prec = opts.C_prec,
        prec_drag_floor = opts.prec_drag_floor,
        reuse_preconditioner = opts.reuse_preconditioner,
        amg_max_levels = opts.amg_max_levels,
        amg_max_coarse = opts.amg_max_coarse,
        amg_setup_reg_eps = opts.amg_setup_reg_eps,
        amg_jacobi_ω = opts.amg_jacobi_ω,
        amg_pre_iters = opts.amg_pre_iters,
        amg_post_iters = opts.amg_post_iters,
        amg_cycles_per_apply = opts.amg_cycles_per_apply,
        shifted_amg_gpw_smoothing_threshold = opts.shifted_amg_gpw_smoothing_threshold,
        shifted_amg_gpw_smoothing_iters = opts.shifted_amg_gpw_smoothing_iters,
        helmholtz_reg_eps = opts.helmholtz_reg_eps,
        helmholtz_max_reg_tries = opts.helmholtz_max_reg_tries,
        verbose = opts.verbose,
    )
end

function initial_state(p::Params2DHarm; solution_filename::Union{Nothing,String} = nothing)
    y0 = zeros(Float64, state_length(p))
    _ = get_scratch!(p, Float64, 1)

    if solution_filename === nothing
        initialize_aux_speed!(y0, p)
    else
        y0 .= load_solution_vector(solution_filename, length(y0))
    end
    return y0
end

function nonlinear_problem(p::Params2DHarm, y0::AbstractVector)
    f = NonlinearFunction(residuals2D!)
    return NonlinearProblem(f, y0, p)
end

function solve_problem(
    cfg::ModelConfig;
    solver_options::SolverOptions = SolverOptions(),
    preconditioner_options::PreconditionerOptions = PreconditionerOptions(),
)
    params = build_params(cfg)
    init_file = solver_options.load_initial_solution ? solver_options.initial_solution_filename : nothing
    y0 = initial_state(params; solution_filename = init_file)

    F0 = similar(y0)
    residuals2D!(F0, y0, params)

    prob = nonlinear_problem(params, y0)

    builder = make_preconditioner_builder(params, preconditioner_options)
    gmres_trace = GMRESNewtonTrace(Float64)
    gmres_callback = make_gmres_newton_callback(gmres_trace)

    alg = NewtonRaphson(
        autodiff = solver_options.autodiff,
        linsolve = LS.KrylovJL_GMRES(
            precs = builder,
            verbose = solver_options.gmres_verbose,
            history = solver_options.gmres_history,
            callback = gmres_callback,
            gmres_restart = solver_options.gmres_restart,
        ),
        forcing = solver_options.forcing,
    )

    sol = solve(
        prob,
        alg;
        abstol = solver_options.abstol,
        reltol = solver_options.reltol,
        maxiters = solver_options.maxiters,
        show_trace = Val(solver_options.show_trace),
        trace_level = solver_options.trace_level,
    )

    if solver_options.save_final_solution
        filename = something(solver_options.final_solution_filename, "saved_solution_vector.ser")
        save_solution_vector(filename, sol.u)
    end

    return SolveArtifacts(sol, params, gmres_trace, builder)
end
