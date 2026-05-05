module HarmonicShallowWater2D

using NonlinearSolve
using LinearAlgebra
using SparseArrays
using FFTW
using Statistics
using Plots
using Measures
using ADTypes
using Dates
using Profile
using Serialization
using AlgebraicMultigrid
import LinearSolve as LS
import ParU_jll
using PolyesterForwardDiff

export ModelConfig,
       PreconditionerOptions,
       SolverOptions,
       SolveArtifacts,
       Params2DHarm,
       GMRESNewtonTrace,
       PartialHelmholtzSinCosBuilder,
       build_params,
       effective_coriolis,
       sinusoidal_topography,
       state_length,
       initial_state,
       nonlinear_problem,
       solve_problem,
       make_preconditioner_builder,
       residuals2D!,
       initialize_aux_speed!,
       gmres_iters_per_newton,
       plot_midrow_xt,
       plot_gmres_across_newton,
       reconstruct_center_fields,
       save_solution_vector,
       load_solution_vector

include("Constants.jl")
include("Core.jl")
include("Preconditioners.jl")
include("API.jl")

end
