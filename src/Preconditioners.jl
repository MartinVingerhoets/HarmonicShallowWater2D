struct HelmholtzGridOps{T}
    Gx::SparseMatrixCSC{T,Int}
    Gy::SparseMatrixCSC{T,Int}
    Dx::SparseMatrixCSC{T,Int}
    Dy::SparseMatrixCSC{T,Int}
    D0u::Vector{T}
    D0v::Vector{T}
    gDu::Vector{T}
    gDv::Vector{T}
end

function default_constant_D0(p::Params2DHarm)
    su = 0.0
    sv = 0.0
    @inbounds for j in 1:p.Ny, i in 1:p.Nx
        su += p.H - h_on_U_face(p, i, j)
        sv += p.H - h_on_V_face(p, i, j)
    end
    return 0.5 * (su + sv) / (p.Nx * p.Ny)
end

function build_helmholtz_grid_ops(
    p::Params2DHarm;
    use_constant_D0::Bool = false,
    D0_const::Union{Nothing,Float64} = nothing,
)
    T = Float64
    n = p.nface

    Igx = Int[]; Jgx = Int[]; Vgx = T[]
    Igy = Int[]; Jgy = Int[]; Vgy = T[]
    Idx = Int[]; Jdx = Int[]; Vdx = T[]
    Idy = Int[]; Jdy = Int[]; Vdy = T[]
    sizehint!(Igx, 2n); sizehint!(Jgx, 2n); sizehint!(Vgx, 2n)
    sizehint!(Igy, 2n); sizehint!(Jgy, 2n); sizehint!(Vgy, 2n)
    sizehint!(Idx, 2n); sizehint!(Jdx, 2n); sizehint!(Vdx, 2n)
    sizehint!(Idy, 2n); sizehint!(Jdy, 2n); sizehint!(Vdy, 2n)

    D0u = zeros(T, n)
    D0v = zeros(T, n)

    invΔx = inv(T(p.Δx))
    invΔy = inv(T(p.Δy))
    D0_fixed = use_constant_D0 ? T(something(D0_const, default_constant_D0(p))) : zero(T)

    for j in 1:p.Ny, i in 1:p.Nx
        face = face_idx(i, j, p.Nx)
        im = im1(i, p.Nx)
        ip = ip1(i, p.Nx)
        jm = im1(j, p.Ny)
        jp = ip1(j, p.Ny)

        face_im = face_idx(im, j,  p.Nx)
        face_ip = face_idx(ip, j,  p.Nx)
        face_jm = face_idx(i,  jm, p.Nx)
        face_jp = face_idx(i,  jp, p.Nx)

        push!(Igx, face); push!(Jgx, face);    push!(Vgx,  invΔx)
        push!(Igx, face); push!(Jgx, face_im); push!(Vgx, -invΔx)

        push!(Igy, face); push!(Jgy, face);    push!(Vgy,  invΔy)
        push!(Igy, face); push!(Jgy, face_jm); push!(Vgy, -invΔy)

        push!(Idx, face); push!(Jdx, face_ip); push!(Vdx,  invΔx)
        push!(Idx, face); push!(Jdx, face);    push!(Vdx, -invΔx)

        push!(Idy, face); push!(Jdy, face_jp); push!(Vdy,  invΔy)
        push!(Idy, face); push!(Jdy, face);    push!(Vdy, -invΔy)

        if use_constant_D0
            D0u[face] = D0_fixed
            D0v[face] = D0_fixed
        else
            D0u[face] = p.H - h_on_U_face(p, i, j)
            D0v[face] = p.H - h_on_V_face(p, i, j)
        end
    end

    Gx = sparse(Igx, Jgx, Vgx, n, n)
    Gy = sparse(Igy, Jgy, Vgy, n, n)
    Dx = sparse(Idx, Jdx, Vdx, n, n)
    Dy = sparse(Idy, Jdy, Vdy, n, n)

    return HelmholtzGridOps{T}(Gx, Gy, Dx, Dy, D0u, D0v, p.g .* D0u, p.g .* D0v)
end

function robust_sparse_lu(B::SparseMatrixCSC{T,Int}; reg_eps::Real = 1e-12, max_reg_tries::Int = 6) where {T}
    n = size(B, 1)
    scale = max(one(T), T(opnorm(B, Inf)))

    for k in 0:max_reg_tries
        try
            if k == 0
                return lu(B)
            else
                δ = T(reg_eps * (10.0^(k - 1))) * scale
                return lu(B + spdiagm(0 => fill(δ, n)))
            end
        catch err
            if err isa SingularException || err isa LinearAlgebra.ZeroPivotException
                continue
            else
                rethrow()
            end
        end
    end

    error("Sparse LU factorization failed, even after diagonal regularization.")
end

mutable struct PartialHelmholtzApplyScratch{T}
    fu_s::Vector{T}
    fu_c::Vector{T}
    fv_s::Vector{T}
    fv_c::Vector{T}
    fz_s::Vector{T}
    fz_c::Vector{T}
    tmp1::Vector{T}
    tmp2::Vector{T}
    tmp3::Vector{T}
    tmp4::Vector{T}
    rhs2::Vector{T}
    z2::Vector{T}
    gradx_s::Vector{T}
    gradx_c::Vector{T}
    grady_s::Vector{T}
    grady_c::Vector{T}
    us::Vector{T}
    uc::Vector{T}
    vs::Vector{T}
    vc::Vector{T}
end

function PartialHelmholtzApplyScratch(::Type{T}, nface::Int) where {T}
    rvec(n) = zeros(T, n)
    return PartialHelmholtzApplyScratch{T}(
        rvec(nface), rvec(nface), rvec(nface), rvec(nface), rvec(nface), rvec(nface),
        rvec(nface), rvec(nface), rvec(nface), rvec(nface),
        rvec(2 * nface), rvec(2 * nface),
        rvec(nface), rvec(nface), rvec(nface), rvec(nface),
        rvec(nface), rvec(nface), rvec(nface), rvec(nface),
    )
end

struct AMGTransferCache{T}
    A_real::SparseMatrixCSC{T,Int}
    P_levels::Vector{SparseMatrixCSC{T,Int}}
    R_levels::Vector{SparseMatrixCSC{T,Int}}
    Pbig_levels::Vector{SparseMatrixCSC{T,Int}}
    Rbig_levels::Vector{SparseMatrixCSC{T,Int}}
end

mutable struct ShiftedAMGWorkspace{T}
    residuals::Vector{Vector{T}}
    coarse_bs::Vector{Vector{T}}
    coarse_xs::Vector{Vector{T}}
end

mutable struct ScalarAMGWorkspace{T}
    residuals::Vector{Vector{T}}
    coarse_bs::Vector{Vector{T}}
    coarse_xs::Vector{Vector{T}}
end

struct ShiftedAMGHierarchy{T,LF}
    H_levels::Vector{SparseMatrixCSC{T,Int}}
    Pbig_levels::Vector{SparseMatrixCSC{T,Int}}
    Rbig_levels::Vector{SparseMatrixCSC{T,Int}}
    Dinv_levels::Vector{Vector{T}}
    coarse_factor::LF
    jacobi_ω::T
    pre_iters::Int
    post_iters::Int
    cycles_per_apply::Int
    workspaces::Vector{ShiftedAMGWorkspace{T}}
end

struct ScalarAMGHierarchy{T,LF}
    A_levels::Vector{SparseMatrixCSC{T,Int}}
    P_levels::Vector{SparseMatrixCSC{T,Int}}
    R_levels::Vector{SparseMatrixCSC{T,Int}}
    Dinv_levels::Vector{Vector{T}}
    coarse_factor::LF
    jacobi_ω::T
    pre_iters::Int
    post_iters::Int
    cycles_per_apply::Int
    workspaces::Vector{ScalarAMGWorkspace{T}}
end

struct ShiftedLUSolver{T,LF}
    H::SparseMatrixCSC{T,Int}
    factor::LF
end

struct ScalarLUSolver{T,LF}
    A::SparseMatrixCSC{T,Int}
    factor::LF
end

struct SinCosHelmholtzHarmonicBlock{T,HF}
    m::Int
    α::T
    γ::T
    hierarchy::HF
    a::T
    b::T
    pcoef::T
    qcoef::T
end

struct MeanModeAMGBlock{T,HF}
    drag::T
    hierarchy::HF
    nface::Int
end

struct PartialHelmholtzSinCosPrec{T,MB,HBV}
    ops::HelmholtzGridOps{T}
    mean_block::MB
    helm_blocks::HBV
    scratch::Vector{PartialHelmholtzApplyScratch{T}}
    nface::Int
    harm_block::Int
    Kloc::Int
    n::Int
end

Base.eltype(::PartialHelmholtzSinCosPrec{T}) where {T} = T
Base.size(P::PartialHelmholtzSinCosPrec) = (P.n, P.n)

Base.@kwdef mutable struct PartialHelmholtzSinCosBuilder
    p::Params2DHarm
    reduced_solver::Symbol = :amg
    constant_D0_in_helmholtz::Bool = true
    constant_D0_value::Union{Nothing,Float64} = nothing
    C_prec::Float64 = 1.0
    prec_drag_floor::Float64 = 0.0
    reuse_preconditioner::Bool = false
    amg_max_levels::Int = 10
    amg_max_coarse::Int = 16
    amg_setup_reg_eps::Float64 = 0.0
    amg_jacobi_ω::Float64 = 0.8
    amg_pre_iters::Int = 2
    amg_post_iters::Int = 2
    amg_cycles_per_apply::Int = 1
    helmholtz_reg_eps::Float64 = 1e-10
    helmholtz_max_reg_tries::Int = 6
    verbose::Bool = true
    cache::Any = nothing
    last_effective_drag::Float64 = NaN
end

function validate_reduced_solver(solver::Symbol)
    solver in (:amg, :lu) || error("Unsupported reduced_solver = $(repr(solver)). Use :amg or :lu.")
    return solver
end

reduced_solver_label(::Val{:amg}) = "AMG"
reduced_solver_label(::Val{:lu}) = "LU"
reduced_solver_label(solver::Symbol) = reduced_solver_label(Val(validate_reduced_solver(solver)))

@inline function _prec_idx(P::PartialHelmholtzSinCosPrec, m::Int, field::Int, face::Int)
    return (m - 1) * P.harm_block + (field - 1) * P.nface + face
end

function build_real_helmholtz_matrix(ops::HelmholtzGridOps{T}) where {T}
    return sparse(-(ops.Dx * ops.Gx + ops.Dy * ops.Gy))
end

function build_real_amg_transfer_cache(A_real::SparseMatrixCSC{T,Int}, builder::PartialHelmholtzSinCosBuilder) where {T}
    # A_setup = builder.amg_setup_reg_eps > 0 ?
    #     A_real + spdiagm(0 => fill(T(builder.amg_setup_reg_eps), size(A_real, 1))) :
    #     A_real

    ml = AlgebraicMultigrid.ruge_stuben(
        sparse(A_real);
        max_levels = builder.amg_max_levels,
        max_coarse = builder.amg_max_coarse,
    )

    ntrans = length(ml.levels)
    P_levels = Vector{SparseMatrixCSC{T,Int}}(undef, ntrans)
    R_levels = Vector{SparseMatrixCSC{T,Int}}(undef, ntrans)
    Pbig_levels = Vector{SparseMatrixCSC{T,Int}}(undef, ntrans)
    Rbig_levels = Vector{SparseMatrixCSC{T,Int}}(undef, ntrans)

    for l in 1:ntrans
        P = sparse(ml.levels[l].P)
        R = sparse(ml.levels[l].R)
        P_levels[l] = P
        R_levels[l] = R
        Pbig_levels[l] = blockdiag(P, P)
        Rbig_levels[l] = blockdiag(R, R)
    end

    return AMGTransferCache{T}(A_real, P_levels, R_levels, Pbig_levels, Rbig_levels)
end

function pin_sparse_dof(A::SparseMatrixCSC{T,Int}, idx::Int = 1) where {T}
    I, J, V = findnz(A)
    keepI = Int[]
    keepJ = Int[]
    keepV = T[]
    sizehint!(keepI, length(I) + 1)
    sizehint!(keepJ, length(J) + 1)
    sizehint!(keepV, length(V) + 1)
    @inbounds for k in eachindex(I)
        i = I[k]
        j = J[k]
        if i == idx || j == idx
            continue
        end
        push!(keepI, i); push!(keepJ, j); push!(keepV, V[k])
    end
    push!(keepI, idx); push!(keepJ, idx); push!(keepV, one(T))
    return sparse(keepI, keepJ, keepV, size(A,1), size(A,2))
end

function ShiftedAMGWorkspace(::Type{T}, H_levels::Vector{SparseMatrixCSC{T,Int}}) where {T}
    nl = length(H_levels)
    residuals = [zeros(T, size(H_levels[l], 1)) for l in 1:(nl - 1)]
    coarse_bs = [zeros(T, size(H_levels[l + 1], 1)) for l in 1:(nl - 1)]
    coarse_xs = [zeros(T, size(H_levels[l + 1], 1)) for l in 1:(nl - 1)]
    return ShiftedAMGWorkspace{T}(residuals, coarse_bs, coarse_xs)
end

function ScalarAMGWorkspace(::Type{T}, A_levels::Vector{SparseMatrixCSC{T,Int}}) where {T}
    nl = length(A_levels)
    residuals = [zeros(T, size(A_levels[l], 1)) for l in 1:(nl - 1)]
    coarse_bs = [zeros(T, size(A_levels[l + 1], 1)) for l in 1:(nl - 1)]
    coarse_xs = [zeros(T, size(A_levels[l + 1], 1)) for l in 1:(nl - 1)]
    return ScalarAMGWorkspace{T}(residuals, coarse_bs, coarse_xs)
end

function build_scalar_amg_hierarchy(A0::SparseMatrixCSC{T,Int}, builder::PartialHelmholtzSinCosBuilder; pin_dof::Int = 1) where {T}
    A_setup = pin_sparse_dof(A0, pin_dof)
    ml = AlgebraicMultigrid.ruge_stuben(
        sparse(A_setup);
        max_levels = builder.amg_max_levels,
        max_coarse = builder.amg_max_coarse,
    )

    ntrans = length(ml.levels)
    P_levels = Vector{SparseMatrixCSC{T,Int}}(undef, ntrans)
    R_levels = Vector{SparseMatrixCSC{T,Int}}(undef, ntrans)
    A_levels = Vector{SparseMatrixCSC{T,Int}}(undef, ntrans + 1)
    A_levels[1] = A_setup

    for l in 1:ntrans
        P = sparse(ml.levels[l].P)
        R = sparse(ml.levels[l].R)
        P_levels[l] = P
        R_levels[l] = R
        A_levels[l + 1] = sparse(R * A_levels[l] * P)
    end

    Dinv_levels = Vector{Vector{T}}(undef, length(A_levels) - 1)
    for l in 1:length(Dinv_levels)
        d = diag(A_levels[l])
        invd = similar(d)
        @inbounds for i in eachindex(d)
            invd[i] = d[i] == zero(T) ? zero(T) : inv(d[i])
        end
        Dinv_levels[l] = invd
    end

    coarse_factor = robust_sparse_lu(
        A_levels[end];
        reg_eps = builder.helmholtz_reg_eps,
        max_reg_tries = builder.helmholtz_max_reg_tries,
    )
    workspaces = [ScalarAMGWorkspace(T, A_levels) for _ in 1:Threads.maxthreadid()]

    return ScalarAMGHierarchy{T, typeof(coarse_factor)}(
        A_levels, P_levels, R_levels, Dinv_levels, coarse_factor,
        T(builder.amg_jacobi_ω), builder.amg_pre_iters, builder.amg_post_iters,
        builder.amg_cycles_per_apply, workspaces
    )
end

function build_scalar_lu_solver(A0::SparseMatrixCSC{T,Int}, builder::PartialHelmholtzSinCosBuilder; pin_dof::Int = 1) where {T}
    A = pin_sparse_dof(A0, pin_dof)
    factor = robust_sparse_lu(
        A;
        reg_eps = builder.helmholtz_reg_eps,
        max_reg_tries = builder.helmholtz_max_reg_tries,
    )
    return ScalarLUSolver{T, typeof(factor)}(A, factor)
end

function weighted_jacobi_scalar!(x::AbstractVector{T}, A::SparseMatrixCSC{T,Int}, b::AbstractVector{T}, Dinv::AbstractVector{T}, ω::T, niters::Int, tmp::AbstractVector{T}) where {T}
    for _ in 1:niters
        mul!(tmp, A, x)
        @inbounds for i in eachindex(x)
            x[i] += ω * Dinv[i] * (b[i] - tmp[i])
        end
    end
    return x
end

function weighted_gauss_seidel_scalar!(x::AbstractVector{T}, A::SparseMatrixCSC{T,Int}, b::AbstractVector{T}, Dinv::AbstractVector{T}, ω::T, niters::Int, tmp::AbstractVector{T}) where {T}
    n = length(x)
    rows = rowvals(A)
    vals = nonzeros(A)

    for _ in 1:niters
        mul!(tmp, A, x)

        @inbounds for i in 1:n
            δ = ω * Dinv[i] * (b[i] - tmp[i])
            x[i] += δ

            if δ != zero(T)
                for ptr in nzrange(A, i)
                    row = rows[ptr]
                    tmp[row] += vals[ptr] * δ
                end
            end
        end
    end

    return x
end

function scalar_vcycle!(x::AbstractVector{T}, hier::ScalarAMGHierarchy{T}, lvl::Int, b::AbstractVector{T}, ws::ScalarAMGWorkspace{T}) where {T}
    if lvl == length(hier.A_levels)
        ldiv!(x, hier.coarse_factor, b)
        return x
    end
    A = hier.A_levels[lvl]
    tmp = ws.residuals[lvl]
    weighted_jacobi_scalar!(x, A, b, hier.Dinv_levels[lvl], hier.jacobi_ω, hier.pre_iters, tmp)

    mul!(tmp, A, x)
    @inbounds for i in eachindex(tmp)
        tmp[i] = b[i] - tmp[i]
    end

    coarse_b = ws.coarse_bs[lvl]
    mul!(coarse_b, hier.R_levels[lvl], tmp)

    coarse_x = ws.coarse_xs[lvl]
    fill!(coarse_x, zero(T))
    scalar_vcycle!(coarse_x, hier, lvl + 1, coarse_b, ws)

    mul!(tmp, hier.P_levels[lvl], coarse_x)
    @inbounds for i in eachindex(x)
        x[i] += tmp[i]
    end

    weighted_jacobi_scalar!(x, A, b, hier.Dinv_levels[lvl], hier.jacobi_ω, hier.post_iters, tmp)
    return x
end

function scalar_amg_apply!(x::AbstractVector{T}, hier::ScalarAMGHierarchy{T}, b::AbstractVector{T}) where {T}
    fill!(x, zero(T))
    ws = hier.workspaces[Threads.threadid()]
    for _ in 1:hier.cycles_per_apply
        scalar_vcycle!(x, hier, 1, b, ws)
    end
    return x
end

function scalar_solver_apply!(x::AbstractVector{T}, hier::ScalarAMGHierarchy{T}, b::AbstractVector{T}) where {T}
    return scalar_amg_apply!(x, hier, b)
end

function scalar_solver_apply!(x::AbstractVector{T}, solver::ScalarLUSolver{T}, b::AbstractVector{T}) where {T}
    ldiv!(x, solver.factor, b)
    return x
end

function build_shifted_block_matrix(A_real::SparseMatrixCSC{T,Int}, shift_real::T, shift_imag::T) where {T}
    n = size(A_real, 1)
    B = A_real + spdiagm(0 => fill(shift_real, n))
    H = blockdiag(B, B)
    if shift_imag != zero(T)
        H += spdiagm(n => fill(-shift_imag, n), -n => fill(shift_imag, n))
    end
    return sparse(H)
end

function build_shifted_amg_hierarchy(cache::AMGTransferCache{T}, shift_real::T, shift_imag::T, builder::PartialHelmholtzSinCosBuilder) where {T}
    H_levels = Vector{SparseMatrixCSC{T,Int}}(undef, length(cache.Pbig_levels) + 1)
    H_levels[1] = build_shifted_block_matrix(cache.A_real, shift_real, shift_imag)

    for l in 1:length(cache.Pbig_levels)
        H_levels[l + 1] = sparse(cache.Rbig_levels[l] * H_levels[l] * cache.Pbig_levels[l])
    end

    Dinv_levels = Vector{Vector{T}}(undef, length(H_levels) - 1)
    for l in 1:length(Dinv_levels)
        d = diag(H_levels[l])
        invd = similar(d)
        @inbounds for i in eachindex(d)
            invd[i] = d[i] == zero(T) ? zero(T) : inv(d[i])
        end
        Dinv_levels[l] = invd
    end

    coarse_factor = robust_sparse_lu(
        H_levels[end];
        reg_eps = builder.helmholtz_reg_eps,
        max_reg_tries = builder.helmholtz_max_reg_tries,
    )
    workspaces = [ShiftedAMGWorkspace(T, H_levels) for _ in 1:Threads.maxthreadid()]

    return ShiftedAMGHierarchy{T, typeof(coarse_factor)}(
        H_levels, cache.Pbig_levels, cache.Rbig_levels, Dinv_levels, coarse_factor,
        T(builder.amg_jacobi_ω), builder.amg_pre_iters, builder.amg_post_iters,
        builder.amg_cycles_per_apply, workspaces
    )
end

function build_shifted_lu_solver(A_real::SparseMatrixCSC{T,Int}, shift_real::T, shift_imag::T, builder::PartialHelmholtzSinCosBuilder) where {T}
    H = build_shifted_block_matrix(A_real, shift_real, shift_imag)
    factor = robust_sparse_lu(
        H;
        reg_eps = builder.helmholtz_reg_eps,
        max_reg_tries = builder.helmholtz_max_reg_tries,
    )
    return ShiftedLUSolver{T, typeof(factor)}(H, factor)
end

function weighted_jacobi!(x::AbstractVector{T}, A::SparseMatrixCSC{T,Int}, b::AbstractVector{T}, Dinv::AbstractVector{T}, ω::T, niters::Int, tmp::AbstractVector{T}) where {T}
    for _ in 1:niters
        mul!(tmp, A, x)
        @inbounds for i in eachindex(x)
            x[i] += ω * Dinv[i] * (b[i] - tmp[i])
        end
    end
    return x
end

function weighted_gauss_seidel!(x::AbstractVector{T}, A::SparseMatrixCSC{T,Int}, b::AbstractVector{T}, Dinv::AbstractVector{T}, ω::T, niters::Int, tmp::AbstractVector{T}) where {T}
    n = length(x)

    for _ in 1:niters
        mul!(tmp, A, x)

        @inbounds for i in 1:n
            δ = ω * Dinv[i] * (b[i] - tmp[i])
            x[i] += δ

            if δ != zero(T)
                for ptr in nzrange(A, i)
                    row = rowvals(A)[ptr]
                    tmp[row] += nonzeros(A)[ptr] * δ
                end
            end
        end
    end

    return x
end

function shifted_vcycle!(x::AbstractVector{T}, hier::ShiftedAMGHierarchy{T}, lvl::Int, b::AbstractVector{T}, ws::ShiftedAMGWorkspace{T}) where {T}
    if lvl == length(hier.H_levels)
        ldiv!(x, hier.coarse_factor, b)
        return x
    end

    H = hier.H_levels[lvl]
    tmp = ws.residuals[lvl]
    weighted_jacobi!(x, H, b, hier.Dinv_levels[lvl], hier.jacobi_ω, hier.pre_iters, tmp)

    mul!(tmp, H, x)
    @inbounds for i in eachindex(tmp)
        tmp[i] = b[i] - tmp[i]
    end

    coarse_b = ws.coarse_bs[lvl]
    mul!(coarse_b, hier.Rbig_levels[lvl], tmp)

    coarse_x = ws.coarse_xs[lvl]
    fill!(coarse_x, zero(T))
    shifted_vcycle!(coarse_x, hier, lvl + 1, coarse_b, ws)

    mul!(tmp, hier.Pbig_levels[lvl], coarse_x)
    @inbounds for i in eachindex(x)
        x[i] += tmp[i]
    end

    weighted_jacobi!(x, H, b, hier.Dinv_levels[lvl], hier.jacobi_ω, hier.post_iters, tmp)
    return x
end

function shifted_amg_apply!(x::AbstractVector{T}, hier::ShiftedAMGHierarchy{T}, b::AbstractVector{T}) where {T}
    fill!(x, zero(T))
    ws = hier.workspaces[Threads.threadid()]
    for _ in 1:hier.cycles_per_apply
        shifted_vcycle!(x, hier, 1, b, ws)
    end
    return x
end

function shifted_solver_apply!(x::AbstractVector{T}, hier::ShiftedAMGHierarchy{T}, b::AbstractVector{T}) where {T}
    return shifted_amg_apply!(x, hier, b)
end

function shifted_solver_apply!(x::AbstractVector{T}, solver::ShiftedLUSolver{T}, b::AbstractVector{T}) where {T}
    ldiv!(x, solver.factor, b)
    return x
end

function normalize_shifted_rhs!(rhs_s::AbstractVector{T}, rhs_c::AbstractVector{T}, α::T, γ::T) where {T}
    den = α * α + γ * γ
    invden = inv(den)
    @inbounds for i in eachindex(rhs_s)
        rs = rhs_s[i]
        rc = rhs_c[i]
        rhs_s[i] = (α * rs - γ * rc) * invden
        rhs_c[i] = (α * rc + γ * rs) * invden
    end
    return nothing
end

function physical_prec_drag_baseline(
    builder::PartialHelmholtzSinCosBuilder,
    p::Params2DHarm,
    ::Type{T},
) where {T}
    return T(2) * T(builder.C_prec) * T(p.r) * T(abs(p.U))
end

function frozen_prec_drag(builder::PartialHelmholtzSinCosBuilder, p::Params2DHarm, ::Type{T}) where {T}
    drag = max(physical_prec_drag_baseline(builder, p, T), T(builder.prec_drag_floor))
    builder.last_effective_drag = Float64(drag)
    return drag
end

function build_sincos_helmholtz_block(
    m::Int,
    p::Params2DHarm,
    ops::HelmholtzGridOps{T},
    A_real::SparseMatrixCSC{T,Int},
    cache::Union{Nothing,AMGTransferCache{T}},
    builder::PartialHelmholtzSinCosBuilder,
) where {T}
    ω = T(m - 1) * T(p.σ)
    ω > zero(T) || error("Helmholtz block should only be built for positive harmonic frequencies")

    D0 = ops.D0u[1]
    if any(!isapprox(v, D0; atol = 1e-12, rtol = 1e-12) for v in ops.D0u) ||
       any(!isapprox(v, D0; atol = 1e-12, rtol = 1e-12) for v in ops.D0v)
        error("Normalized AMG Helmholtz blocks require constant D0 in the Helmholtz model.")
    end

    drag = frozen_prec_drag(builder, p, T)
    den = drag^2 + (ω * D0)^2
    a = drag / den
    b = (ω * D0) / den
    pcoef = D0 * a
    qcoef = D0 * b
    α = T(p.g) * D0 * pcoef
    γ = T(p.g) * D0 * qcoef
    shift_den = α * α + γ * γ
    k2 = ω * γ / shift_den
    β  = ω * α / shift_den
    solver = validate_reduced_solver(builder.reduced_solver)
    hierarchy = solver == :amg ?
        build_shifted_amg_hierarchy(cache::AMGTransferCache{T}, -k2, β, builder) :
        build_shifted_lu_solver(A_real, -k2, β, builder)

    return SinCosHelmholtzHarmonicBlock{T, typeof(hierarchy)}(m, α, γ, hierarchy, a, b, pcoef, qcoef)
end

function build_mean_mode_amg_block(p::Params2DHarm, ops::HelmholtzGridOps{T}, builder::PartialHelmholtzSinCosBuilder) where {T}
    drag = frozen_prec_drag(builder, p, T)
    coeffx = (ops.D0u .* ops.gDu) ./ drag
    coeffy = (ops.D0v .* ops.gDv) ./ drag
    Amean = sparse(-(ops.Dx * spdiagm(0 => coeffx) * ops.Gx + ops.Dy * spdiagm(0 => coeffy) * ops.Gy))
    solver = validate_reduced_solver(builder.reduced_solver)
    hierarchy = solver == :amg ?
        build_scalar_amg_hierarchy(Amean, builder; pin_dof = 1) :
        build_scalar_lu_solver(Amean, builder; pin_dof = 1)
    return MeanModeAMGBlock{T, typeof(hierarchy)}(drag, hierarchy, p.nface)
end

function prepare_reduced_mode_rhs!(y, x, P::PartialHelmholtzSinCosPrec{T}, blk::SinCosHelmholtzHarmonicBlock{T}, sc::PartialHelmholtzApplyScratch{T}) where {T}
    m = blk.m
    n = P.nface

    @inbounds for face in 1:n
        sc.fu_s[face] = x[_prec_idx(P, m, FIELD_u_s, face)]
        sc.fu_c[face] = x[_prec_idx(P, m, FIELD_u_c, face)]
        sc.fv_s[face] = x[_prec_idx(P, m, FIELD_v_s, face)]
        sc.fv_c[face] = x[_prec_idx(P, m, FIELD_v_c, face)]
        sc.fz_s[face] = x[_prec_idx(P, m, FIELD_z_s, face)]
        sc.fz_c[face] = x[_prec_idx(P, m, FIELD_z_c, face)]

        y[_prec_idx(P, m, FIELD_sU_s, face)] = x[_prec_idx(P, m, FIELD_sU_s, face)]
        y[_prec_idx(P, m, FIELD_sU_c, face)] = x[_prec_idx(P, m, FIELD_sU_c, face)]
        y[_prec_idx(P, m, FIELD_sV_s, face)] = x[_prec_idx(P, m, FIELD_sV_s, face)]
        y[_prec_idx(P, m, FIELD_sV_c, face)] = x[_prec_idx(P, m, FIELD_sV_c, face)]
    end

    rhs_s = @view sc.rhs2[1:n]
    rhs_c = @view sc.rhs2[n + 1:2n]
    copyto!(rhs_s, sc.fz_s)
    copyto!(rhs_c, sc.fz_c)

    @inbounds for face in 1:n
        sc.tmp3[face] = blk.pcoef * sc.fu_s[face] + blk.qcoef * sc.fu_c[face]
    end
    mul!(sc.tmp4, P.ops.Dx, sc.tmp3)
    @inbounds for face in 1:n
        rhs_s[face] -= sc.tmp4[face]
    end

    @inbounds for face in 1:n
        sc.tmp3[face] = -blk.qcoef * sc.fu_s[face] + blk.pcoef * sc.fu_c[face]
    end
    mul!(sc.tmp4, P.ops.Dx, sc.tmp3)
    @inbounds for face in 1:n
        rhs_c[face] -= sc.tmp4[face]
    end

    @inbounds for face in 1:n
        sc.tmp3[face] = blk.pcoef * sc.fv_s[face] + blk.qcoef * sc.fv_c[face]
    end
    mul!(sc.tmp4, P.ops.Dy, sc.tmp3)
    @inbounds for face in 1:n
        rhs_s[face] -= sc.tmp4[face]
    end

    @inbounds for face in 1:n
        sc.tmp3[face] = -blk.qcoef * sc.fv_s[face] + blk.pcoef * sc.fv_c[face]
    end
    mul!(sc.tmp4, P.ops.Dy, sc.tmp3)
    @inbounds for face in 1:n
        rhs_c[face] -= sc.tmp4[face]
    end

    normalize_shifted_rhs!(rhs_s, rhs_c, blk.α, blk.γ)
    return nothing
end

function recover_reduced_mode_solution!(y, P::PartialHelmholtzSinCosPrec{T}, blk::SinCosHelmholtzHarmonicBlock{T}, sc::PartialHelmholtzApplyScratch{T}) where {T}
    m = blk.m
    n = P.nface
    z_s = @view sc.z2[1:n]
    z_c = @view sc.z2[n + 1:2n]

    mul!(sc.gradx_s, P.ops.Gx, z_s)
    mul!(sc.gradx_c, P.ops.Gx, z_c)
    mul!(sc.grady_s, P.ops.Gy, z_s)
    mul!(sc.grady_c, P.ops.Gy, z_c)

    @inbounds for face in 1:n
        rhsu_s = sc.fu_s[face] - P.ops.gDu[face] * sc.gradx_s[face]
        rhsu_c = sc.fu_c[face] - P.ops.gDu[face] * sc.gradx_c[face]
        rhsv_s = sc.fv_s[face] - P.ops.gDv[face] * sc.grady_s[face]
        rhsv_c = sc.fv_c[face] - P.ops.gDv[face] * sc.grady_c[face]

        sc.us[face] =  blk.a * rhsu_s + blk.b * rhsu_c
        sc.uc[face] = -blk.b * rhsu_s + blk.a * rhsu_c
        sc.vs[face] =  blk.a * rhsv_s + blk.b * rhsv_c
        sc.vc[face] = -blk.b * rhsv_s + blk.a * rhsv_c
    end

    @inbounds for face in 1:n
        y[_prec_idx(P, m, FIELD_u_s, face)] = sc.us[face]
        y[_prec_idx(P, m, FIELD_u_c, face)] = sc.uc[face]
        y[_prec_idx(P, m, FIELD_v_s, face)] = sc.vs[face]
        y[_prec_idx(P, m, FIELD_v_c, face)] = sc.vc[face]
        y[_prec_idx(P, m, FIELD_z_s, face)] = z_s[face]
        y[_prec_idx(P, m, FIELD_z_c, face)] = z_c[face]
    end
    return nothing
end

function apply_mean_mode!(y, x, P::PartialHelmholtzSinCosPrec{T}, blk::MeanModeAMGBlock{T}, sc::PartialHelmholtzApplyScratch{T}) where {T}
    n = P.nface
    m = 1

    @inbounds for face in 1:n
        sc.fu_s[face] = x[_prec_idx(P, m, FIELD_u_s, face)]
        sc.fu_c[face] = x[_prec_idx(P, m, FIELD_u_c, face)]
        sc.fv_s[face] = x[_prec_idx(P, m, FIELD_v_s, face)]
        sc.fv_c[face] = x[_prec_idx(P, m, FIELD_v_c, face)]
        sc.fz_s[face] = x[_prec_idx(P, m, FIELD_z_s, face)]
        sc.fz_c[face] = x[_prec_idx(P, m, FIELD_z_c, face)]

        y[_prec_idx(P, m, FIELD_u_s, face)] = sc.fu_s[face]
        y[_prec_idx(P, m, FIELD_v_s, face)] = sc.fv_s[face]
        y[_prec_idx(P, m, FIELD_z_s, face)] = sc.fz_s[face]
        y[_prec_idx(P, m, FIELD_sU_s, face)] = x[_prec_idx(P, m, FIELD_sU_s, face)]
        y[_prec_idx(P, m, FIELD_sU_c, face)] = x[_prec_idx(P, m, FIELD_sU_c, face)]
        y[_prec_idx(P, m, FIELD_sV_s, face)] = x[_prec_idx(P, m, FIELD_sV_s, face)]
        y[_prec_idx(P, m, FIELD_sV_c, face)] = x[_prec_idx(P, m, FIELD_sV_c, face)]
    end

    @inbounds for face in 1:n
        sc.tmp1[face] = (P.ops.D0u[face] / blk.drag) * sc.fu_c[face]
    end
    mul!(sc.tmp3, P.ops.Dx, sc.tmp1)

    @inbounds for face in 1:n
        sc.tmp2[face] = (P.ops.D0v[face] / blk.drag) * sc.fv_c[face]
    end
    mul!(sc.tmp4, P.ops.Dy, sc.tmp2)

    rhs = @view sc.rhs2[1:n]
    @inbounds for face in 1:n
        rhs[face] = sc.fz_c[face] - sc.tmp3[face] - sc.tmp4[face]
    end
    rhs[1] = sc.fz_c[1]

    z = @view sc.z2[1:n]
    scalar_solver_apply!(z, blk.hierarchy, rhs)

    mul!(sc.gradx_c, P.ops.Gx, z)
    mul!(sc.grady_c, P.ops.Gy, z)
    @inbounds for face in 1:n
        y[_prec_idx(P, m, FIELD_u_c, face)] = (sc.fu_c[face] - P.ops.gDu[face] * sc.gradx_c[face]) / blk.drag
        y[_prec_idx(P, m, FIELD_v_c, face)] = (sc.fv_c[face] - P.ops.gDv[face] * sc.grady_c[face]) / blk.drag
        y[_prec_idx(P, m, FIELD_z_c, face)] = z[face]
    end
    return nothing
end

function apply_helmholtz_mode!(y, x, P::PartialHelmholtzSinCosPrec{T}, blk::SinCosHelmholtzHarmonicBlock{T}, sc::PartialHelmholtzApplyScratch{T}) where {T}
    prepare_reduced_mode_rhs!(y, x, P, blk, sc)
    shifted_solver_apply!(sc.z2, blk.hierarchy, sc.rhs2)
    recover_reduced_mode_solution!(y, P, blk, sc)
    return nothing
end

function build_partial_helmholtz_sincos_prec(builder::PartialHelmholtzSinCosBuilder)
    p = builder.p
    T = Float64

    solver = validate_reduced_solver(builder.reduced_solver)
    builder.constant_D0_in_helmholtz || error("This preconditioner requires constant_D0_in_helmholtz = true.")

    eff_drag = frozen_prec_drag(builder, p, T)

    ops = build_helmholtz_grid_ops(
        p;
        use_constant_D0 = builder.constant_D0_in_helmholtz,
        D0_const = builder.constant_D0_value,
    )
    A_real = build_real_helmholtz_matrix(ops)
    transfer_cache = solver == :amg ? build_real_amg_transfer_cache(A_real, builder) : nothing
    mean_block = build_mean_mode_amg_block(p, ops, builder)

    if p.Kloc >= 2
        first_helm = build_sincos_helmholtz_block(2, p, ops, A_real, transfer_cache, builder)
        helm_blocks = Vector{Union{Nothing, typeof(first_helm)}}(undef, p.Kloc)
        helm_blocks[1] = nothing
        helm_blocks[2] = first_helm
        for m in 3:p.Kloc
            helm_blocks[m] = build_sincos_helmholtz_block(m, p, ops, A_real, transfer_cache, builder)
        end
    else
        helm_blocks = Union{Nothing, Nothing}[nothing]
    end

    scratch = [PartialHelmholtzApplyScratch(T, p.nface) for _ in 1:Threads.maxthreadid()]

    if builder.verbose
        baseline_drag = physical_prec_drag_baseline(builder, p, T)
        label = reduced_solver_label(solver)
        println("Constructed preconditioner (mean-mode Poisson-$label + oscillatory Helmholtz-$label):")
        println("  effective drag = $(eff_drag)")
        println("  physical baseline drag = $(baseline_drag) = C_prec * (3r|U|/π)")
        println("  C_prec = $(builder.C_prec)")
        println("  reduced_solver = :$(solver)")
        println("  mean harmonic 0 -> Poisson-$label")
        for m in 2:p.Kloc
            k = m - 1
            println("  harmonic $k -> Helmholtz-$label")
        end
    end

    return PartialHelmholtzSinCosPrec{T, typeof(mean_block), typeof(helm_blocks)}(
        ops,
        mean_block,
        helm_blocks,
        scratch,
        p.nface,
        p.harm_block,
        p.Kloc,
        p.Kloc * p.harm_block,
    )
end

function LinearAlgebra.ldiv!(y::AbstractVector{T}, P::PartialHelmholtzSinCosPrec{T}, x::AbstractVector{T}) where {T}
    length(y) == P.n || throw(DimensionMismatch("length(y) = $(length(y)) but preconditioner size is $(P.n)"))
    length(x) == P.n || throw(DimensionMismatch("length(x) = $(length(x)) but preconditioner size is $(P.n)"))

    if P.Kloc <= 8
        for m in 1:P.Kloc
            sc = P.scratch[1]
            if m == 1
                apply_mean_mode!(y, x, P, P.mean_block, sc)
            else
                apply_helmholtz_mode!(y, x, P, P.helm_blocks[m], sc)
            end
        end
    else
        Threads.@threads for m in 1:P.Kloc
            sc = P.scratch[Threads.threadid()]
            if m == 1
                apply_mean_mode!(y, x, P, P.mean_block, sc)
            else
                apply_helmholtz_mode!(y, x, P, P.helm_blocks[m], sc)
            end
        end
    end
    return y
end

function LinearAlgebra.ldiv!(P::PartialHelmholtzSinCosPrec{T}, x::AbstractVector{T}) where {T}
    tmp = similar(x)
    ldiv!(tmp, P, x)
    copyto!(x, tmp)
    return x
end

function (builder::PartialHelmholtzSinCosBuilder)(A, _)
    rebuild_each_time = !builder.reuse_preconditioner
    if builder.cache === nothing || rebuild_each_time
        builder.cache = build_partial_helmholtz_sincos_prec(builder)
    end
    return builder.cache, I
end

mutable struct GMRESNewtonTrace{T}
    residuals::Vector{T}
    newton_starts::Vector{Int}
end

GMRESNewtonTrace(::Type{T}) where {T} = GMRESNewtonTrace{T}(T[], Int[])

function make_gmres_newton_callback(trace::GMRESNewtonTrace{T}) where {T}
    function cb(workspace)
        r = workspace.stats.residuals
        if workspace.inner_iter == 1 && length(r) == 2
            push!(trace.newton_starts, length(trace.residuals) + 1)
        end
        push!(trace.residuals, T(r[end]))
        return false
    end
    return cb
end

function gmres_iters_per_newton(trace::GMRESNewtonTrace)
    starts = trace.newton_starts
    total  = length(trace.residuals)
    counts = zeros(Int, length(starts))
    @inbounds for i in 1:length(starts)
        s = starts[i]
        e = (i < length(starts)) ? (starts[i + 1] - 1) : total
        counts[i] = e - s + 1
    end
    return counts
end

function plot_gmres_across_newton(
    trace::GMRESNewtonTrace;
    savepath::Union{Nothing,String} = "gmres_across_newton.png",
)
    total = length(trace.residuals)
    total == 0 && error("No GMRES iterations were logged.")

    counts = gmres_iters_per_newton(trace)
    x = collect(1:total)

    p1 = plot(
        x, trace.residuals;
        xlabel = "Cumulative GMRES iteration",
        ylabel = "GMRES residual norm",
        yscale = :log10,
        lw = 2,
        label = "GMRES residual",
        title = "GMRES residual history across Newton iterations",
        margin = 5mm,
    )

    for (k, s) in enumerate(trace.newton_starts)
        vline!(
            p1, [s - 0.5];
            color = :black,
            linestyle = :dash,
            alpha = 0.45,
            label = (k == 1 ? "Newton start" : nothing),
        )
    end

    p2 = bar(
        1:length(counts), counts;
        xlabel = "Newton iteration",
        ylabel = "GMRES iterations",
        title = "GMRES iterations per Newton step",
        label = false,
        margin = 5mm,
    )

    fig = plot(p1, p2; layout = (2, 1), size = (1100, 800))

    if savepath !== nothing
        savefig(fig, savepath)
        println("Saved GMRES/Newton plot to $savepath")
    end

    println("\nGMRES iterations per Newton step:")
    for (k, c) in enumerate(counts)
        println("  Newton $(lpad(k, 2)): $c")
    end
    println("  Total GMRES iterations: $total")

    return fig, counts
end

function save_solution_vector(filename::AbstractString, y::AbstractVector)
    serialize(filename, Vector(y))
    return filename
end

function load_solution_vector(filename::AbstractString, nexpected::Int)
    isfile(filename) || error("Saved solution file not found: $filename")
    y = deserialize(filename)
    y isa AbstractVector || error("Saved solution in $filename is not a vector.")
    length(y) == nexpected || error("Saved solution length $(length(y)) does not match expected length $nexpected.")
    return Vector{Float64}(y)
end

