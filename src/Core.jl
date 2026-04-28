function make_time_basis(Kpos::Int, σ::Float64; Nt::Int = 512)
    Kloc = Kpos + 1
    Tper = 2π / σ
    t = range(0.0, Tper; length = Nt + 1)[1:Nt]

    S = zeros(Float64, Kloc, Nt)
    C = ones(Float64, Kloc, Nt)

    @inbounds for m in 2:Kloc
        k = m - 1
        @views S[m, :] .= sin.(k * σ .* t)
        @views C[m, :] .= cos.(k * σ .* t)
    end
    return Nt, S, C, t
end

function project_signal_to_basis(sig_t::AbstractVector{<:Real}, S::AbstractMatrix, C::AbstractMatrix, Kpos::Int)
    Kloc = Kpos + 1
    Nt = length(sig_t)

    s = zeros(Float64, Kloc)
    c = zeros(Float64, Kloc)

    invNt = 1.0 / Nt
    c[1] = sum(sig_t) * invNt
    s[1] = 0.0

    scale = 2.0 * invNt
    @inbounds for m in 2:Kloc
        s[m] = scale * sum(sig_t[n] * S[m, n] for n in 1:Nt)
        c[m] = scale * sum(sig_t[n] * C[m, n] for n in 1:Nt)
    end
    return s, c
end

function load_series_padded!(S2, C2, s, c, Kloc::Int; add_s1 = 0.0)
    T = eltype(S2)
    z = zero(T)
    fill!(S2, z)
    fill!(C2, z)
    @inbounds for m in 1:Kloc
        k = m - 1
        S2[m] = (k == 0) ? z : T(s[m])
        C2[m] = T(c[m])
    end
    if Kloc >= 2
        S2[2] += T(add_s1)
    end
    return nothing
end

function mul_coeffs!(
    out_s::AbstractVector{Tp},
    out_c::AbstractVector{Tp},
    a_s::AbstractVector{Ta},
    a_c::AbstractVector{Ta},
    b_s::AbstractVector{Tb},
    b_c::AbstractVector{Tb},
    Kpos::Int,
) where {Tp, Ta, Tb}

    @inbounds begin
        fill!(out_s, zero(Tp))
        fill!(out_c, zero(Tp))

        half = Tp(0.5)

        a0 = Tp(a_c[1])
        b0 = Tp(b_c[1])

        out_c[1] = a0 * b0
        for m in 1:Kpos
            mi = m + 1
            out_c[1] += half * (Tp(a_s[mi]) * Tp(b_s[mi]) + Tp(a_c[mi]) * Tp(b_c[mi]))
        end
        out_s[1] = zero(Tp)

        for k in 1:Kpos
            ki = k + 1
            out_c[ki] = a0 * Tp(b_c[ki]) + b0 * Tp(a_c[ki])
            out_s[ki] = a0 * Tp(b_s[ki]) + b0 * Tp(a_s[ki])
        end

        for m in 1:Kpos
            mi = m + 1
            for n in 1:Kpos
                ni = n + 1

                ksum = m + n
                if ksum <= Kpos
                    ki = ksum + 1
                    out_c[ki] += half * (-Tp(a_s[mi]) * Tp(b_s[ni]) + Tp(a_c[mi]) * Tp(b_c[ni]))
                    out_s[ki] += half * ( Tp(a_s[mi]) * Tp(b_c[ni]) + Tp(a_c[mi]) * Tp(b_s[ni]))
                end

                kdiff = abs(m - n)
                if kdiff != 0 && kdiff <= Kpos
                    ki = kdiff + 1
                    out_c[ki] += half * (Tp(a_s[mi]) * Tp(b_s[ni]) + Tp(a_c[mi]) * Tp(b_c[ni]))

                    sgn = (m > n) ? one(Tp) : -one(Tp)
                    out_s[ki] += half * sgn * (Tp(a_s[mi]) * Tp(b_c[ni]) - Tp(a_c[mi]) * Tp(b_s[ni]))
                end
            end
        end
    end

    return nothing
end


mutable struct ResidualScratch{T}
    ζs_face::Vector{T}
    ζc_face::Vector{T}

    us_face::Vector{T}
    uc_face::Vector{T}
    vs_face::Vector{T}
    vc_face::Vector{T}

    ps_face::Vector{T}
    pc_face::Vector{T}

    qxL_s::Vector{T}
    qxL_c::Vector{T}
    qxR_s::Vector{T}
    qxR_c::Vector{T}
    qyB_s::Vector{T}
    qyB_c::Vector{T}
    qyT_s::Vector{T}
    qyT_c::Vector{T}

    vface_s::Vector{T}
    vface_c::Vector{T}
    uface_s::Vector{T}
    uface_c::Vector{T}

    Ds2::Vector{T}
    Dc2::Vector{T}
    As2::Vector{T}
    Ac2::Vector{T}
    Bs2::Vector{T}
    Bc2::Vector{T}
    out_s2::Vector{T}
    out_c2::Vector{T}
    tmp_s2::Vector{T}
    tmp_c2::Vector{T}

    Ts::Vector{T}
    Tc::Vector{T}
    Pm_s::Vector{T}
    Pm_c::Vector{T}

    drag_u_s::Vector{T}
    drag_u_c::Vector{T}
    drag_v_s::Vector{T}
    drag_v_c::Vector{T}

    u_s_loc::Vector{T}
    u_c_loc::Vector{T}
    v_s_loc::Vector{T}
    v_c_loc::Vector{T}
    z_s_loc::Vector{T}
    z_c_loc::Vector{T}
    su_s_loc::Vector{T}
    su_c_loc::Vector{T}
    sv_s_loc::Vector{T}
    sv_c_loc::Vector{T}

    u_s_nb::Vector{T}
    u_c_nb::Vector{T}
    v_s_nb::Vector{T}
    v_c_nb::Vector{T}
    z_s_nb::Vector{T}
    z_c_nb::Vector{T}

    Ru_s_loc::Vector{T}
    Ru_c_loc::Vector{T}
    Rv_s_loc::Vector{T}
    Rv_c_loc::Vector{T}
    Rz_s_loc::Vector{T}
    Rz_c_loc::Vector{T}
    Rsu_s_loc::Vector{T}
    Rsu_c_loc::Vector{T}
    Rsv_s_loc::Vector{T}
    Rsv_c_loc::Vector{T}

    bg_s::Vector{T}
    bg_c::Vector{T}
end

function ResidualScratch(::Type{T}, Kloc::Int, K2loc::Int, bg_s_f64, bg_c_f64) where {T}
    vK()  = zeros(T, Kloc)
    vK2() = zeros(T, K2loc)

    bgs = Vector{T}(undef, Kloc)
    bgc = Vector{T}(undef, Kloc)
    @inbounds for m in 1:Kloc
        bgs[m] = T(bg_s_f64[m])
        bgc[m] = T(bg_c_f64[m])
    end

    return ResidualScratch{T}(
        vK(), vK(),

        vK(), vK(),
        vK(), vK(),

        vK(), vK(),

        vK(), vK(),
        vK(), vK(),
        vK(), vK(),
        vK(), vK(),

        vK(), vK(),
        vK(), vK(),

        vK2(), vK2(),
        vK2(), vK2(),
        vK2(), vK2(),
        vK2(), vK2(),
        vK2(), vK2(),

        vK(), vK(),
        vK(), vK(),

        vK(), vK(),
        vK(), vK(),

        vK(), vK(),
        vK(), vK(),
        vK(), vK(),
        vK(), vK(),
        vK(), vK(),

        vK(), vK(),
        vK(), vK(),
        vK(), vK(),

        vK(), vK(),
        vK(), vK(),
        vK(), vK(),
        vK(), vK(),
        vK(), vK(),

        bgs, bgc
    )
end

mutable struct Params2DHarm
    Nx::Int
    Ny::Int
    Δx::Float64
    Δy::Float64
    H::Float64
    g::Float64
    r::Float64
    U::Float64
    σ::Float64
    Ef::Float64
    drag_eps::Float64
    hC::Matrix{Float64}

    Kpos::Int
    Kloc::Int
    K2pos::Int
    K2loc::Int

    nface::Int
    nfields::Int
    harm_block::Int

    bg_s::Vector{Float64}
    bg_c::Vector{Float64}

    scratch_cache::IdDict{DataType, Any}
    scratch_lock::ReentrantLock

end

function get_scratch!(p::Params2DHarm, ::Type{T}, slot::Int) where {T}
    max_tid = Threads.maxthreadid()

    vec_any = get(p.scratch_cache, T, nothing)
    if vec_any === nothing
        lock(p.scratch_lock)
        try
            vec_any = get(p.scratch_cache, T, nothing)
            if vec_any === nothing
                v = Vector{ResidualScratch{T}}(undef, max_tid)
                @inbounds for k in 1:max_tid
                    v[k] = ResidualScratch(T, p.Kloc, p.K2loc, p.bg_s, p.bg_c)
                end
                p.scratch_cache[T] = v
                vec_any = v
            end
        finally
            unlock(p.scratch_lock)
        end
    end

    v = vec_any::Vector{ResidualScratch{T}}
    @boundscheck 1 <= slot <= length(v) || throw(BoundsError(v, slot))
    return v[slot]
end

get_scratch!(p::Params2DHarm, ::Type{T}) where {T} = get_scratch!(p, T, Threads.threadid())

function idx_harmonic_major(p::Params2DHarm, m::Int, field::Int, face::Int)
    return (m - 1) * p.harm_block + (field - 1) * p.nface + face
end

function set_coeff!(y, p::Params2DHarm, m::Int, field::Int, i::Int, j::Int, val)
    f = face_idx(i, j, p.Nx)
    y[idx_harmonic_major(p, m, field, f)] = val
    return nothing
end

function load_face_coeffs!(s::AbstractVector, c::AbstractVector, y, p, field_s::Int, field_c::Int, i::Int, j::Int)
    face = face_idx(i, j, p.Nx)
    @inbounds for m in 1:p.Kloc
        s[m] = y[idx_harmonic_major(p, m, field_s, face)]
        c[m] = y[idx_harmonic_major(p, m, field_c, face)]
    end
    return nothing
end

function store_face_coeffs!(F, p, field_s::Int, field_c::Int, i::Int, j::Int, Rs, Rc)
    face = face_idx(i, j, p.Nx)
    @inbounds for m in 1:p.Kloc
        F[idx_harmonic_major(p, m, field_s, face)] = Rs[m]
        F[idx_harmonic_major(p, m, field_c, face)] = Rc[m]
    end
    return nothing
end

function h_on_U_face(p::Params2DHarm, i::Int, j::Int)
    im = im1(i, p.Nx)
    return 0.5 * (p.hC[im, j] + p.hC[i, j])
end

function h_on_V_face(p::Params2DHarm, i::Int, j::Int)
    jm = im1(j, p.Ny)
    return 0.5 * (p.hC[i, jm] + p.hC[i, j])
end

function zeta_on_U_face!(ζs_face, ζc_face, y, p::Params2DHarm, i::Int, j::Int, sc)
    im = im1(i, p.Nx)
    load_face_coeffs!(sc.z_s_nb, sc.z_c_nb, y, p, FIELD_z_s, FIELD_z_c, im, j)
    load_face_coeffs!(sc.z_s_loc, sc.z_c_loc, y, p, FIELD_z_s, FIELD_z_c, i,  j)
    @inbounds for m in 1:p.Kloc
        ζs_face[m] = 0.5 * (sc.z_s_nb[m] + sc.z_s_loc[m])
        ζc_face[m] = 0.5 * (sc.z_c_nb[m] + sc.z_c_loc[m])
    end
    return nothing
end

function zeta_on_V_face!(ζs_face, ζc_face, y, p::Params2DHarm, i::Int, j::Int, sc)
    jm = im1(j, p.Ny)
    load_face_coeffs!(sc.z_s_nb, sc.z_c_nb, y, p, FIELD_z_s, FIELD_z_c, i, jm)
    load_face_coeffs!(sc.z_s_loc, sc.z_c_loc, y, p, FIELD_z_s, FIELD_z_c, i, j)
    @inbounds for m in 1:p.Kloc
        ζs_face[m] = 0.5 * (sc.z_s_nb[m] + sc.z_s_loc[m])
        ζc_face[m] = 0.5 * (sc.z_c_nb[m] + sc.z_c_loc[m])
    end
    return nothing
end

function v_interp_to_U_face!(vs, vc, y, p::Params2DHarm, i::Int, j::Int, sc)
    im = im1(i, p.Nx)
    jp = ip1(j, p.Ny)

    load_face_coeffs!(sc.v_s_loc, sc.v_c_loc, y, p, FIELD_v_s, FIELD_v_c, im, j)
    load_face_coeffs!(sc.v_s_nb,  sc.v_c_nb,  y, p, FIELD_v_s, FIELD_v_c, i,  j)
    @inbounds for m in 1:p.Kloc
        vs[m] = sc.v_s_loc[m] + sc.v_s_nb[m]
        vc[m] = sc.v_c_loc[m] + sc.v_c_nb[m]
    end

    load_face_coeffs!(sc.v_s_loc, sc.v_c_loc, y, p, FIELD_v_s, FIELD_v_c, im, jp)
    load_face_coeffs!(sc.v_s_nb,  sc.v_c_nb,  y, p, FIELD_v_s, FIELD_v_c, i,  jp)
    @inbounds for m in 1:p.Kloc
        vs[m] = 0.25 * (vs[m] + sc.v_s_loc[m] + sc.v_s_nb[m])
        vc[m] = 0.25 * (vc[m] + sc.v_c_loc[m] + sc.v_c_nb[m])
    end
    return nothing
end

function u_interp_to_V_face!(us, uc, y, p::Params2DHarm, i::Int, j::Int, sc)
    ip = ip1(i, p.Nx)
    jm = im1(j, p.Ny)

    load_face_coeffs!(sc.u_s_loc, sc.u_c_loc, y, p, FIELD_u_s, FIELD_u_c, i,  jm)
    load_face_coeffs!(sc.u_s_nb,  sc.u_c_nb,  y, p, FIELD_u_s, FIELD_u_c, ip, jm)
    @inbounds for m in 1:p.Kloc
        us[m] = sc.u_s_loc[m] + sc.u_s_nb[m]
        uc[m] = sc.u_c_loc[m] + sc.u_c_nb[m]
    end

    load_face_coeffs!(sc.u_s_loc, sc.u_c_loc, y, p, FIELD_u_s, FIELD_u_c, i,  j)
    load_face_coeffs!(sc.u_s_nb,  sc.u_c_nb,  y, p, FIELD_u_s, FIELD_u_c, ip, j)
    @inbounds for m in 1:p.Kloc
        us[m] = 0.25 * (us[m] + sc.u_s_loc[m] + sc.u_s_nb[m])
        uc[m] = 0.25 * (uc[m] + sc.u_c_loc[m] + sc.u_c_nb[m])
    end
    return nothing
end

function _face_to_ij(face::Int, Nx::Int)
    j = (face - 1) ÷ Nx + 1
    i = face - (j - 1) * Nx
    return i, j
end

function _chunk_bounds(chunk::Int, nchunks::Int, nface::Int)
    first = ((chunk - 1) * nface) ÷ nchunks + 1
    last  = (chunk * nface) ÷ nchunks
    return first, last
end


function residuals2D!(F, y, p::Params2DHarm)
    Ty = eltype(y)

    Kpos  = p.Kpos
    Kloc  = p.Kloc
    K2pos = p.K2pos
    K2loc = p.K2loc

    Nx = p.Nx
    Ny = p.Ny
    nface = p.nface

    ΔxT = Ty(p.Δx)
    ΔyT = Ty(p.Δy)
    invΔxT  = inv(ΔxT)
    invΔyT  = inv(ΔyT)
    inv2ΔxT = inv(Ty(2) * ΔxT)
    inv2ΔyT = inv(Ty(2) * ΔyT)

    HT  = Ty(p.H)
    gT  = Ty(p.g)
    rT  = Ty(p.r)
    UT  = Ty(p.U)
    σT  = Ty(p.σ)
    EfT = Ty(p.Ef)
    drag_epsT = Ty(p.drag_eps)

    nchunks = min(max(Threads.nthreads(), 1), nface)

    Threads.@threads for chunk in 1:nchunks
        sc = get_scratch!(p, Ty, chunk)
        tid = chunk
        face_lo, face_hi = _chunk_bounds(chunk, nchunks, nface)

        local_udrag = UInt64(0)

        for face in face_lo:face_hi
            i, j = _face_to_ij(face, Nx)
            ip = ip1(i, Nx)
            im = im1(i, Nx)
            jp = ip1(j, Ny)
            jm = im1(j, Ny)

            ζs_face = sc.ζs_face
            ζc_face = sc.ζc_face
            vface_s = sc.vface_s
            vface_c = sc.vface_c

            Ds2    = sc.Ds2
            Dc2    = sc.Dc2
            As2    = sc.As2
            Ac2    = sc.Ac2
            Bs2    = sc.Bs2
            Bc2    = sc.Bc2
            out_s2 = sc.out_s2
            out_c2 = sc.out_c2
            tmp_s2 = sc.tmp_s2
            tmp_c2 = sc.tmp_c2

            Ts   = sc.Ts
            Tc   = sc.Tc
            Pm_s = sc.Pm_s
            Pm_c = sc.Pm_c

            drag_u_s = sc.drag_u_s
            drag_u_c = sc.drag_u_c
            bg_s = sc.bg_s
            bg_c = sc.bg_c

            @inbounds begin
                hU = Ty(h_on_U_face(p, i, j))
                D0 = HT - hU

                zeta_on_U_face!(ζs_face, ζc_face, y, p, i, j, sc)
                fill!(Ds2, zero(Ty)); fill!(Dc2, zero(Ty))
                for m in 1:Kloc
                    Ds2[m] = Ty(ζs_face[m])
                    Dc2[m] = Ty(ζc_face[m])
                end
                Dc2[1] += D0

                load_face_coeffs!(sc.u_s_loc, sc.u_c_loc, y, p, FIELD_u_s, FIELD_u_c, i,  j)
                load_face_coeffs!(sc.u_s_nb,  sc.u_c_nb,  y, p, FIELD_u_s, FIELD_u_c, ip, j)
                load_face_coeffs!(sc.z_s_nb,  sc.z_c_nb,  y, p, FIELD_u_s, FIELD_u_c, im, j)
                load_face_coeffs!(sc.z_s_loc, sc.z_c_loc, y, p, FIELD_u_s, FIELD_u_c, i,  jp)
                load_face_coeffs!(sc.v_s_nb,  sc.v_c_nb,  y, p, FIELD_u_s, FIELD_u_c, i,  jm)

                v_interp_to_U_face!(vface_s, vface_c, y, p, i, j, sc)


                load_face_coeffs!(sc.su_s_loc, sc.su_c_loc, y, p, FIELD_sU_s, FIELD_sU_c, i, j)

                load_series_padded!(As2, Ac2, sc.u_s_loc, sc.u_c_loc, Kloc; add_s1 = UT)
                mul_coeffs!(out_s2, out_c2, As2, Ac2, As2, Ac2, K2pos)

                load_series_padded!(Bs2, Bc2, vface_s, vface_c, Kloc)
                mul_coeffs!(tmp_s2, tmp_c2, Bs2, Bc2, Bs2, Bc2, K2pos)

                load_series_padded!(As2, Ac2, sc.su_s_loc, sc.su_c_loc, Kloc)
                mul_coeffs!(Bs2, Bc2, As2, Ac2, As2, Ac2, K2pos)

                sc.Rsu_s_loc[1] = Ty(sc.su_s_loc[1])
                sc.Rsu_c_loc[1] = Bc2[1] - out_c2[1] - tmp_c2[1] - drag_epsT
                for m in 2:Kloc
                    sc.Rsu_s_loc[m] = Bs2[m] - out_s2[m] - tmp_s2[m]
                    sc.Rsu_c_loc[m] = Bc2[m] - out_c2[m] - tmp_c2[m]
                end

                load_series_padded!(As2, Ac2, sc.u_s_loc, sc.u_c_loc, Kloc; add_s1 = UT)
                load_series_padded!(Bs2, Bc2, sc.su_s_loc, sc.su_c_loc, Kloc)
                mul_coeffs!(out_s2, out_c2, As2, Ac2, Bs2, Bc2, K2pos)

                load_series_padded!(As2, Ac2, bg_s, bg_c, Kloc)
                mul_coeffs!(tmp_s2, tmp_c2, Ds2, Dc2, As2, Ac2, K2pos)

                for m in 1:Kloc
                    drag_u_s[m] = rT * out_s2[m] - tmp_s2[m]
                    drag_u_c[m] = rT * out_c2[m] - tmp_c2[m]
                end


                fill!(As2, zero(Ty)); fill!(Ac2, zero(Ty))
                for m in 2:Kloc
                    k = m - 1
                    ω = Ty(k) * σT
                    As2[m] = (-ω) * Ty(sc.u_c_loc[m])
                    Ac2[m] = ( ω) * Ty(sc.u_s_loc[m])
                end
                mul_coeffs!(out_s2, out_c2, Ds2, Dc2, As2, Ac2, K2pos)
                for m in 1:Kloc
                    Ts[m] = out_s2[m]
                    Tc[m] = out_c2[m]
                end

                load_face_coeffs!(sc.z_s_nb, sc.z_c_nb, y, p, FIELD_z_s, FIELD_z_c, im, j)
                load_face_coeffs!(sc.z_s_loc, sc.z_c_loc, y, p, FIELD_z_s, FIELD_z_c, i,  j)
                fill!(As2, zero(Ty)); fill!(Ac2, zero(Ty))
                for m in 1:Kloc
                    As2[m] = (Ty(sc.z_s_loc[m]) - Ty(sc.z_s_nb[m])) * invΔxT
                    Ac2[m] = (Ty(sc.z_c_loc[m]) - Ty(sc.z_c_nb[m])) * invΔxT
                end
                mul_coeffs!(out_s2, out_c2, Ds2, Dc2, As2, Ac2, K2pos)
                for m in 1:Kloc
                    Pm_s[m] = out_s2[m]
                    Pm_c[m] = out_c2[m]
                end

                load_face_coeffs!(sc.u_s_nb, sc.u_c_nb, y, p, FIELD_u_s, FIELD_u_c, ip, j)
                load_face_coeffs!(sc.z_s_nb, sc.z_c_nb, y, p, FIELD_u_s, FIELD_u_c, im, j)
                load_face_coeffs!(sc.z_s_loc, sc.z_c_loc, y, p, FIELD_u_s, FIELD_u_c, i,  jp)
                load_face_coeffs!(sc.v_s_nb, sc.v_c_nb, y, p, FIELD_u_s, FIELD_u_c, i,  jm)

                fill!(As2, zero(Ty)); fill!(Ac2, zero(Ty))
                fill!(Bs2, zero(Ty)); fill!(Bc2, zero(Ty))
                for m in 1:Kloc
                    As2[m] = (Ty(sc.u_s_nb[m]) - Ty(sc.z_s_nb[m])) * inv2ΔxT
                    Ac2[m] = (Ty(sc.u_c_nb[m]) - Ty(sc.z_c_nb[m])) * inv2ΔxT
                    Bs2[m] = (Ty(sc.z_s_loc[m]) - Ty(sc.v_s_nb[m])) * inv2ΔyT
                    Bc2[m] = (Ty(sc.z_c_loc[m]) - Ty(sc.v_c_nb[m])) * inv2ΔyT
                end

                load_series_padded!(tmp_s2, tmp_c2, sc.u_s_loc, sc.u_c_loc, Kloc; add_s1 = UT)
                mul_coeffs!(out_s2, out_c2, tmp_s2, tmp_c2, As2, Ac2, K2pos)

                load_series_padded!(tmp_s2, tmp_c2, vface_s, vface_c, Kloc)
                mul_coeffs!(As2, Ac2, tmp_s2, tmp_c2, Bs2, Bc2, K2pos)

                fill!(Bs2, zero(Ty)); fill!(Bc2, zero(Ty))
                for m in 1:K2loc
                    Bs2[m] = out_s2[m] + As2[m]
                    Bc2[m] = out_c2[m] + Ac2[m]
                end
                mul_coeffs!(out_s2, out_c2, Ds2, Dc2, Bs2, Bc2, K2pos)

                load_series_padded!(Bs2, Bc2, vface_s, vface_c, Kloc)
                mul_coeffs!(tmp_s2, tmp_c2, Ds2, Dc2, Bs2, Bc2, K2pos)

                fill!(Bs2, zero(Ty)); fill!(Bc2, zero(Ty))

                for m in 1:Kloc
                    k = m - 1
                    if k == 0
                        sc.Ru_s_loc[m] = Ty(sc.u_s_loc[m])
                    else
                        sc.Ru_s_loc[m] = Ts[m] + out_s2[m] + drag_u_s[m] + gT * Pm_s[m] - EfT * tmp_s2[m] + Bs2[m]
                    end
                    sc.Ru_c_loc[m] = Tc[m] + out_c2[m] + drag_u_c[m] + gT * Pm_c[m] - EfT * tmp_c2[m] + Bc2[m]
                end

                store_face_coeffs!(F, p, FIELD_u_s,  FIELD_u_c,  i, j, sc.Ru_s_loc,  sc.Ru_c_loc)
                store_face_coeffs!(F, p, FIELD_sU_s, FIELD_sU_c, i, j, sc.Rsu_s_loc, sc.Rsu_c_loc)
            end
        end

    end

    Threads.@threads for chunk in 1:nchunks
        sc = get_scratch!(p, Ty, chunk)
        tid = chunk
        face_lo, face_hi = _chunk_bounds(chunk, nchunks, nface)

        local_vdrag = UInt64(0)

        for face in face_lo:face_hi
            i, j = _face_to_ij(face, Nx)
            ip = ip1(i, Nx)
            im = im1(i, Nx)
            jp = ip1(j, Ny)
            jm = im1(j, Ny)

            ζs_face = sc.ζs_face
            ζc_face = sc.ζc_face
            uface_s = sc.uface_s
            uface_c = sc.uface_c

            Ds2    = sc.Ds2
            Dc2    = sc.Dc2
            As2    = sc.As2
            Ac2    = sc.Ac2
            Bs2    = sc.Bs2
            Bc2    = sc.Bc2
            out_s2 = sc.out_s2
            out_c2 = sc.out_c2
            tmp_s2 = sc.tmp_s2
            tmp_c2 = sc.tmp_c2

            Ts   = sc.Ts
            Tc   = sc.Tc
            Pm_s = sc.Pm_s
            Pm_c = sc.Pm_c

            drag_v_s = sc.drag_v_s
            drag_v_c = sc.drag_v_c

            @inbounds begin
                hV = Ty(h_on_V_face(p, i, j))
                D0 = HT - hV

                zeta_on_V_face!(ζs_face, ζc_face, y, p, i, j, sc)
                fill!(Ds2, zero(Ty)); fill!(Dc2, zero(Ty))
                for m in 1:Kloc
                    Ds2[m] = Ty(ζs_face[m])
                    Dc2[m] = Ty(ζc_face[m])
                end
                Dc2[1] += D0

                load_face_coeffs!(sc.v_s_loc, sc.v_c_loc, y, p, FIELD_v_s, FIELD_v_c, i,  j)
                load_face_coeffs!(sc.v_s_nb,  sc.v_c_nb,  y, p, FIELD_v_s, FIELD_v_c, ip, j)
                load_face_coeffs!(sc.z_s_nb,  sc.z_c_nb,  y, p, FIELD_v_s, FIELD_v_c, im, j)
                load_face_coeffs!(sc.z_s_loc, sc.z_c_loc, y, p, FIELD_v_s, FIELD_v_c, i,  jp)
                load_face_coeffs!(sc.u_s_nb,  sc.u_c_nb,  y, p, FIELD_v_s, FIELD_v_c, i,  jm)

                u_interp_to_V_face!(uface_s, uface_c, y, p, i, j, sc)


                load_face_coeffs!(sc.sv_s_loc, sc.sv_c_loc, y, p, FIELD_sV_s, FIELD_sV_c, i, j)

                load_series_padded!(As2, Ac2, uface_s, uface_c, Kloc; add_s1 = UT)
                mul_coeffs!(out_s2, out_c2, As2, Ac2, As2, Ac2, K2pos)

                load_series_padded!(Bs2, Bc2, sc.v_s_loc, sc.v_c_loc, Kloc)
                mul_coeffs!(tmp_s2, tmp_c2, Bs2, Bc2, Bs2, Bc2, K2pos)

                load_series_padded!(As2, Ac2, sc.sv_s_loc, sc.sv_c_loc, Kloc)
                mul_coeffs!(Bs2, Bc2, As2, Ac2, As2, Ac2, K2pos)

                sc.Rsv_s_loc[1] = Ty(sc.sv_s_loc[1])
                sc.Rsv_c_loc[1] = Bc2[1] - out_c2[1] - tmp_c2[1] - drag_epsT
                for m in 2:Kloc
                    sc.Rsv_s_loc[m] = Bs2[m] - out_s2[m] - tmp_s2[m]
                    sc.Rsv_c_loc[m] = Bc2[m] - out_c2[m] - tmp_c2[m]
                end

                load_series_padded!(As2, Ac2, sc.v_s_loc, sc.v_c_loc, Kloc)
                load_series_padded!(Bs2, Bc2, sc.sv_s_loc, sc.sv_c_loc, Kloc)
                mul_coeffs!(out_s2, out_c2, As2, Ac2, Bs2, Bc2, K2pos)
                for m in 1:Kloc
                    drag_v_s[m] = rT * out_s2[m]
                    drag_v_c[m] = rT * out_c2[m]
                end

                fill!(As2, zero(Ty)); fill!(Ac2, zero(Ty))
                for m in 2:Kloc
                    k = m - 1
                    ω = Ty(k) * σT
                    As2[m] = (-ω) * Ty(sc.v_c_loc[m])
                    Ac2[m] = ( ω) * Ty(sc.v_s_loc[m])
                end
                mul_coeffs!(out_s2, out_c2, Ds2, Dc2, As2, Ac2, K2pos)
                for m in 1:Kloc
                    Ts[m] = out_s2[m]
                    Tc[m] = out_c2[m]
                end

                load_face_coeffs!(sc.z_s_nb, sc.z_c_nb, y, p, FIELD_z_s, FIELD_z_c, i, jm)
                load_face_coeffs!(sc.z_s_loc, sc.z_c_loc, y, p, FIELD_z_s, FIELD_z_c, i, j)
                fill!(As2, zero(Ty)); fill!(Ac2, zero(Ty))
                for m in 1:Kloc
                    As2[m] = (Ty(sc.z_s_loc[m]) - Ty(sc.z_s_nb[m])) * invΔyT
                    Ac2[m] = (Ty(sc.z_c_loc[m]) - Ty(sc.z_c_nb[m])) * invΔyT
                end
                mul_coeffs!(out_s2, out_c2, Ds2, Dc2, As2, Ac2, K2pos)
                for m in 1:Kloc
                    Pm_s[m] = out_s2[m]
                    Pm_c[m] = out_c2[m]
                end

                load_face_coeffs!(sc.v_s_nb, sc.v_c_nb, y, p, FIELD_v_s, FIELD_v_c, ip, j)
                load_face_coeffs!(sc.z_s_nb, sc.z_c_nb, y, p, FIELD_v_s, FIELD_v_c, im, j)
                load_face_coeffs!(sc.z_s_loc, sc.z_c_loc, y, p, FIELD_v_s, FIELD_v_c, i,  jp)
                load_face_coeffs!(sc.u_s_nb, sc.u_c_nb, y, p, FIELD_v_s, FIELD_v_c, i,  jm)

                fill!(As2, zero(Ty)); fill!(Ac2, zero(Ty))
                fill!(Bs2, zero(Ty)); fill!(Bc2, zero(Ty))
                for m in 1:Kloc
                    As2[m] = (Ty(sc.v_s_nb[m]) - Ty(sc.z_s_nb[m])) * inv2ΔxT
                    Ac2[m] = (Ty(sc.v_c_nb[m]) - Ty(sc.z_c_nb[m])) * inv2ΔxT
                    Bs2[m] = (Ty(sc.z_s_loc[m]) - Ty(sc.u_s_nb[m])) * inv2ΔyT
                    Bc2[m] = (Ty(sc.z_c_loc[m]) - Ty(sc.u_c_nb[m])) * inv2ΔyT
                end

                load_series_padded!(tmp_s2, tmp_c2, uface_s, uface_c, Kloc; add_s1 = UT)
                mul_coeffs!(out_s2, out_c2, tmp_s2, tmp_c2, As2, Ac2, K2pos)

                load_series_padded!(tmp_s2, tmp_c2, sc.v_s_loc, sc.v_c_loc, Kloc)
                mul_coeffs!(As2, Ac2, tmp_s2, tmp_c2, Bs2, Bc2, K2pos)

                fill!(Bs2, zero(Ty)); fill!(Bc2, zero(Ty))
                for m in 1:K2loc
                    Bs2[m] = out_s2[m] + As2[m]
                    Bc2[m] = out_c2[m] + Ac2[m]
                end
                mul_coeffs!(out_s2, out_c2, Ds2, Dc2, Bs2, Bc2, K2pos)

                load_series_padded!(Bs2, Bc2, uface_s, uface_c, Kloc; add_s1 = UT)
                mul_coeffs!(tmp_s2, tmp_c2, Ds2, Dc2, Bs2, Bc2, K2pos)

                fill!(Bs2, zero(Ty)); fill!(Bc2, zero(Ty))

                for m in 1:Kloc
                    k = m - 1
                    if k == 0
                        sc.Rv_s_loc[m] = Ty(sc.v_s_loc[m])
                    else
                        sc.Rv_s_loc[m] = Ts[m] + out_s2[m] + drag_v_s[m] + gT * Pm_s[m] + EfT * tmp_s2[m] + Bs2[m]
                    end
                    sc.Rv_c_loc[m] = Tc[m] + out_c2[m] + drag_v_c[m] + gT * Pm_c[m] + EfT * tmp_c2[m] + Bc2[m]
                end

                store_face_coeffs!(F, p, FIELD_v_s,  FIELD_v_c,  i, j, sc.Rv_s_loc,  sc.Rv_c_loc)
                store_face_coeffs!(F, p, FIELD_sV_s, FIELD_sV_c, i, j, sc.Rsv_s_loc, sc.Rsv_c_loc)
            end
        end

    end

    Threads.@threads for chunk in 1:nchunks
        sc = get_scratch!(p, Ty, chunk)
        tid = chunk
        face_lo, face_hi = _chunk_bounds(chunk, nchunks, nface)

        for face in face_lo:face_hi
            i, j = _face_to_ij(face, Nx)
            ip = ip1(i, Nx)
            jp = ip1(j, Ny)

            ζs_face = sc.ζs_face
            ζc_face = sc.ζc_face
            us_face = sc.us_face
            uc_face = sc.uc_face
            vs_face = sc.vs_face
            vc_face = sc.vc_face
            ps_face = sc.ps_face
            pc_face = sc.pc_face
            qxL_s = sc.qxL_s; qxL_c = sc.qxL_c
            qxR_s = sc.qxR_s; qxR_c = sc.qxR_c
            qyB_s = sc.qyB_s; qyB_c = sc.qyB_c
            qyT_s = sc.qyT_s; qyT_c = sc.qyT_c

            @inbounds begin
                load_face_coeffs!(sc.z_s_loc, sc.z_c_loc, y, p, FIELD_z_s, FIELD_z_c, i, j)
                sc.Rz_s_loc[1] = Ty(sc.z_s_loc[1])

                hU_L = Ty(h_on_U_face(p, i, j))
                D0L = HT - hU_L
                zeta_on_U_face!(ζs_face, ζc_face, y, p, i, j, sc)
                load_face_coeffs!(sc.u_s_loc, sc.u_c_loc, y, p, FIELD_u_s, FIELD_u_c, i, j)

                for m in 1:Kloc
                    k = m - 1
                    us_face[m] = (k == 0) ? zero(Ty) : Ty(sc.u_s_loc[m])
                    uc_face[m] = Ty(sc.u_c_loc[m])
                end
                if Kpos >= 1
                    us_face[2] += UT
                end
                mul_coeffs!(ps_face, pc_face, ζs_face, ζc_face, us_face, uc_face, Kpos)
                for m in 1:Kloc
                    qxL_s[m] = D0L * us_face[m] + ps_face[m]
                    qxL_c[m] = D0L * uc_face[m] + pc_face[m]
                end

                hU_R = Ty(h_on_U_face(p, ip, j))
                D0R = HT - hU_R
                zeta_on_U_face!(ζs_face, ζc_face, y, p, ip, j, sc)
                load_face_coeffs!(sc.u_s_loc, sc.u_c_loc, y, p, FIELD_u_s, FIELD_u_c, ip, j)

                for m in 1:Kloc
                    k = m - 1
                    us_face[m] = (k == 0) ? zero(Ty) : Ty(sc.u_s_loc[m])
                    uc_face[m] = Ty(sc.u_c_loc[m])
                end
                if Kpos >= 1
                    us_face[2] += UT
                end
                mul_coeffs!(ps_face, pc_face, ζs_face, ζc_face, us_face, uc_face, Kpos)
                for m in 1:Kloc
                    qxR_s[m] = D0R * us_face[m] + ps_face[m]
                    qxR_c[m] = D0R * uc_face[m] + pc_face[m]
                end

                hV_B = Ty(h_on_V_face(p, i, j))
                D0B = HT - hV_B
                zeta_on_V_face!(ζs_face, ζc_face, y, p, i, j, sc)
                load_face_coeffs!(sc.v_s_loc, sc.v_c_loc, y, p, FIELD_v_s, FIELD_v_c, i, j)

                for m in 1:Kloc
                    k = m - 1
                    vs_face[m] = (k == 0) ? zero(Ty) : Ty(sc.v_s_loc[m])
                    vc_face[m] = Ty(sc.v_c_loc[m])
                end
                mul_coeffs!(ps_face, pc_face, ζs_face, ζc_face, vs_face, vc_face, Kpos)
                for m in 1:Kloc
                    qyB_s[m] = D0B * vs_face[m] + ps_face[m]
                    qyB_c[m] = D0B * vc_face[m] + pc_face[m]
                end

                hV_T = Ty(h_on_V_face(p, i, jp))
                D0T = HT - hV_T
                zeta_on_V_face!(ζs_face, ζc_face, y, p, i, jp, sc)
                load_face_coeffs!(sc.v_s_loc, sc.v_c_loc, y, p, FIELD_v_s, FIELD_v_c, i, jp)

                for m in 1:Kloc
                    k = m - 1
                    vs_face[m] = (k == 0) ? zero(Ty) : Ty(sc.v_s_loc[m])
                    vc_face[m] = Ty(sc.v_c_loc[m])
                end
                mul_coeffs!(ps_face, pc_face, ζs_face, ζc_face, vs_face, vc_face, Kpos)
                for m in 1:Kloc
                    qyT_s[m] = D0T * vs_face[m] + ps_face[m]
                    qyT_c[m] = D0T * vc_face[m] + pc_face[m]
                end

                for m in 1:Kloc
                    k = m - 1
                    ω = Ty(k) * σT

                    div_s = (qxR_s[m] - qxL_s[m]) * invΔxT + (qyT_s[m] - qyB_s[m]) * invΔyT
                    div_c = (qxR_c[m] - qxL_c[m]) * invΔxT + (qyT_c[m] - qyB_c[m]) * invΔyT

                    if k == 0
                        if i == 1 && j == 1
                            sc.Rz_c_loc[m] = Ty(sc.z_c_loc[m])
                        else
                            sc.Rz_c_loc[m] = div_c
                        end
                    else
                        sc.Rz_s_loc[m] = (-ω) * Ty(sc.z_c_loc[m]) + div_s
                        sc.Rz_c_loc[m] = ( ω) * Ty(sc.z_s_loc[m]) + div_c
                    end
                end

                store_face_coeffs!(F, p, FIELD_z_s, FIELD_z_c, i, j, sc.Rz_s_loc, sc.Rz_c_loc)
            end
        end

    end

    return nothing
end

function eval_series_at(u_s, u_c, σ, t; addU::Float64 = 0.0)
    Kloc = length(u_s)
    acc = u_c[1]
    for m in 2:Kloc
        k = m - 1
        us_eff = u_s[m]
        if k == 1
            us_eff += addU
        end
        acc += us_eff * sin(k * σ * t) + u_c[m] * cos(k * σ * t)
    end
    return acc
end

function reconstruct_center_fields(y, p::Params2DHarm, t::Float64)
    Nx, Ny = p.Nx, p.Ny
    σ = p.σ
    Kloc = p.Kloc

    uF = zeros(Float64, Nx, Ny)
    vF = zeros(Float64, Nx, Ny)
    ζC = zeros(Float64, Nx, Ny)

    us = zeros(Float64, Kloc); uc = zeros(Float64, Kloc)
    vs = zeros(Float64, Kloc); vc = zeros(Float64, Kloc)
    zs = zeros(Float64, Kloc); zc = zeros(Float64, Kloc)

    @inbounds for j in 1:Ny, i in 1:Nx
        load_face_coeffs!(us, uc, y, p, FIELD_u_s, FIELD_u_c, i, j)
        load_face_coeffs!(vs, vc, y, p, FIELD_v_s, FIELD_v_c, i, j)
        load_face_coeffs!(zs, zc, y, p, FIELD_z_s, FIELD_z_c, i, j)

        uF[i, j] = eval_series_at(us, uc, σ, t; addU = p.U)
        vF[i, j] = eval_series_at(vs, vc, σ, t; addU = 0.0)
        ζC[i, j] = eval_series_at(zs, zc, σ, t; addU = 0.0)
    end

    uC = similar(ζC)
    vC = similar(ζC)
    @inbounds for j in 1:Ny, i in 1:Nx
        im = (i == 1) ? Nx : i - 1
        jm = (j == 1) ? Ny : j - 1
        uC[i, j] = 0.5 * (uF[i, j] + uF[im, j])
        vC[i, j] = 0.5 * (vF[i, j] + vF[i, jm])
    end

    return uC, vC, ζC
end

function extrema_many(arrs)
    mn = Inf
    mx = -Inf
    for A in arrs
        a, b = extrema(A)
        mn = min(mn, a)
        mx = max(mx, b)
    end
    return mn, mx
end

function eval_u_pert_face_at(y, p::Params2DHarm, i::Int, j::Int, t::Float64)
    Kloc = p.Kloc
    us = Vector{Float64}(undef, Kloc)
    uc = Vector{Float64}(undef, Kloc)
    load_face_coeffs!(us, uc, y, p, FIELD_u_s, FIELD_u_c, i, j)
    return eval_series_at(us, uc, p.σ, t; addU = 0.0)
end

function upert_center_row_xt!(urow, y, p::Params2DHarm, t::Float64, j::Int)
    Nx = p.Nx
    @inbounds for i in 1:Nx
        im = (i == 1) ? Nx : i - 1
        uF_i  = eval_u_pert_face_at(y, p, i,  j, t)
        uF_im = eval_u_pert_face_at(y, p, im, j, t)
        urow[i] = 0.5 * (uF_i + uF_im)
    end
    return nothing
end


function plot_midrow_xt(sol, p::Params2DHarm; nperiods::Int = 30, nt_per_period::Int = 64,
                        savepath::Union{Nothing,String} = "ymid_xt_30periods_harmonic_major.png")

    xC = [(i - 0.5) * p.Δx for i in 1:p.Nx]
    Tper = 2π / p.σ
    ts = range(0.0, nperiods * Tper; length = nperiods * nt_per_period + 1)
    τ = ts ./ Tper

    y_centers = [(j - 0.5) * p.Δy for j in 1:p.Ny]
    j_mid = argmin(abs.(y_centers .- (p.Ny * p.Δy) / 2))
    y_mid = y_centers[j_mid]

    Uxt = zeros(Float64, length(ts), p.Nx)
    Vxt = zeros(Float64, length(ts), p.Nx)
    Zxt = zeros(Float64, length(ts), p.Nx)

    for (it, t) in enumerate(ts)
        uC, vC, ζC = reconstruct_center_fields(sol.u, p, t)
        @views Uxt[it, :] .= uC[:, j_mid] .- p.U * sin(p.σ * t)
        @views Vxt[it, :] .= vC[:, j_mid]
        @views Zxt[it, :] .= ζC[:, j_mid]
    end

    pu = heatmap(
        τ, xC, permutedims(Uxt);
        xlabel = "t / T",
        ylabel = "x [m]",
        title = "u′(x,t) at y ≈ $(round(y_mid, digits=2)) m",
        colorbar_title = "u′ [m/s]",
        aspect_ratio = :auto,
        margin = 5mm,
    )

    pv = heatmap(
        τ, xC, permutedims(Vxt);
        xlabel = "t / T",
        ylabel = "x [m]",
        title = "v(x,t) at y ≈ $(round(y_mid, digits=2)) m",
        colorbar_title = "v [m/s]",
        aspect_ratio = :auto,
        margin = 5mm,
    )

    pz = heatmap(
        τ, xC, permutedims(Zxt);
        xlabel = "t / T",
        ylabel = "x [m]",
        title = "ζ(x,t) at y ≈ $(round(y_mid, digits=2)) m",
        colorbar_title = "ζ [m]",
        aspect_ratio = :auto,
        margin = 5mm,
    )

    fig = plot(pu, pv, pz; layout = (3, 1), size = (1000, 1100))

    if savepath !== nothing
        savefig(fig, savepath)
        println("Saved mid-row x–t plot to $savepath")
    end

    display(fig)
    return fig, pu
end

function initialize_aux_speed!(y0, p::Params2DHarm)
    _, S0, C0, _ = make_time_basis(p.Kpos, p.σ; Nt = 32)
    U0_t = p.U .* vec(S0[2, :])
    s0_t = sqrt.(U0_t .^ 2 .+ p.drag_eps)

    s0_s, s0_c = project_signal_to_basis(s0_t, S0, C0, p.Kpos)

    @inbounds for j in 1:p.Ny, i in 1:p.Nx
        for m in 1:p.Kloc
            set_coeff!(y0, p, m, FIELD_sU_s, i, j, s0_s[m])
            set_coeff!(y0, p, m, FIELD_sU_c, i, j, s0_c[m])
            set_coeff!(y0, p, m, FIELD_sV_s, i, j, s0_s[m])
            set_coeff!(y0, p, m, FIELD_sV_c, i, j, s0_c[m])
        end
    end
    return y0
end




function Base.copy(p::Params2DHarm)
    Params2DHarm(
        p.Nx,
        p.Ny,
        p.Δx,
        p.Δy,
        p.H,
        p.g,
        p.r,
        p.U,
        p.σ,
        p.Ef,
        p.drag_eps,
        p.hC,
        p.Kpos,
        p.Kloc,
        p.K2pos,
        p.K2loc,
        p.nface,
        p.nfields,
        p.harm_block,
        p.bg_s,
        p.bg_c,
        IdDict{DataType, Any}(),
        ReentrantLock(),
    )
end
