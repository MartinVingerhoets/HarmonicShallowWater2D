const FIELD_u_s  = 1
const FIELD_u_c  = 2
const FIELD_v_s  = 3
const FIELD_v_c  = 4
const FIELD_z_s  = 5
const FIELD_z_c  = 6
const FIELD_sU_s = 7
const FIELD_sU_c = 8
const FIELD_sV_s = 9
const FIELD_sV_c = 10

const JFIELD_GROUP_LABELS = ["u", "v", "ζ", "sU", "sV"]
const JRESID_GROUP_LABELS = ["R_u", "R_v", "R_ζ", "R_sU", "R_sV"]

ip1(i, N) = (i == N) ? 1 : (i + 1)
im1(i, N) = (i == 1) ? N : (i - 1)
face_idx(i::Int, j::Int, Nx::Int) = (j - 1) * Nx + i

state_length(p) = p.Kloc * p.harm_block

sinusoidal_topography(Lx::Real; amplitude::Real = 1.0, phase::Real = 0.0) =
    (x, y) -> amplitude * sin(2π * x / Lx + phase)
