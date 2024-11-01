import pickle

import numpy as np
import taichi as ti
import torch
from matplotlib import cm

ti.init(arch=ti.cuda)

# -------------------------------------------------------------------------------------------------
# The implementation of M-IGKS
# The main reference is:
# [Yang, L. & Wang, Yan & Chen, Zhen & Shu, Chang. (2020). Lattice Boltzmann and
# Gas Kinetic Flux Solvers: Theory and Applications. 10.1142/11949].
# without further annotation, [Yang et al. (2020)] is referring to the above book.
# -------------------------------------------------------------------------------------------------

# Variables on-cell
Nx = 500 + 2
Ny = 500 + 2
dtype = ti.f32

CFL = 0.8
c_s = 1
gamma = 2
mfp = gamma / (2 * c_s**2)
mfp = 1.5
u_ref = 0.18
dt = ti.field(dtype=dtype, shape=())
dt[None] = CFL / (u_ref + c_s)
Re = 1000
tau = 3 * u_ref * (Nx - 2) / Re
stride = 500

# Extra configurations
dynamic_dt = False

print("=== M-IGKS Parameters ===")
print(f"= mfp: {mfp:.5f}")
print(f"= Re:  {Re:.5f}")
print(f"= tau: {tau:.5f}")
print(f"= dt:  {dt[None]:.5f}")
print("=========================")

# Boundary conditions
GAS = 0
NO_SLIP = 1
SLIP = 2
VELOCITY = 3
u_bc = ti.Vector([u_ref, 0.0])

# -------------------------------------------------------------------------------------------------
# W = [rho, rho u, ...]
# F = [rho u, rho uu + pI - tau, ...]
#
# Some conventions to define:
# - <xi f>: integrate (xi*f) on all velocities, i.e., R^d
# - phi = [1, xi, ...]: a moment vector, two elements for IGKS
# -------------------------------------------------------------------------------------------------
flag = ti.field(dtype=ti.i8, shape=(Nx, Ny))
rho = ti.field(dtype=dtype, shape=(Nx, Ny))
u = ti.Vector.field(2, dtype=dtype, shape=(Nx, Ny))
rho_new = ti.field(dtype=dtype, shape=(Nx, Ny))
u_new = ti.Vector.field(2, dtype=dtype, shape=(Nx, Ny))

# Face normal
N = ti.Vector.field(2, dtype=ti.i32, shape=4)
N.from_numpy(np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]))

V = ti.Vector.field(2, dtype=ti.i32, shape=8)
V.from_numpy(np.array([[1, 1], [1, 0], [0, 1], [1, 1], [0, 0], [0, 1], [1, 0], [0, 0]]))


@ti.func
def is_inside(i: int, j: int):
    return i >= 0 and i < Nx and j >= 0 and j < Ny


@ti.func
def get_rho_at_face(i: int, j: int, face_id: int):
    rho_c = rho[i, j]
    i += N[face_id][0]
    j += N[face_id][1]
    if is_inside(i, j):
        rho_c = 0.5 * (rho_c + rho[i, j])
    return rho_c


@ti.func
def get_u_at_face(i: int, j: int, face_id: int):
    u_c = u[i, j]
    i += N[face_id][0]
    j += N[face_id][1]
    if is_inside(i, j):
        u_c = 0.5 * (u_c + u[i, j])
    return u_c


@ti.func
def get_rho_at_vertex(i: int, j: int):
    rho_s = 0.0
    cnt = 0
    for dx in ti.static([-1, 0]):
        for dy in ti.static([-1, 0]):
            i_n = i + dx
            j_n = j + dy
            if is_inside(i_n, j_n):
                rho_s += rho[i_n, j_n]
                cnt += 1
    return rho_s / cnt


@ti.func
def get_u_at_vertex(i: int, j: int):
    u_s = ti.Vector([0.0, 0.0])
    cnt = 0
    for dx in ti.static([-1, 0]):
        for dy in ti.static([-1, 0]):
            i_n = i + dx
            j_n = j + dy
            if is_inside(i_n, j_n):
                u_s += u[i_n, j_n]
                cnt += 1
    return u_s / cnt


@ti.func
def get_d_rho_towards(i: int, j: int, face_id: int):
    drho = 0.0
    i_n = i + N[face_id][0]
    j_n = j + N[face_id][1]
    if is_inside(i_n, j_n):
        drho = rho[i_n, j_n] - rho[i, j]
    return drho


@ti.func
def get_dm_towards(i: int, j: int, face_id: int):
    dm = ti.Vector([0.0, 0.0])
    i_n = i + N[face_id][0]
    j_n = j + N[face_id][1]
    if is_inside(i_n, j_n):
        dm = rho[i_n, j_n] * u[i_n, j_n] - rho[i, j] * u[i, j]
    return dm


@ti.func
def get_W_at(i: int, j: int):
    return ti.Vector([
        rho[i, j],
        rho[i, j] * u[i, j][0],
        rho[i, j] * u[i, j][1],
    ])

@ti.func
def migks_recursive_moments(T0, T1, u, mfp):
    """
    Compute the moments of the distribution function recursively.
    """
    m = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    m[0] = T0
    m[1] = T1
    for k in ti.static(range(4)):
        m[k + 2] = m[k + 1] * u + m[k] * (k + 1) / (2 * mfp)
    return m


@ti.func
def migks_combined_moments(i: ti.template(), j: ti.template(), Mx, My):
    return Mx[i] * My[j]


@ti.func
def migks_h_base(a, b, Mx, My, oi: ti.template(), oj: ti.template()):
    return -(
        a[0] * migks_combined_moments(1 + oi, 0 + oj, Mx, My)
        + a[1] * migks_combined_moments(2 + oi, 0 + oj, Mx, My)
        + a[2] * migks_combined_moments(1 + oi, 1 + oj, Mx, My)
        + b[0] * migks_combined_moments(0 + oi, 1 + oj, Mx, My)
        + b[1] * migks_combined_moments(1 + oi, 1 + oj, Mx, My)
        + b[2] * migks_combined_moments(0 + oi, 2 + oj, Mx, My)
    )


@ti.func
def migks_F_base(T, Mx, My, oi: ti.template(), oj: ti.template()):
    return (
        T[0] * migks_combined_moments(1 + oi, 0 + oj, Mx, My)
        + T[1] * migks_combined_moments(2 + oi, 0 + oj, Mx, My)
        + T[2] * migks_combined_moments(3 + oi, 0 + oj, Mx, My)
        + T[3] * migks_combined_moments(1 + oi, 1 + oj, Mx, My)
        + T[4] * migks_combined_moments(2 + oi, 1 + oj, Mx, My)
        + T[5] * migks_combined_moments(1 + oi, 2 + oj, Mx, My)
    )


@ti.func
def migks_solve_for_coeff(h0: dtype, h1: dtype, h2: dtype, u1: dtype, u2: dtype):
    """
    This function solves for equation like:
        <a_0 + a_1 xi_1 + a_2 xi_2 phi_a g> = h*rho,
    which can be written as
        M @ [a_0, a_1, a_2]^T = [h0, h1, h2]^T,
    where M is a 3x3 matrix defined in the book. One's role is to calculate h0, h1, h2 accordingly.
    As for the details of the coefficients, see [Appendix C, Yang et al. 2020].
    """
    r1 = h1 - u1 * h0
    r2 = h2 - u2 * h0
    a1 = 2 * mfp * r1
    a2 = 2 * mfp * r2
    a0 = h0 - u1 * a1 - u2 * a2
    return a0, a1, a2


@ti.func
def migks_compute_flux_bc(i: int, j: int, face_id: int):
    """
    Compute the flux for boundary condition on the face_id-th face
    """
    pass


@ti.func
def migks_compute_flux(i: int, j: int, face_id: int):
    """
    Compute the flux on the face_id-th face
    """

    # -------------------------------------------------------------------------------------------------
    # Rotate u_i to local(face) coordinate
    # [eq 5.24; eq 5.25; eq 5.26, Yang et al. 2020]
    # -------------------------------------------------------------------------------------------------
    n_i = N[face_id]
    R = ti.Matrix([[n_i[0], n_i[1]], [-n_i[1], n_i[0]]], dt=ti.i32)
    R_inv = R.transpose()

    rho_i = get_rho_at_face(i, j, face_id)
    u_i = R @ get_u_at_face(i, j, face_id)
    u1 = u_i[0]
    u2 = u_i[1]

    # -------------------------------------------------------------------------------------------------
    # (1) Compute pdv(rho, x) and pdv(rho u, x)
    # [eq 8.42, Yang et al. 2020]
    # -------------------------------------------------------------------------------------------------
    h0 = get_d_rho_towards(i, j, face_id) / rho_i
    dmx = R @ get_dm_towards(i, j, face_id) / rho_i

    # (1.5) Compute a1, a2, a0 correspondingly
    a0, a1, a2 = migks_solve_for_coeff(h0, dmx[0], dmx[1], u1, u2)

    # -------------------------------------------------------------------------------------------------
    # (2) Compute pdv(rho, y) and pdv(rho u, y)
    # Performed with upper vertex and lower vertex
    # [eq 8.43, Yang et al. 2020]
    # -------------------------------------------------------------------------------------------------
    uij = ti.Vector([i, j]) + V[2 * face_id]
    dij = ti.Vector([i, j]) + V[2 * face_id + 1]
    rho_uij = get_rho_at_vertex(uij[0], uij[1])
    rho_dij = get_rho_at_vertex(dij[0], dij[1])
    u_uij = get_u_at_vertex(uij[0], uij[1])
    u_dij = get_u_at_vertex(dij[0], dij[1])

    h0 = (rho_uij - rho_dij) / rho_i
    dmy = R @ (rho_uij * u_uij - rho_dij * u_dij) / rho_i

    # (2) Compute b1, b2, b0 correspondingly
    b0, b1, b2 = migks_solve_for_coeff(h0, dmy[0], dmy[1], u1, u2)

    # -------------------------------------------------------------------------------------------------
    # (3) Compute A0, A1, A2 with a0, a1, a2, b0, b1, b2
    # [eq 8.45, Yang et al. 2020]
    # -------------------------------------------------------------------------------------------------
    MX = migks_recursive_moments(1, u1, u1, mfp)
    MY = migks_recursive_moments(1, u2, u2, mfp)

    a = ti.Vector([a0, a1, a2])
    b = ti.Vector([b0, b1, b2])
    h0 = migks_h_base(a, b, MX, MY, 0, 0)
    h1 = migks_h_base(a, b, MX, MY, 1, 0)
    h2 = migks_h_base(a, b, MX, MY, 0, 1)
    A0, A1, A2 = migks_solve_for_coeff(h0, h1, h2, u1, u2)

    # -------------------------------------------------------------------------------------------------
    # (4) Compute the flux correspondingly
    # [eq 8.46, Yang et al. 2020]
    # -------------------------------------------------------------------------------------------------

    # A factor to determine the method
    # Set alpha=0 to fallback to M-GKFS variant of incompressible solver
    alpha = dt[None]

    T0 = 1 + alpha * A0 / 2 - tau * A0
    T1 = -tau * a0 + alpha * A1 / 2 - tau * A1
    T2 = -tau * a1
    T3 = alpha * A2 / 2 - tau * A2 - tau * b0
    T4 = -tau * a2 - tau * b1
    T5 = -tau * b2
    T = ti.Vector([T0, T1, T2, T3, T4, T5])

    F0 = rho_i * migks_F_base(T, MX, MY, 0, 0)
    F1 = rho_i * migks_F_base(T, MX, MY, 1, 0)
    F2 = rho_i * migks_F_base(T, MX, MY, 0, 1)
    Fl = R_inv @ ti.Vector([F1, F2])

    return ti.Vector([F0, Fl[0], Fl[1]])


@ti.kernel
def migks_init_sphere(radius: int, cx: int, cy: int):
    for i, j in ti.ndrange(Nx, Ny):
        if (i - cx) ** 2 + (j - cy) ** 2 < radius**2:
            flag[i, j] = NO_SLIP


def migks_init_flag():
    flag_np = flag.to_numpy()
    flag_np[:, Ny - 1] = VELOCITY
    flag_np[0, :] = NO_SLIP
    flag_np[:, 0] = NO_SLIP
    flag_np[Nx - 1, :] = NO_SLIP
    flag.from_numpy(flag_np)


def migks_update_dt():
    """
    Update the time step size with CFL condition.
    """

    @torch.compile
    def torch_op(u_torch):
        return torch.max(torch.norm(u_torch, dim=-1))

    u_torch = u.to_torch(device="cuda:0")
    u_max = torch_op(u_torch)
    dt[None] = CFL / (max(u_max, u_ref) + c_s)


@ti.kernel
def migks_boundary_condition():
    for i, j in ti.ndrange(Nx, Ny):
        if flag[i, j] == GAS:
            continue

        rho_ = 0.0
        u_ = ti.Vector([0.0, 0.0])
        cnt_ = 0

        for face_id in ti.static(range(4)):
            i_n = i + N[face_id][0]
            j_n = j + N[face_id][1]

            if is_inside(i_n, j_n) and flag[i_n, j_n] == GAS:
                rho_ += rho[i_n, j_n]
                cnt_ += 1
                if flag[i, j] == NO_SLIP:
                    u_ += -u[i_n, j_n]
                elif flag[i, j] == SLIP:
                    u_ += u[i_n, j_n] - 2 * (u[i_n, j_n] @ N[face_id]) * N[face_id]
                elif flag[i, j] == VELOCITY:
                    u_ += u_bc

        if cnt_ > 0:
            rho[i, j] = rho_ / cnt_
            u[i, j] = u_ / cnt_


@ti.kernel
def migks_step():
    """
    Update the variables on-cell through FVM.
    """
    for i, j in ti.ndrange(Nx, Ny):
        if flag[i, j] != GAS:
            rho_new[i, j] = rho[i, j]
            u_new[i, j] = u[i, j]
            continue

        dW = ti.Vector([0.0, 0.0, 0.0])
        for face_id in range(4):
            dW = dW + dt[None] * migks_compute_flux(i, j, face_id)

        rho_new[i, j] = rho[i, j] - dW[0]
        u_new[i, j] = (rho[i, j] * u[i, j] - ti.Vector([dW[1], dW[2]])) / rho_new[i, j]


def save():
    with open("u_MIGKS_Re1000.pickle", "wb") as f:
        pickle.dump(u.to_numpy(), f, pickle.HIGHEST_PROTOCOL)
        col = cm.coolwarm(np.linalg.norm(u.to_numpy(), axis=-1) / (u_ref * 1.2))
        ti.tools.imwrite(col, "u_MIGKS_Re1000.png")

        print("Saved 'u_MIGKS_Re1000.pickle' and 'u_MIGKS_Re1000.png'")


def main():
    flag.fill(GAS)
    rho.fill(1.0)
    migks_init_flag()
    gui = ti.GUI("M-IGKS", (Nx, Ny), background_color=0x212121)

    frame_id = 0
    while gui.running:
        if gui.get_event(ti.GUI.ESCAPE):
            gui.running = False
            save()
            break

        migks_boundary_condition()
        if dynamic_dt:
            migks_update_dt()
        migks_step()

        rho.copy_from(rho_new)
        u.copy_from(u_new)

        if frame_id % stride == 0:
            col = cm.coolwarm(np.linalg.norm(u.to_numpy(), axis=-1) / (u_ref * 1.2))
            gui.set_image(col)
            gui.show()
        frame_id += 1


if __name__ == "__main__":
    main()
