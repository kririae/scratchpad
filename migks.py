import pickle

import numpy as np
import taichi as ti
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
u_ref = 0.1
Re = 1000
tau = 3 * u_ref * (Nx - 2) / Re
stride = 360
print("=== M-IGKS Parameters ===")
print(f"= Re:  {Re:.5f}")
print(f"= tau: {tau:.5f}")
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
    a1 = 3 * r1
    a2 = 3 * r2
    a0 = h0 - u1 * a1 - u2 * a2
    return a0, a1, a2


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
    # integrated by Mathematica
    # [eq 8.45, Yang et al. 2020]
    # -------------------------------------------------------------------------------------------------
    Mx = rho_i * u1
    My = rho_i * u2
    Mxx = rho_i * u1 * u1 + rho_i / 3.0
    Myy = rho_i * u2 * u2 + rho_i / 3.0
    Mxy = rho_i * u1 * u2
    Mxxx = rho_i * u1 + rho_i * u1 * u1 * u1
    Myyy = rho_i * u2 + rho_i * u2 * u2 * u2
    Mxxy = rho_i * (1 + 3 * u1 * u1) * u2 / 3.0
    Mxyy = rho_i * (1 + 3 * u2 * u2) * u1 / 3.0

    h0 = -(a0 * Mx + a1 * Mxx + a2 * Mxy + b0 * My + b1 * Mxy + b2 * Myy)
    h1 = -(a0 * Mxx + a1 * Mxxx + a2 * Mxxy + b0 * Mxy + b1 * Mxxy + b2 * Mxyy)
    h2 = -(a0 * Mxy + a1 * Mxxy + a2 * Mxyy + b0 * Myy + b1 * Mxyy + b2 * Myyy)
    A0, A1, A2 = migks_solve_for_coeff(h0, h1, h2, u1, u2)

    # -------------------------------------------------------------------------------------------------
    # (4) Compute the flux correspondingly
    # [eq 8.46, Yang et al. 2020]
    # -------------------------------------------------------------------------------------------------
    Mxxxx = rho_i * (1 + 6 * u1 * u1 + 3 * u1 * u1 * u1 * u1) / 3.0
    Mxxxy = rho_i * (1 + u1 * u1) * u1 * u2
    Mxyyy = rho_i * (1 + u2 * u2) * u1 * u2
    Mxxyy = (1 + 3 * u1 * u1) * (1 + 3 * u2 * u2) * rho_i / 9.0

    T0 = 1 + A0 / 2 - tau * A0
    T1 = -tau * a0 + A1 / 2 - tau * A1
    T2 = -tau * a1
    T3 = A2 / 2 - tau * A2 - tau * b0
    T4 = -tau * a2 - tau * b1
    T5 = -tau * b2

    F0 = Mx * T0 + Mxx * T1 + Mxxx * T2 + Mxy * T3 + Mxxy * T4 + Mxyy * T5
    F1 = Mxx * T0 + Mxxx * T1 + Mxxxx * T2 + Mxxy * T3 + Mxxxy * T4 + Mxxyy * T5
    F2 = Mxy * T0 + Mxxy * T1 + Mxxxy * T2 + Mxyy * T3 + Mxxyy * T4 + Mxyyy * T5
    Fl = R_inv @ ti.Vector([F1, F2])
    return F0, Fl[0], Fl[1]


def migks_init_flag():
    flag_np = flag.to_numpy()
    flag_np[:, Ny - 1] = VELOCITY
    flag_np[0, :] = NO_SLIP
    flag_np[Nx - 1, :] = NO_SLIP
    flag_np[:, 0] = NO_SLIP
    flag.from_numpy(flag_np)


@ti.kernel
def migks_bc():
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

        d_rho = 0.0
        d_m = ti.Vector([0.0, 0.0])

        for face_id in ti.static(range(4)):
            F0, F1, F2 = migks_compute_flux(i, j, face_id)
            d_rho += F0
            d_m += ti.Vector([F1, F2])

        # Perform the FVM update
        rho_prev = rho[i, j]
        u_prev = u[i, j]
        rho_new[i, j] = rho_prev - d_rho
        u_new[i, j] = (rho_prev * u_prev - d_m) / (rho_prev - d_rho)


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
        migks_bc()
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
