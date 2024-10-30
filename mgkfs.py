import pickle

import numpy as np
import taichi as ti
from matplotlib import cm

ti.init(arch=ti.cuda, debug=True)

# -------------------------------------------------------------------------------------------------
# The implementation of M-GKFS (Maxwellian Gas Kinetic Flux Solver)
# The main reference is:
# the book: [Lattice Boltzmann and Gas Kinetic Flux Solvers: Theory and Applications]
#   and its original paper
# the paper: [Explicit formulations of gas-kinetic flux solver for simulation of
#   incompressible and compressible viscous flows]
# Without further annotation, [Yang et al. (2020)] is referring to the above
# book, [Sun et al. (2015)] is referring to the original paper.
# -------------------------------------------------------------------------------------------------

# Variables on-cell
Nx = 1 + 2
Ny = 1 + 2
D = 2
K = 0
b = K + D
dtype = ti.f32

PI = 3.1415926535897
EPS = 1e-5
c_s = 1 / ti.sqrt(3)  # Sound of speed, at sqrt(gamma*R*T)
u_ref = 0.001  # Reference velocity
T_ref = 1.0  # Reference temperature
gamma = (b + 2) / b  # Heat ratio
R = c_s**2 / (gamma * T_ref)  # Gas constant
Ma = u_ref / c_s  # Mach number

Re = 10000
tau = 3 * u_ref * (Nx - 2) / Re
stride = 1
K = 1
print("=== M-GKFS Parameters ===")
print(f"= R:   {R:.5f}")
print(f"= mfp: {1.0 / (2 * R * T_ref):.5f}")
print("=")
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
# W = [rho, rho u, rho E]
# F = [rho u, rho uu + pI - tau, (energy term)]
# Actually, we don't need the concrete form of F.
#
# Some conventions to define:
# - <xi f>: integrate (xi*f) on all velocities, i.e., R^d
# - phi = [1, xi_1, xi_2, 0.5(xi_1^2 + xi_2^2)]: a moment vector, four elements for M-GKFS.
# -------------------------------------------------------------------------------------------------
flag = ti.field(dtype=ti.i8, shape=(Nx, Ny))
rho = ti.field(dtype=dtype, shape=(Nx, Ny))
u = ti.Vector.field(2, dtype=dtype, shape=(Nx, Ny))
T = ti.field(dtype=dtype, shape=(Nx, Ny))
rho_new = ti.field(dtype=dtype, shape=(Nx, Ny))
u_new = ti.Vector.field(2, dtype=dtype, shape=(Nx, Ny))
T_new = ti.field(dtype=dtype, shape=(Nx, Ny))

# Face normal
N = ti.Vector.field(2, dtype=ti.i32, shape=4)
N.from_numpy(np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]))

V = ti.Vector.field(2, dtype=ti.i32, shape=8)
V.from_numpy(np.array([[1, 1], [1, 0], [0, 1], [1, 1], [0, 0], [0, 1], [1, 0], [0, 0]]))


# A approximation function from https://stackoverflow.com/questions/457408/is-there-an-easily-available-implementation-of-erf-for-python
@ti.func
def erf(x):
    # save the sign of x
    sign = 1 if x >= 0 else -1
    x = abs(x)

    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * ti.exp(-x * x)
    return sign * y


@ti.func
def erfc(x):
    return 1 - erf(x)


@ti.func
def get_mfp(i: int, j: int):
    return 1.0 / (2 * R * T[i, j])


@ti.func
def is_inside(i: int, j: int):
    return i >= 0 and i < Nx and j >= 0 and j < Ny


@ti.func
def get_rho_E_at(i: int, j: int):
    return rho[i, j] * ti.math.dot(u[i, j], u[i, j]) / 2.0 + rho[i, j] * R * T[i, j] / (
        gamma - 1
    )


@ti.func
def get_W_at(i: int, j: int):
    return ti.Vector([
        rho[i, j],
        rho[i, j] * u[i, j][0],
        rho[i, j] * u[i, j][1],
        get_rho_E_at(i, j),
    ])


@ti.func
def get_grad_W(i: int, j: int, R):
    grad_rho = ti.Vector([0.0, 0.0])
    grad_rhoU1 = ti.Vector([0.0, 0.0])
    grad_rhoU2 = ti.Vector([0.0, 0.0])
    grad_rhoE = ti.Vector([0.0, 0.0])
    W_c = get_W_at(i, j)

    for k in ti.static(range(4)):
        i_n, j_n = i + N[k][0], j + N[k][1]
        n = R @ N[k]
        if is_inside(i_n, j_n):
            W_n = get_W_at(i_n, j_n)
            grad_rho += (W_n[0] + W_c[0]) * n / 2.0
            grad_rhoU1 += (W_n[1] + W_c[1]) * n / 2.0
            grad_rhoU2 += (W_n[2] + W_c[2]) * n / 2.0
            grad_rhoE += (W_n[3] + W_c[3]) * n / 2.0
        else:
            grad_rho += W_c[0] * n
            grad_rhoU1 += W_c[1] * n
            grad_rhoU2 += W_c[2] * n
            grad_rhoE += W_c[3] * n

    return grad_rho, grad_rhoU1, grad_rhoU2, grad_rhoE


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
def mgkfs_recursive_moments(T0, T1, u, mfp):
    """
    Compute the moments of the distribution function recursively.
    """
    m = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    m[0] = T0
    m[1] = T1
    for k in ti.static(6 - 2):
        m[k + 2] = m[k + 1] * u + m[k] * (k + 1) / (2 * mfp)
    return m


@ti.func
def mgkfs_solve_for_coeff(
    h0: dtype, h1: dtype, h2: dtype, h3: dtype, u1: dtype, u2: dtype, mfp: dtype
):
    """
    This function solves for equation like:
        <a_0 + a_1 xi_1 + a_2 xi_2 + a_3 (...) phi_a g> = h*rho,
    which can be written as
        M @ [a_0, a_1, a_2, a_3]^T = [h0, h1, h2, h3]^T,
    where M is a 4x4 matrix defined in the book. One's role is to calculate h0, h1, h2, h3 accordingly.
    As for the details of the coefficients, see [Appendix B, Yang et al. 2020].
    """
    r0 = u1**2 + u2**2 + (K + 2) / (2 * mfp)
    r1 = h1 - u1 * h0
    r2 = h2 - u2 * h0
    r3 = 2 * h3 - r0 * h0
    a3 = (4 * mfp**2) / (K + 2) * (r3 - 2 * u1 * r1 - 2 * u2 * r2)
    a1 = 2 * mfp * r2 - u2 * a3
    a2 = 2 * mfp * r1 - u1 * a3
    a0 = h0 - u1 * a1 - u2 * a2 - a3 * r0 / 2
    return a0, a1, a2, a3


@ti.func
def mgkfs_compute_flux(i: int, j: int, face_id: int):
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

    i_L, j_L = i, j
    i_R, j_R = i + N[face_id][0], j + N[face_id][1]

    rho_L, rho_R = rho[i_L, j_L], rho[i_R, j_R]
    u_L, u_R = R @ u[i_L, j_L], R @ u[i_R, j_R]
    u1_L, u1_R = u_L[0], u_R[0]
    u2_L, u2_R = u_L[1], u_R[1]
    T_L, T_R = T[i_L, j_L], T[i_R, j_R]
    mfp_L, mfp_R = get_mfp(i_L, j_L), get_mfp(i_R, j_R)
    smfp_L, smfp_R = ti.sqrt(mfp_L), ti.sqrt(mfp_R)

    T0 = erfc(-smfp_L * u1_L) / 2.0
    T1 = u1_L * T0 + ti.exp(-mfp_L * u1_L**2) / (2 * ti.sqrt(PI * mfp_L))
    M_L = mgkfs_recursive_moments(T0, T1, u1_L, mfp_L)

    T0 = erfc(smfp_R * u1_R) / 2.0
    T1 = u1_R * T0 - ti.exp(-mfp_R * u1_R**2) / (2 * ti.sqrt(PI * mfp_R))
    M_R = mgkfs_recursive_moments(T0, T1, u1_R, mfp_R)

    M_CL = mgkfs_recursive_moments(1, u2_L, u2_L, mfp_L)
    M_CR = mgkfs_recursive_moments(1, u2_R, u2_R, mfp_R)

    rho_i = M_L[0] * rho_L + M_R[0] * rho_R
    rho_i = ti.max(rho_i, EPS)
    u1 = (M_L[1] * rho_L + M_R[1] * rho_R) / rho_i
    u2 = (M_L[0] * rho_L * u2_L + M_R[0] * rho_R * u2_R) / rho_i

    T0 = (u2_L**2 + (b - 1) * R * T_L) * M_L[0]
    T1 = (u2_R**2 + (b - 1) * R * T_R) * M_R[0]
    E_i = ((M_L[2] + T0) * rho_L + (M_R[2] + T1) * rho_R) / (2 * rho_i)

    # -------------------------------------------------------------------------------------------------
    Mx_P = M_L[1]
    My_P = M_L[1]
    Mxx_P = M_L[2]
    Mxy_P = Mx_P * My_P
    Myy_P = M_CL[2]
    Mxxx_P = M_L[3]
    Mxxy_P = 0
    Mxyy_P = 0
    Myyy_P = M_CL[3]

    # -------------------------------------------------------------------------------------------------
    grad_rho_L, grad_rhoU1_L, grad_rhoU2_L, grad_rhoE_L = get_grad_W(i_L, j_L, R_inv)
    pUpX_L = R @ ti.Vector([grad_rhoU1_L[0], grad_rhoU2_L[0]])
    pUpY_L = R @ ti.Vector([grad_rhoU1_L[1], grad_rhoU2_L[1]])
    a0_L, a1_L, a2_L, a3_L = mgkfs_solve_for_coeff(
        grad_rho_L[0] / rho_L,
        pUpX_L[0] / rho_L,
        pUpX_L[1] / rho_L,
        grad_rhoE_L[0] / rho_L,
        u1_L,
        u2_L,
        mfp_L,
    )
    b0_L, b1_L, b2_L, b3_L = mgkfs_solve_for_coeff(
        grad_rho_L[1] / rho_L,
        pUpY_L[0] / rho_L,
        pUpY_L[1] / rho_L,
        grad_rhoE_L[1] / rho_L,
        u1_L,
        u2_L,
        mfp_L,
    )

    grad_rho_R, grad_rhoU1_R, grad_rhoU2_R, grad_rhoE_R = get_grad_W(i_R, j_R, R_inv)
    pUpX_R = R @ ti.Vector([grad_rhoU1_R[0], grad_rhoU2_R[0]])
    pUpY_R = R @ ti.Vector([grad_rhoU1_R[1], grad_rhoU2_R[1]])
    a0_R, a1_R, a2_R, a3_R = mgkfs_solve_for_coeff(
        grad_rho_R[0] / rho_R,
        pUpX_R[0] / rho_R,
        pUpX_R[1] / rho_R,
        grad_rhoE_R[0] / rho_R,
        u1_R,
        u2_R,
        mfp_R,
    )
    b0_R, b1_R, b2_R, b3_R = mgkfs_solve_for_coeff(
        grad_rho_R[1] / rho_R,
        pUpY_R[0] / rho_R,
        pUpY_R[1] / rho_R,
        grad_rhoE_R[1] / rho_R,
        u1_R,
        u2_R,
        mfp_R,
    )

    # -------------------------------------------------------------------------------------------------
    return 0, 0, 0


def mgkfs_init_flag():
    flag_np = flag.to_numpy()
    flag_np[:, Ny - 1] = NO_SLIP
    flag_np[0, :] = NO_SLIP
    flag_np[Nx - 1, :] = NO_SLIP
    flag_np[:, 0] = NO_SLIP
    flag.from_numpy(flag_np)


def mgkfs_init_u():
    u_np = u.to_numpy()
    u_np[2, 1] = np.array([2.0, 1.0])
    u.from_numpy(u_np)


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
def migks_bc_stable():
    for i, j in ti.ndrange(Nx, Ny):
        if flag[i, j] == NO_SLIP or flag[i, j] == SLIP:
            u[i, j] = ti.Vector([0.0, 0.0])
            rho[i, j] = 1.0
        elif flag[i, j] == VELOCITY:
            u[i, j] = u_bc
            rho[i, j] = 1.0


@ti.kernel
def migks_step():
    """
    Update the variables on-cell through FVM.
    """
    for i, j in ti.ndrange(Nx, Ny):
        if flag[i, j] != GAS:
            rho_new[i, j] = rho[i, j]
            u_new[i, j] = u[i, j]
            T_new[i, j] = T[i, j]
            continue

        d_rho = 0.0
        d_m = ti.Vector([0.0, 0.0])

        for face_id in ti.static(range(4)):
            F0, F1, F2 = mgkfs_compute_flux(i, j, face_id)
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
    T.fill(T_ref)
    mgkfs_init_flag()
    mgkfs_init_u()
    gui = ti.GUI("M-IGKS", (Nx, Ny), background_color=0x212121)

    frame_id = 0
    while gui.running:
        if gui.get_event(ti.GUI.ESCAPE):
            gui.running = False
            save()
            break
        # migks_bc_stable()
        migks_step()
        break
        rho.copy_from(rho_new)
        u.copy_from(u_new)
        T.copy_from(T_new)
        if frame_id % stride == 0:
            col = cm.coolwarm(np.linalg.norm(u.to_numpy(), axis=-1) / (u_ref * 1.2))
            gui.set_image(col)
            gui.show()
        frame_id += 1


if __name__ == "__main__":
    main()
