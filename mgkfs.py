import pickle

import numpy as np
import taichi as ti
from matplotlib import cm

ti.init(arch=ti.cpu, debug=True)

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
Nx = 500 + 2
Ny = 500 + 2
D = 2
K = 0
b = K + D
dtype = ti.f32

PI = 3.1415926535897
EPS = 1e-5
CFL = 0.01
c_s = 1  # Sound of speed, at sqrt(gamma*R*T)
u_ref = 0.1  # Reference velocity
T_ref = 1.0  # Reference temperature
dt = CFL * 1.0 / (u_ref + c_s)  # Time step
gamma = (b + 2) / b  # Heat ratio
Rg = c_s**2 / (gamma * T_ref)  # Gas constant
Ma = u_ref / c_s  # Mach number

Re = 1000
tau = 3 * u_ref * (Nx - 2) / Re
stride = 1

print("=== M-GKFS Parameters ===")
print(f"= R:   {Rg:.5f}")
print(f"= mfp: {1.0 / (2 * Rg * T_ref):.5f}")
print(f"= Ma:  {Ma:.5f}")
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
    return 1.0 / (2 * Rg * T[i, j])


@ti.func
def is_inside(i: int, j: int):
    return i >= 0 and i < Nx and j >= 0 and j < Ny


@ti.func
def get_rho_E_at(i: int, j: int):
    return rho[i, j] * ti.math.dot(u[i, j], u[i, j]) / 2.0 + rho[i, j] * Rg * T[i, j] / (gamma - 1)


@ti.func
def get_W_at(i: int, j: int):
    return ti.Vector([
        rho[i, j],
        rho[i, j] * u[i, j][0],
        rho[i, j] * u[i, j][1],
        get_rho_E_at(i, j),
    ])


@ti.func
def get_grad_W(i: int, j: int):
    grad_rho = ti.Vector([0.0, 0.0])
    grad_rhoU1 = ti.Vector([0.0, 0.0])
    grad_rhoU2 = ti.Vector([0.0, 0.0])
    grad_rhoE = ti.Vector([0.0, 0.0])
    W_c = get_W_at(i, j)

    for k in ti.static(range(4)):
        n = N[k]
        i_n, j_n = i + n[0], j + n[1]
        W_n = W_c
        if is_inside(i_n, j_n):
            W_n = get_W_at(i_n, j_n)
        grad_rho += (W_n[0] + W_c[0]) * n / 2.0
        grad_rhoU1 += (W_n[1] + W_c[1]) * n / 2.0
        grad_rhoU2 += (W_n[2] + W_c[2]) * n / 2.0
        grad_rhoE += (W_n[3] + W_c[3]) * n / 2.0
    return grad_rho, grad_rhoU1, grad_rhoU2, grad_rhoE


@ti.func
def mgkfs_recursive_moments(T0, T1, u, mfp):
    """
    Compute the moments of the distribution function recursively.
    """
    m = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    m[0] = T0
    m[1] = T1
    for k in ti.static(range(5)):
        m[k + 2] = m[k + 1] * u + m[k] * (k + 1) / (2 * mfp)
    return m


@ti.func
def mgkfs_combined_moments(i: ti.template(), j: ti.template(), Mx, My):
    return Mx[i] * My[j]


@ti.func
def mgkfs_high_order_base(a, b, Mx, My, oi: ti.template(), oj: ti.template()):
    return (
        a[0] * mgkfs_combined_moments(1 + oi, 0 + oj, Mx, My)
        + a[1] * mgkfs_combined_moments(2 + oi, 0 + oj, Mx, My)
        + a[3] * mgkfs_combined_moments(3 + oi, 0 + oj, Mx, My) / 2.0
        + b[0] * mgkfs_combined_moments(0 + oi, 1 + oj, Mx, My)
        + (a[2] + b[1]) * mgkfs_combined_moments(1 + oi, 1 + oj, Mx, My)
        + b[3] * mgkfs_combined_moments(2 + oi, 1 + oj, Mx, My) / 2.0
        + b[2] * mgkfs_combined_moments(0 + oi, 2 + oj, Mx, My)
        + a[3] * mgkfs_combined_moments(1 + oi, 2 + oj, Mx, My) / 2.0
        + b[3] * mgkfs_combined_moments(0 + oi, 3 + oj, Mx, My) / 2.0
    )


@ti.func
def mgkfs_F0_base(B, Mx, My, oi: ti.template(), oj: ti.template()):
    return mgkfs_combined_moments(1, 0, Mx, My) - tau * (
        B[0] * mgkfs_combined_moments(1 + oi, 0 + oj, Mx, My)
        + B[1] * mgkfs_combined_moments(2 + oi, 0 + oj, Mx, My)
        + B[3] * mgkfs_combined_moments(3 + oi, 0 + oj, Mx, My) / 2.0
        + B[2] * mgkfs_combined_moments(1 + oi, 1 + oj, Mx, My)
        + B[3] * mgkfs_combined_moments(1 + oi, 2 + oj, Mx, My) / 2.0
    )


@ti.func
def mgkfs_solve_for_coeff(h0: dtype, h1: dtype, h2: dtype, h3: dtype, u1: dtype, u2: dtype, mfp: dtype):
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

    rho_L = (rho_L + rho_R) / 2
    rho_R = rho_L
    u1_L = (u1_L + u1_R) / 2
    u1_R = u1_L
    T_L = (T_L + T_R) / 2
    T_R = T_L
    mfp_L = (mfp_L + mfp_R) / 2
    mfp_R = mfp_L
    smfp_L = ti.sqrt(mfp_L)
    smfp_R = smfp_L

    T0 = erfc(-smfp_L * u1_L) / 2.0
    T1 = u1_L * T0 + ti.exp(-mfp_L * u1_L**2) / (2 * ti.sqrt(PI * mfp_L))
    M_L = mgkfs_recursive_moments(T0, T1, u1_L, mfp_L)

    T0 = erfc(smfp_R * u1_R) / 2.0
    T1 = u1_R * T0 - ti.exp(-mfp_R * u1_R**2) / (2 * ti.sqrt(PI * mfp_R))
    M_R = mgkfs_recursive_moments(T0, T1, u1_R, mfp_R)

    M_CL = mgkfs_recursive_moments(1, u2_L, u2_L, mfp_L)
    M_CR = mgkfs_recursive_moments(1, u2_R, u2_R, mfp_R)

    rho_i = M_L[0] * rho_L + M_R[0] * rho_R
    u1 = (M_L[1] * rho_L + M_R[1] * rho_R) / rho_i
    u2 = (M_L[0] * rho_L * u2_L + M_R[0] * rho_R * u2_R) / rho_i

    T0 = (u2_L**2 + (b - 1) * Rg * T_L) * M_L[0]
    T1 = (u2_R**2 + (b - 1) * Rg * T_R) * M_R[0]
    E_i = ((M_L[2] + T0) * rho_L + (M_R[2] + T1) * rho_R) / (2 * rho_i)
    e_i = E_i - (u1**2 + u2**2) / 2.0
    mfp_i = 1.0 / (2 * e_i * (gamma - 1))

    # -------------------------------------------------------------------------------------------------
    grad_rho_L, grad_rhoU1_L, grad_rhoU2_L, grad_rhoE_L = get_grad_W(i_L, j_L)

    i1 = N[face_id]
    i2 = N[(face_id + 1) % 4]

    drhodx_L = ti.math.dot(grad_rho_L, i1) / rho_L
    drhody_L = ti.math.dot(grad_rho_L, i2) / rho_L

    drhoU1dx_L = ti.math.dot(grad_rhoU1_L, i1) / rho_L
    drhoU1dy_L = ti.math.dot(grad_rhoU1_L, i2) / rho_L

    drhoU2dx_L = ti.math.dot(grad_rhoU2_L, i1) / rho_L
    drhoU2dy_L = ti.math.dot(grad_rhoU2_L, i2) / rho_L

    drhoEdx_L = ti.math.dot(grad_rhoE_L, i1) / rho_L
    drhoEdy_L = ti.math.dot(grad_rhoE_L, i2) / rho_L

    pUpX_L = R @ ti.Vector([drhoU1dx_L, drhoU2dx_L])
    pUpY_L = R @ ti.Vector([drhoU1dy_L, drhoU2dy_L])
    a0_L, a1_L, a2_L, a3_L = mgkfs_solve_for_coeff(
        drhodx_L,
        pUpX_L[0],
        pUpX_L[1],
        drhoEdx_L,
        u1_L,
        u2_L,
        mfp_L,
    )
    a_L = ti.Vector([a0_L, a1_L, a2_L, a3_L])
    b0_L, b1_L, b2_L, b3_L = mgkfs_solve_for_coeff(
        drhody_L,
        pUpY_L[0],
        pUpY_L[1],
        drhoEdy_L,
        u1_L,
        u2_L,
        mfp_L,
    )
    b_L = ti.Vector([b0_L, b1_L, b2_L, b3_L])

    grad_rho_R, grad_rhoU1_R, grad_rhoU2_R, grad_rhoE_R = get_grad_W(i_R, j_R)

    drhodx_R = ti.math.dot(grad_rho_R, i1) / rho_R
    drhody_R = ti.math.dot(grad_rho_R, i2) / rho_R

    drhoU1dx_R = ti.math.dot(grad_rhoU1_R, i1) / rho_R
    drhoU1dy_R = ti.math.dot(grad_rhoU1_R, i2) / rho_R

    drhoU2dx_R = ti.math.dot(grad_rhoU2_R, i1) / rho_R
    drhoU2dy_R = ti.math.dot(grad_rhoU2_R, i2) / rho_R

    drhoEdx_R = ti.math.dot(grad_rhoE_R, i1) / rho_R
    drhoEdy_R = ti.math.dot(grad_rhoE_R, i2) / rho_R

    pUpX_R = R @ ti.Vector([drhoU1dx_R, drhoU2dx_R])
    pUpY_R = R @ ti.Vector([drhoU1dy_R, drhoU2dy_R])

    a0_R, a1_R, a2_R, a3_R = mgkfs_solve_for_coeff(
        drhodx_R,
        pUpX_R[0],
        pUpX_R[1],
        drhoEdx_R,
        u1_R,
        u2_R,
        mfp_R,
    )
    a_R = ti.Vector([a0_R, a1_R, a2_R, a3_R])
    b0_R, b1_R, b2_R, b3_R = mgkfs_solve_for_coeff(
        drhody_R,
        pUpY_R[0],
        pUpY_R[1],
        drhoEdy_R,
        u1_R,
        u2_R,
        mfp_R,
    )
    b_R = ti.Vector([b0_R, b1_R, b2_R, b3_R])

    # -------------------------------------------------------------------------------------------------
    # Scheme 3
    h0_L = mgkfs_high_order_base(a_L, b_L, M_L, M_CL, 0, 0)
    h1_L = mgkfs_high_order_base(a_L, b_L, M_L, M_CL, 1, 0)
    h2_L = mgkfs_high_order_base(a_L, b_L, M_L, M_CL, 0, 1)
    h3_L = (mgkfs_high_order_base(a_L, b_L, M_L, M_CL, 2, 0) + mgkfs_high_order_base(a_L, b_L, M_L, M_CL, 0, 2)) / 2.0

    h0_R = mgkfs_high_order_base(a_R, b_R, M_R, M_CR, 0, 0)
    h1_R = mgkfs_high_order_base(a_R, b_R, M_R, M_CR, 1, 0)
    h2_R = mgkfs_high_order_base(a_R, b_R, M_R, M_CR, 0, 1)
    h3_R = (mgkfs_high_order_base(a_R, b_R, M_R, M_CR, 2, 0) + mgkfs_high_order_base(a_R, b_R, M_R, M_CR, 0, 2)) / 2.0

    h0 = -(rho_L * h0_L + rho_R * h0_R) / rho_i
    h1 = -(rho_L * h1_L + rho_R * h1_R) / rho_i
    h2 = -(rho_L * h2_L + rho_R * h2_R) / rho_i
    h3 = -(rho_L * h3_L + rho_R * h3_R) / rho_i
    B0, B1, B2, B3 = mgkfs_solve_for_coeff(h0, h1, h2, h3, u1, u2, mfp_i)

    B = ti.Vector([B0, B1, B2, B3])
    MX = mgkfs_recursive_moments(1, u1, u1, mfp_i)
    MY = mgkfs_recursive_moments(1, u2, u2, mfp_i)
    F0_I = rho_i * mgkfs_F0_base(B, MX, MY, 0, 0)
    F1_I = rho_i * mgkfs_F0_base(B, MX, MY, 1, 0)
    F2_I = rho_i * mgkfs_F0_base(B, MX, MY, 0, 1)
    F3_I = rho_i * (mgkfs_F0_base(B, MX, MY, 2, 0) + mgkfs_F0_base(B, MX, MY, 0, 2)) / 2.0

    F1_L = rho_L * mgkfs_high_order_base(a_L, b_L, M_L, M_CL, 2, 0)
    F2_L = rho_L * mgkfs_high_order_base(a_L, b_L, M_L, M_CL, 1, 1)
    F3_L = (
        rho_L
        * (mgkfs_high_order_base(a_L, b_L, M_L, M_CL, 3, 0) + mgkfs_high_order_base(a_L, b_L, M_L, M_CL, 1, 2))
        / 2.0
    )

    F1_R = rho_R * mgkfs_high_order_base(a_R, b_R, M_R, M_CR, 2, 0)
    F2_R = rho_R * mgkfs_high_order_base(a_R, b_R, M_R, M_CR, 1, 1)
    F3_R = (
        rho_R
        * (mgkfs_high_order_base(a_R, b_R, M_R, M_CR, 3, 0) + mgkfs_high_order_base(a_R, b_R, M_R, M_CR, 1, 2))
        / 2.0
    )

    F0 = F0_I
    F1 = F1_I - tau * (F1_L + F1_R)
    F2 = F2_I - tau * (F2_L + F2_R)
    F3 = F3_I - tau * (F3_L + F3_R)
    Fl = R_inv @ ti.Vector([F1, F2])
    return F0, Fl[0], Fl[1], F3


def mgkfs_init_flag():
    flag_np = flag.to_numpy()
    flag_np[:, Ny - 1] = NO_SLIP
    flag_np[0, :] = NO_SLIP
    flag_np[Nx - 1, :] = NO_SLIP
    flag_np[:, 0] = NO_SLIP
    flag.from_numpy(flag_np)


def mgkfs_init_u():
    u_np = u.to_numpy()
    hw = Nx // 6
    u_np[Nx // 2 - hw : Nx // 2 + hw, Ny // 2 - hw : Ny // 2 + hw, 1] = u_ref
    u.from_numpy(u_np)

    # T_np = T.to_numpy()
    # T_np[Nx // 2 - hw : Nx // 2 + hw, Ny // 2 - hw : Ny // 2 + hw] = 1.1
    # T.from_numpy(T_np)


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
        d_rE = 0.0

        for face_id in range(4):
            F0, F1, F2, F3 = mgkfs_compute_flux(i, j, face_id)
            d_rho += dt * F0
            d_m += dt * ti.Vector([F1, F2])
            d_rE += dt * F3

        # Perform the FVM update
        rho_prev = rho[i, j]
        u_prev = u[i, j]
        E_prev = get_rho_E_at(i, j)
        rho_new[i, j] = rho_prev - d_rho
        u_new[i, j] = (rho_prev * u_prev - d_m) / (rho_prev - d_rho)

        E_new = (rho_prev * E_prev - d_rE) / (rho_prev - d_rho)
        e_new = E_new - 0.5 * ti.math.dot(u_new[i, j], u_new[i, j])
        T_new[i, j] = e_new * (gamma - 1) / Rg


def save():
    with open("u_MGKFS_Re1000.pickle", "wb") as f:
        pickle.dump(u.to_numpy(), f, pickle.HIGHEST_PROTOCOL)
        col = cm.coolwarm(np.linalg.norm(u.to_numpy(), axis=-1) / (u_ref * 1.2))
        ti.tools.imwrite(col, "u_MGKFS_Re1000.png")
        print("Saved 'u_MGKFS_Re1000.pickle' and 'u_MGKFS_Re1000.png'")


def main():
    flag.fill(GAS)
    rho.fill(1.0)
    T.fill(T_ref)
    mgkfs_init_flag()
    mgkfs_init_u()
    gui = ti.GUI("M-GKFS", (Nx, Ny), background_color=0x212121)

    frame_id = 0
    while gui.running:
        if gui.get_event(ti.GUI.ESCAPE):
            gui.running = False
            save()
            break
        # migks_bc_stable()
        migks_step()
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
