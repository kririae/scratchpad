import pickle

import numpy as np
import taichi as ti
import torch
from matplotlib import cm

ti.init(arch=ti.gpu)

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
Nx = 2000 + 2
Ny = 1200 + 2
D = 2
K = 0
b = K + D
dtype = ti.f32

PI = 3.1415926535897
EPS = 1e-3
CFL = 0.6
c_s = 1  # Sound of speed, at sqrt(gamma*R*T)
u_ref = 0.4  # Reference velocity
T_ref = 1.0  # Reference temperature
dt = ti.field(dtype=dtype, shape=())
gamma = (b + 2) / b  # Heat ratio
Rg = c_s**2 / (gamma * T_ref)  # Gas constant
Ma = u_ref / c_s  # Mach number

Re = 220000
tau = 3 * u_ref * (Nx - 2) / Re
tau = 0
stride = 100

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
FORWARD = 4
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
rho_grad = ti.field(dtype=dtype, shape=(Nx, Ny))

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
def get_E_at(i: int, j: int):
    return ti.math.dot(u[i, j], u[i, j]) / 2.0 + Rg * T[i, j] / (gamma - 1)


@ti.func
def get_W_at(i: int, j: int, W):
    W_ret = W
    if is_inside(i, j):
        W_ret = ti.Vector([
            rho[i, j],
            rho[i, j] * u[i, j][0],
            rho[i, j] * u[i, j][1],
            rho[i, j] * get_E_at(i, j),
        ])
    return W_ret


@ti.func
def get_S_at(i: int, j: int, S):
    S_ret = S
    if is_inside(i, j):
        S_ret = ti.Vector([
            rho[i, j],
            u[i, j][0],
            u[i, j][1],
            get_E_at(i, j),
        ])
    return S_ret


# @ti.func
# def get_grad_W(i: int, j: int):
#     grad_rho = ti.Vector([0.0, 0.0])
#     grad_rhoU1 = ti.Vector([0.0, 0.0])
#     grad_rhoU2 = ti.Vector([0.0, 0.0])
#     grad_rhoE = ti.Vector([0.0, 0.0])
#     W_c = get_W_at(i, j, ti.Vector([0.0, 0.0, 0.0, 0.0]))

#     for k in ti.static(range(4)):
#         n = N[k]
#         i_n, j_n = i + n[0], j + n[1]
#         W_n = get_W_at(i_n, j_n, W_c)
#         grad_rho += (W_n[0] + W_c[0]) * n / 2.0
#         grad_rhoU1 += (W_n[1] + W_c[1]) * n / 2.0
#         grad_rhoU2 += (W_n[2] + W_c[2]) * n / 2.0
#         grad_rhoE += (W_n[3] + W_c[3]) * n / 2.0
#     return grad_rho, grad_rhoU1, grad_rhoU2, grad_rhoE


@ti.func
def get_grad_W(i: int, j: int):
    W0 = get_W_at(i + 0, j + 0, ti.Vector([0.0, 0.0, 0.0, 0.0]))
    W1 = get_W_at(i + 1, j + 0, W0)
    W2 = get_W_at(i + 0, j + 1, W0)
    W3 = get_W_at(i - 1, j + 0, W0)
    W4 = get_W_at(i + 0, j - 1, W0)
    W5 = get_W_at(i + 1, j + 1, W0)
    W6 = get_W_at(i - 1, j + 1, W0)
    W7 = get_W_at(i - 1, j - 1, W0)
    W8 = get_W_at(i + 1, j - 1, W0)
    c1o6 = 1.0 / 6.0
    c4o6 = 4.0 / 6.0
    dWdX = (c1o6 * (W5 - W6) + c4o6 * (W1 - W3) + c1o6 * (W8 - W7)) / 2.0
    dWdY = (c1o6 * (W6 - W7) + c4o6 * (W2 - W4) + c1o6 * (W5 - W8)) / 2.0
    return (
        ti.Vector([dWdX[0], dWdY[0]]),
        ti.Vector([dWdX[1], dWdY[1]]),
        ti.Vector([dWdX[2], dWdY[2]]),
        ti.Vector([dWdX[3], dWdY[3]]),
    )


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
    return mgkfs_combined_moments(1 + oi, 0 + oj, Mx, My) - tau * (
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
    a2 = 2 * mfp * r2 - u2 * a3
    a1 = 2 * mfp * r1 - u1 * a3
    a0 = h0 - u1 * a1 - u2 * a2 - a3 * r0 / 2
    return a0, a1, a2, a3


@ti.func
def venkatakrishnan_limiter(i: int, j: int, D2):
    W_c = get_W_at(i, j, ti.Vector([0.0, 0.0, 0.0, 0.0]))
    W_max = W_c
    W_min = W_c
    for k in ti.static(range(4)):
        i_n = i + N[k][0]
        j_n = j + N[k][1]
        if is_inside(i_n, j_n):
            W_n = get_W_at(i_n, j_n, W_max)
            W_max = ti.max(W_max, W_n)
            W_min = ti.min(W_min, W_n)
    D1_max = W_max - W_c
    D1_min = W_min - W_c
    D1 = ti.Vector([0.0, 0.0, 0.0, 0.0])
    phi = ti.Vector([0.0, 0.0, 0.0, 0.0])

    # select between D1_max and D1_min based on the sign of the gradient
    for k in ti.static(range(4)):
        if D2[k] > 0:
            D1[k] = D1_max[k]
        elif D2[k] < 0:
            D1[k] = D1_min[k]

    # Venkatakrishnan limiter
    epsilon2 = 5**3
    phi = 1.0 / D2 * ((D1**2 + epsilon2) * D2 + 2 * D2**2 * D1) / (D1**2 + 2 * D2**2 + D1 * D2 + epsilon2)
    for k in ti.static(range(4)):
        if D2[k] == 0:
            phi[k] = 1
    return phi


@ti.func
def mgkfs_initial_reconstruction_L(i: int, j: int, face_id: int):
    S_C = get_S_at(i, j, ti.Vector([0.0, 0.0, 0.0, 0.0]))
    S_R = get_S_at(i + N[face_id][0], j + N[face_id][1], S_C)
    D2 = (S_R - S_C) / 2.0
    # limiter = venkatakrishnan_limiter(i, j, D2)
    limiter = 0
    S = S_C + limiter * D2
    return S[0], ti.Vector([S[1], S[2]]), ti.max(S[3], EPS)


@ti.func
def mgkfs_initial_reconstruction_R(i: int, j: int, face_id: int):
    i += N[face_id][0]
    j += N[face_id][1]
    S_C = get_S_at(i, j, ti.Vector([0.0, 0.0, 0.0, 0.0]))
    S_L = get_S_at(i - N[face_id][0], j - N[face_id][1], S_C)
    D2 = -(S_C - S_L) / 2.0
    # limiter = venkatakrishnan_limiter(i, j, D2)
    limiter = 0
    S = S_C + limiter * D2
    return S[0], ti.Vector([S[1], S[2]]), ti.max(S[3], EPS)


@ti.func
def mgkfs_compute_gradients_and_coeffs(i, j, rho, u1, u2, mfp, i1, i2, R):
    grad_rho, grad_rhoU1, grad_rhoU2, grad_rhoE = get_grad_W(i, j)

    drhodx = ti.math.dot(grad_rho, i1) / rho
    drhody = ti.math.dot(grad_rho, i2) / rho

    drhoU1dx = ti.math.dot(grad_rhoU1, i1) / rho
    drhoU1dy = ti.math.dot(grad_rhoU1, i2) / rho

    drhoU2dx = ti.math.dot(grad_rhoU2, i1) / rho
    drhoU2dy = ti.math.dot(grad_rhoU2, i2) / rho

    drhoEdx = ti.math.dot(grad_rhoE, i1) / rho
    drhoEdy = ti.math.dot(grad_rhoE, i2) / rho

    pUpX = R @ ti.Vector([drhoU1dx, drhoU2dx])
    pUpY = R @ ti.Vector([drhoU1dy, drhoU2dy])

    a0, a1, a2, a3 = mgkfs_solve_for_coeff(drhodx, pUpX[0], pUpX[1], drhoEdx, u1, u2, mfp)
    b0, b1, b2, b3 = mgkfs_solve_for_coeff(drhody, pUpY[0], pUpY[1], drhoEdy, u1, u2, mfp)

    a = ti.Vector([a0, a1, a2, a3])
    b = ti.Vector([b0, b1, b2, b3])

    return a, b


@ti.func
def mgkfs_compute_flux(i: int, j: int, face_id: int):
    """
    Compute the flux on the face_id-th face
    """

    # -------------------------------------------------------------------------------------------------
    n_i = N[face_id]
    R = ti.Matrix([[n_i[0], n_i[1]], [-n_i[1], n_i[0]]], dt=ti.i32)
    R_inv = R.transpose()

    i_L, j_L = i, j
    i_R, j_R = i + N[face_id][0], j + N[face_id][1]

    rho_L, u_L, T_L = mgkfs_initial_reconstruction_L(i_L, j_L, face_id)
    rho_R, u_R, T_R = mgkfs_initial_reconstruction_R(i_L, j_L, face_id)
    u1_L, u2_L = R @ u_L
    u1_R, u2_R = R @ u_R
    mfp_L, mfp_R = 1.0 / (2 * Rg * T_L), 1.0 / (2 * Rg * T_R)

    # rho_L = rho[i_L, j_L]
    # rho_R = rho[i_R, j_R]
    # u_L = R @ u[i_L, j_L]
    # u_R = R @ u[i_R, j_R]
    # T_L = T[i_L, j_L]
    # T_R = T[i_R, j_R]

    # rho_L = (rho_L + rho_R) / 2
    # rho_R = rho_L
    # u_L = (u_L + u_R) / 2
    # u_R = u_L
    # u1_L, u2_L = u_L
    # u1_R, u2_R = u_R
    # T_L = (T_L + T_R) / 2
    # T_R = T_L
    # mfp_L, mfp_R = 1.0 / (2 * Rg * T_L), 1.0 / (2 * Rg * T_R)

    # -------------------------------------------------------------------------------------------------
    i1 = N[face_id]
    i2 = N[(face_id + 1) % 4]

    a_L, b_L = mgkfs_compute_gradients_and_coeffs(i_L, j_L, rho_L, u1_L, u2_L, mfp_L, i1, i2, R)
    a_R, b_R = mgkfs_compute_gradients_and_coeffs(i_R, j_R, rho_R, u1_R, u2_R, mfp_R, i1, i2, R)

    # -------------------------------------------------------------------------------------------------
    T0 = erfc(-ti.sqrt(mfp_L) * u1_L) / 2.0
    T1 = u1_L * T0 + ti.exp(-mfp_L * u1_L**2) / (2 * ti.sqrt(PI * mfp_L))
    M_L = mgkfs_recursive_moments(T0, T1, u1_L, mfp_L)

    T0 = erfc(ti.sqrt(mfp_R) * u1_R) / 2.0
    T1 = u1_R * T0 - ti.exp(-mfp_R * u1_R**2) / (2 * ti.sqrt(PI * mfp_R))
    M_R = mgkfs_recursive_moments(T0, T1, u1_R, mfp_R)

    rho_i = M_L[0] * rho_L + M_R[0] * rho_R
    u1 = (M_L[1] * rho_L + M_R[1] * rho_R) / rho_i
    u2 = (M_L[0] * rho_L * u2_L + M_R[0] * rho_R * u2_R) / rho_i

    T0 = (u2_L**2 + (b - 1) * Rg * T_L) * M_L[0]
    T1 = (u2_R**2 + (b - 1) * Rg * T_R) * M_R[0]
    E_i = ((M_L[2] + T0) * rho_L + (M_R[2] + T1) * rho_R) / (2 * rho_i)
    e_i = ti.max(E_i - (u1**2 + u2**2) / 2.0, EPS)
    mfp_i = 1.0 / (2 * e_i * (gamma - 1))

    M_CL = mgkfs_recursive_moments(1, u2_L, u2_L, mfp_L)
    M_CR = mgkfs_recursive_moments(1, u2_R, u2_R, mfp_R)

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

    F0 = rho_i * u1
    F1 = F1_I - tau * (F1_L + F1_R)
    F2 = F2_I - tau * (F2_L + F2_R)
    F3 = F3_I - tau * (F3_L + F3_R)
    Fl = R_inv @ ti.Vector([F1, F2])
    return ti.Vector([F0, Fl[0], Fl[1], F3])


@ti.kernel
def migks_init_sphere(radius: int, cx: int, cy: int):
    for i, j in ti.ndrange(Nx, Ny):
        if (i - cx) ** 2 + (j - cy) ** 2 < radius**2:
            flag[i, j] = SLIP


def mgkfs_init_flag():
    flag_np = flag.to_numpy()
    flag_np[:, 0] = SLIP
    flag_np[:, Ny - 1] = SLIP
    flag_np[0, :] = VELOCITY
    flag_np[Nx - 1, :] = FORWARD
    hw = Ny // 12
    flag_np[Nx // 3 - hw : Nx // 3 + hw, Ny // 2 - hw : Ny // 2 + hw] = NO_SLIP
    flag.from_numpy(flag_np)


def mgkfs_init_u():
    u_np = u.to_numpy()
    u_np[:, :, 0] = u_ref
    u.from_numpy(u_np)


def mgkfs_update_dt():
    """
    Update the time step size with CFL condition.
    """

    @torch.compile
    def torch_op(u_torch):
        return torch.max(torch.norm(u_torch, dim=-1))

    u_max = torch_op(u.to_torch())
    dt[None] = CFL * 1.0 / (max(u_max, u_ref) + c_s)


@ti.kernel
def mgkfs_boundary_condition():
    for i, j in ti.ndrange(Nx, Ny):
        if flag[i, j] == GAS:
            continue

        rho_, u_, T_ = 0.0, ti.Vector([0.0, 0.0]), 0.0
        cnt_ = 0

        for face_id in ti.static(range(4)):
            i_n = i + N[face_id][0]
            j_n = j + N[face_id][1]

            # Macroscopic variables are reflected from the boundary
            if is_inside(i_n, j_n) and flag[i_n, j_n] == GAS:
                rho_ += rho[i_n, j_n]
                T_ += T[i_n, j_n]
                cnt_ += 1
                if flag[i, j] == NO_SLIP:
                    u_ += -u[i_n, j_n]
                elif flag[i, j] == SLIP:
                    u_ += u[i_n, j_n] - 2 * (u[i_n, j_n] @ N[face_id]) * N[face_id]
                elif flag[i, j] == VELOCITY:
                    u_ += u_bc
                elif flag[i, j] == FORWARD:
                    u_ += u[i_n, j_n]

        if cnt_ > 0:
            rho[i, j] = rho_ / cnt_
            u[i, j] = u_ / cnt_
            T[i, j] = T_ / cnt_


@ti.kernel
def mgkfs_step():
    """
    Update the variables on-cell through FVM.
    """
    for i, j in ti.ndrange(Nx, Ny):
        if flag[i, j] != GAS:
            rho_new[i, j] = rho[i, j]
            u_new[i, j] = u[i, j]
            T_new[i, j] = T[i, j]
            continue

        dW = ti.Vector([0.0, 0.0, 0.0, 0.0])
        for face_id in range(4):
            dW += dt[None] * mgkfs_compute_flux(i, j, face_id)

        # -------------------------------------------------------------------------------------------------
        # Update the macroscopic variables
        rho_new[i, j] = rho[i, j] - dW[0]
        u_new[i, j] = (rho[i, j] * u[i, j] - dW[1:3]) / (rho[i, j] - dW[0])
        E_new = (rho[i, j] * get_E_at(i, j) - dW[3]) / (rho[i, j] - dW[0])
        e_new = ti.max(E_new - ti.math.dot(u_new[i, j], u_new[i, j]) / 2.0, EPS)
        T_new[i, j] = e_new * (gamma - 1) / Rg


@ti.kernel
def mgkfs_rho_gradient():
    for i, j in ti.ndrange(Nx, Ny):
        grad_rho, grad_rhoU1, grad_rhoU2, grad_rhoE = get_grad_W(i, j)
        rho_grad[i, j] = ti.sqrt(grad_rho[0] ** 2 + grad_rho[1] ** 2)


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

        mgkfs_update_dt()
        mgkfs_boundary_condition()
        mgkfs_step()

        rho.copy_from(rho_new)
        u.copy_from(u_new)
        T.copy_from(T_new)

        if frame_id % stride == 0:
            mgkfs_rho_gradient()
            if True:
                col = cm.coolwarm(np.linalg.norm(u.to_numpy(), axis=-1) / (u_ref * 1.2))
            else:
                col = cm.binary(rho_grad.to_numpy() * 4)
            gui.set_image(col)
            gui.show()
        frame_id += 1


if __name__ == "__main__":
    main()
