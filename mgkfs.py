import pickle

import numpy as np
import taichi as ti
from matplotlib import cm

ti.init(arch=ti.cuda)

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


# -------------------------------------------------------------------------------------------------
# 1. [x] validate the equation with tau presented
# 2. [x] change the tau to the one in the paper
# 3. [x] add back the non-zero K
# 4. [x] consider the heat flux term in the paper
# 5. [x] switch to RK23/RK45 for time integration
# 6. implement the more elaborate boundary condition
# 7. [x] make the van Leer limiter work
# ... iteratively
# -------------------------------------------------------------------------------------------------


# Options:
# - "velocity"
# - "density"
# - "shockwave"
visualize = "density"

# Variables on-cell
Nx = 2000 + 2
Ny = 600 + 2
D = 2
K = 3
b = K + D
dtype = ti.f32

PI = 3.1415926535897
EPS = 1e-6
CFL = 0.13
c_s = 1  # Sound of speed, at sqrt(gamma*R*T)

u_ref = ti.field(dtype=dtype, shape=())
u_ref[None] = 2.80  # Reference velocity

T_ref = 1.0  # Reference temperature
S_ref = 110.4 / 285.0  # Reference Sutherland's constant
dt = ti.field(dtype=dtype, shape=())
gamma = (b + 2) / b
Rg = c_s**2 / (gamma * T_ref)  # Gas constant
Ma = u_ref[None] / c_s  # Mach number

viscosity = ti.field(dtype=dtype, shape=())
viscosity[None] = 1e-4
stride = 20

Pr = ti.field(dtype=dtype, shape=())
Pr[None] = 0.7

# Limiter get's sharper interface but less stability
# NOTE: in our case, van Albada limiter is stabler, working with the MUSCL scheme
#       but venkatakrishnan limiter is sharper
LM_NONE = 0
LM_VAN_ALBADA = 1
LM_VENKATAKRISHNAN = 2

limiter = LM_VAN_ALBADA

print("[mgkfs] === M-GKFS Parameters ===")
print(f"[mgkfs] = mfp:   {1.0 / (2 * Rg * T_ref):.5f}")
print(f"[mgkfs] = Ma:    {Ma:.5f}")
print("[mgkfs] =")
print(f"[mgkfs] = Re:    {u_ref[None] / viscosity[None]:.2f}")
print(f"[mgkfs] = gamma: {gamma:.5f}")
print("[mgkfs] =========================")

# Boundary conditions
BC_GAS = 0
BC_NO_SLIP = 1
BC_SLIP = 2
BC_VELOCITY = 3
BC_FORWARD = 4

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

# Runge-Kutta temporaries
rho_k1 = ti.field(dtype=dtype, shape=(Nx, Ny))
rho_k2 = ti.field(dtype=dtype, shape=(Nx, Ny))
rho_k3 = ti.field(dtype=dtype, shape=(Nx, Ny))
rho_k4 = ti.field(dtype=dtype, shape=(Nx, Ny))
u_k1 = ti.Vector.field(2, dtype=dtype, shape=(Nx, Ny))
u_k2 = ti.Vector.field(2, dtype=dtype, shape=(Nx, Ny))
u_k3 = ti.Vector.field(2, dtype=dtype, shape=(Nx, Ny))
u_k4 = ti.Vector.field(2, dtype=dtype, shape=(Nx, Ny))
T_k1 = ti.field(dtype=dtype, shape=(Nx, Ny))
T_k2 = ti.field(dtype=dtype, shape=(Nx, Ny))
T_k3 = ti.field(dtype=dtype, shape=(Nx, Ny))
T_k4 = ti.field(dtype=dtype, shape=(Nx, Ny))

field_type = ti.types.ndarray(dtype=dtype, ndim=2)

# Gradient of rho
grad_rho = ti.field(dtype=dtype, shape=(Nx, Ny))

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


# A approximation function from https://forums.developer.nvidia.com/t/an-accuracy-optimized-performance-competitive-implementation-of-erfcf/222654
@ti.func
def erfc_accu(x):
    TWO_TO_M24 = 5.9604644775390625e-8

    a = ti.abs(x)

    p = a + 2.0
    r = 1.0 / p
    q = a * r - 2.0 * r

    p = -4.00900841e-4
    p = p * q - 1.23049226e-3
    p = p * q + 1.31353654e-3
    p = p * q + 8.63232370e-3
    p = p * q - 8.05992913e-3
    p = p * q - 5.42046241e-2
    p = p * q + 1.64055422e-1
    p = p * q - 1.66031465e-1
    p = p * q - 9.27639827e-2
    p = p * q + 2.76978403e-1

    d = 2.0 * a + 1.0
    r = 1.0 / d
    q = (p + 1.0) * r

    s = a * a
    e = ti.exp(-s) * 2**24
    t = -a * a + s
    r = q * e + q * e * t
    r = r * TWO_TO_M24

    r = 0.0 if a > 10.0546875 else r
    r = 2.0 - r if x < 0.0 else r

    return r


@ti.func
def erfc(x):
    # return 1.0 - erf(x)
    return erfc_accu(x)


@ti.func
def is_inside(i: int, j: int):
    return i >= 0 and i < Nx and j >= 0 and j < Ny


@ti.func
def is_gas(i: int, j: int):
    ret = False
    if is_inside(i, j):
        ret = flag[i, j] == BC_GAS
    return ret


@ti.func
def get_mfp_at(i: int, j: int):
    return 1.0 / (2 * Rg * T[i, j])


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
def get_W_at_unsafe(i: int, j: int):
    return ti.Vector([
        rho[i, j],
        rho[i, j] * u[i, j][0],
        rho[i, j] * u[i, j][1],
        rho[i, j] * get_E_at(i, j),
    ])


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


@ti.func
def W_to_S(W):
    # S: [rho, u1, u2, E]
    # W: [rho, rho u1, rho u2, rho E]
    return ti.Vector([W[0], W[1] / W[0], W[2] / W[0], W[3] / W[0]])


@ti.func
def ensure_physical_state(W):
    W[0] = ti.max(W[0], 0)
    W[3] = ti.max(W[3], 0)
    return W


@ti.func
def E_to_T(E, u1, u2):
    e = E - (u1**2 + u2**2) / 2.0
    return e * (gamma - 1) / Rg


@ti.func
def get_dW_MUSCL(i: int, j: int, face_id: int):
    i_p, j_p = i + N[face_id][0], j + N[face_id][1]
    i_n, j_n = i - N[face_id][0], j - N[face_id][1]

    dW = ti.Vector([0.0, 0.0, 0.0, 0.0])
    W = get_W_at(i, j, ti.Vector([0.0, 0.0, 0.0, 0.0]))
    if is_gas(i_p, j_p) and is_gas(i_n, j_n):
        kappa = 1.0 / 3.0
        W_p = get_W_at(i_p, j_p, ti.Vector([0.0, 0.0, 0.0, 0.0]))
        W_n = get_W_at(i_n, j_n, ti.Vector([0.0, 0.0, 0.0, 0.0]))
        dW = (1 + kappa) / 2 * (W_p - W) + (1 - kappa) / 2 * (W - W_n)
    elif is_gas(i_p, j_p) and (not is_gas(i_n, j_n)):
        dW = get_W_at(i_p, j_p, ti.Vector([0.0, 0.0, 0.0, 0.0])) - W
    elif (not is_gas(i_p, j_p)) and is_gas(i_n, j_n):
        dW = W - get_W_at(i_n, j_n, ti.Vector([0.0, 0.0, 0.0, 0.0]))
    return dW


@ti.func
def get_grad_W_MUSCL(i: int, j: int):
    dWx = get_dW_MUSCL(i, j, 0)
    dWy = get_dW_MUSCL(i, j, 1)
    return ti.Matrix.rows([dWx, dWy])


@ti.func
def get_grad_W_green_gauss(i: int, j: int):
    grad_rho = ti.Vector([0.0, 0.0])
    grad_rhoU1 = ti.Vector([0.0, 0.0])
    grad_rhoU2 = ti.Vector([0.0, 0.0])
    grad_rhoE = ti.Vector([0.0, 0.0])
    W_c = get_W_at(i, j, ti.Vector([0.0, 0.0, 0.0, 0.0]))
    for k in ti.static(range(4)):
        n = N[k]
        i_n, j_n = i + n[0], j + n[1]
        W_n = get_W_at(i_n, j_n, W_c)
        grad_rho += (W_n[0] + W_c[0]) * n / 2.0
        grad_rhoU1 += (W_n[1] + W_c[1]) * n / 2.0
        grad_rhoU2 += (W_n[2] + W_c[2]) * n / 2.0
        grad_rhoE += (W_n[3] + W_c[3]) * n / 2.0
    return ti.Matrix([
        [grad_rho[0], grad_rhoU1[0], grad_rhoU2[0], grad_rhoE[0]],
        [grad_rho[1], grad_rhoU1[1], grad_rhoU2[1], grad_rhoE[1]],
    ])


@ti.func
def get_grad_W_isotropic_finite_difference(i: int, j: int):
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
    # return a 2x4 matrix
    return ti.Matrix.rows([dWdX, dWdY])


@ti.func
def get_grad_W_green_gauss_wide_kernel(i: int, j: int):
    grad_rho = ti.Vector([0.0, 0.0])
    grad_rhoU1 = ti.Vector([0.0, 0.0])
    grad_rhoU2 = ti.Vector([0.0, 0.0])
    grad_rhoE = ti.Vector([0.0, 0.0])

    W_c = get_W_at(i, j, ti.Vector([0.0, 0.0, 0.0, 0.0]))

    # fmt: off
    directions = ti.Matrix([
        [1, 1, 0, -1, -1, -1, 0, 1, 2, 2, 0, -2, -2, -2, 0, 2, 2, 1, -1, -2, -2, -1, 1, 2],
        [0, 1, 1, 1, 0, -1, -1, -1, 0, 2, 2, 2, 0, -2, -2, -2, 1, 2, 2, 1, -1, -2, -2, -1]
    ])

    wr1 = 0.15 
    wr2 = 0.08
    wr2_d = 0.04 

    weights = ti.Vector([
        wr1, wr1, wr1, wr1,
        wr1, wr1, wr1, wr1,
        wr2, wr2, wr2, wr2,
        wr2, wr2, wr2, wr2,
        wr2_d, wr2_d, wr2_d, wr2_d,
        wr2_d, wr2_d, wr2_d, wr2_d
    ])
    # fmt: on

    for k in ti.static(range(24)):
        dx = directions[0, k]
        dy = directions[1, k]
        i_n, j_n = i + dx, j + dy

        W_n = get_W_at(i_n, j_n, W_c)

        dist = ti.sqrt(dx * dx + dy * dy)
        n = ti.Vector([dx / dist, dy / dist])

        weight = weights[k]
        grad_rho += weight * (W_n[0] - W_c[0]) * n / dist
        grad_rhoU1 += weight * (W_n[1] - W_c[1]) * n / dist
        grad_rhoU2 += weight * (W_n[2] - W_c[2]) * n / dist
        grad_rhoE += weight * (W_n[3] - W_c[3]) * n / dist

    return ti.Matrix([
        [grad_rho[0], grad_rhoU1[0], grad_rhoU2[0], grad_rhoE[0]],
        [grad_rho[1], grad_rhoU1[1], grad_rhoU2[1], grad_rhoE[1]],
    ])


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
def mgkfs_zeta_moments(mfp):
    return ti.Vector([0.0, 0.0, K / (2 * mfp), 0.0, 3 * K / (4 * mfp**2) + K * (K - 1) / (4 * mfp**2)])


@ti.func
def mgkfs_combined_moments(i: ti.template(), j: ti.template(), Mx, My):
    return Mx[i] * My[j]


@ti.func
def mgkfs_M_base(a, b, Mx, My, oi: ti.template(), oj: ti.template()):
    return (
        0.0
        + a[0] * mgkfs_combined_moments(1 + oi, 0 + oj, Mx, My)
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
def mgkfs_M_zeta_base(a, b, Mx, My, Mz, oi: ti.template(), oj: ti.template()):
    return (
        0.0
        + a[3] * Mz[2] * mgkfs_combined_moments(1 + oi, 0 + oj, Mx, My)
        + b[3] * Mz[2] * mgkfs_combined_moments(0 + oi, 1 + oj, Mx, My)
    ) / 2.0


@ti.func
def mgkfs_M_zeta_residual(a, b, Mx, My, Mz, oi: ti.template()):
    return (
        0.0
        + a[0] * Mz[2] * mgkfs_combined_moments(1 + oi, 0, Mx, My) / 2.0
        + a[3] * Mz[4] * mgkfs_combined_moments(1 + oi, 0, Mx, My) / 4.0
        + a[1] * Mz[2] * mgkfs_combined_moments(2 + oi, 0, Mx, My) / 2.0
        + a[3] * Mz[2] * mgkfs_combined_moments(3 + oi, 0, Mx, My) / 2.0
        + b[0] * Mz[2] * mgkfs_combined_moments(0 + oi, 1, Mx, My) / 2.0
        + b[3] * Mz[4] * mgkfs_combined_moments(0 + oi, 1, Mx, My) / 4.0
        + a[2] * Mz[2] * mgkfs_combined_moments(1 + oi, 1, Mx, My) / 2.0
        + b[1] * Mz[2] * mgkfs_combined_moments(1 + oi, 1, Mx, My) / 2.0
        + b[3] * Mz[2] * mgkfs_combined_moments(2 + oi, 1, Mx, My) / 2.0
        + b[2] * Mz[2] * mgkfs_combined_moments(0 + oi, 2, Mx, My) / 2.0
        + a[3] * Mz[2] * mgkfs_combined_moments(1 + oi, 2, Mx, My) / 2.0
        + b[3] * Mz[2] * mgkfs_combined_moments(0 + oi, 3, Mx, My) / 2.0
    )


@ti.func
def mgkfs_F0_base(B, Mx, My, tau, oi: ti.template(), oj: ti.template()):
    return mgkfs_combined_moments(1 + oi, 0 + oj, Mx, My) - tau * (
        0.0
        + B[0] * mgkfs_combined_moments(1 + oi, 0 + oj, Mx, My)
        + B[1] * mgkfs_combined_moments(2 + oi, 0 + oj, Mx, My)
        + B[2] * mgkfs_combined_moments(1 + oi, 1 + oj, Mx, My)
        + B[3] * mgkfs_combined_moments(3 + oi, 0 + oj, Mx, My) / 2.0
        + B[3] * mgkfs_combined_moments(1 + oi, 2 + oj, Mx, My) / 2.0
    )


@ti.func
def mgkfs_F0_zeta_base(B, Mx, My, Mz, tau, oi: ti.template(), oj: ti.template()):
    return -tau * B[3] * Mz[2] * mgkfs_combined_moments(1 + oi, 0 + oj, Mx, My) / 2.0


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
    return ti.Vector([a0, a1, a2, a3])


@ti.func
def venkatakrishnan_limiter(i: int, j: int, D2):
    W_c = get_W_at(i, j, ti.Vector([0.0, 0.0, 0.0, 0.0]))
    W_max, W_min = W_c, W_c

    protector = False
    for k1, k2 in ti.ndrange(3, 3):
        i_n = i + k1 - 1
        j_n = j + k2 - 1
        if is_gas(i_n, j_n):
            W_n = get_W_at(i_n, j_n, W_max)
            W_max = ti.math.max(W_max, W_n)
            W_min = ti.math.min(W_min, W_n)

            # In case of near-boundary cells, we should not apply the limiter
            protector &= flag[i_n, j_n] != BC_GAS

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

    # apply Venkatakrishnan limiter
    # choose K0 = 0.3(conservative) or 5(aggressive)
    epsilon2 = 0.3**3
    for k in ti.static(range(4)):
        if ti.abs(D2[k]) < EPS:
            phi[k] = 1
        else:
            phi[k] = (
                ((D1[k] ** 2 + epsilon2) * D2[k] + 2 * D2[k] ** 2 * D1[k])
                / (D1[k] ** 2 + 2 * D2[k] ** 2 + D1[k] * D2[k] + epsilon2)
                / D2[k]
            )

    if protector:
        for k in ti.static(range(4)):
            phi[k] = 0.0
    return phi


@ti.func
def van_leer_limiter(s1, s2):
    limiter = ti.Vector([0.0, 0.0, 0.0, 0.0])
    for k in ti.static(range(4)):
        r = 0.0
        if ti.abs(s1[k]) > EPS:
            r = s2[k] / s1[k]
        else:
            r = s2[k] / EPS
        limiter[k] = (r + ti.abs(r)) / (1 + ti.abs(r))
    return limiter


@ti.func
def mgkfs_initial_reconstruction_L(i: int, j: int, face_id: int, dW):
    S = ti.Vector([0.0, 0.0, 0.0, 0.0])
    W_C = get_W_at(i, j, ti.Vector([0.0, 0.0, 0.0, 0.0]))
    dWT = dW.transpose()
    limiter = ti.Vector([1.0, 1.0, 1.0, 1.0])
    for k in range(4):
        limiter = ti.math.min(limiter, venkatakrishnan_limiter(i, j, dWT @ (N[k])))
    W = W_C + limiter * (dWT @ (N[face_id] / 2.0))
    S = W_to_S(W)
    return S


@ti.func
def mgkfs_initial_reconstruction_R(i: int, j: int, face_id: int, dW):
    i += N[face_id][0]
    j += N[face_id][1]
    W_C = get_W_at(i, j, ti.Vector([0.0, 0.0, 0.0, 0.0]))
    dWT = dW.transpose()
    limiter = ti.Vector([1.0, 1.0, 1.0, 1.0])
    for k in range(4):
        limiter = ti.math.min(limiter, venkatakrishnan_limiter(i, j, dWT @ (N[k])))
    W = W_C + limiter * (dWT @ (-N[face_id] / 2.0))
    S = W_to_S(W)
    return S


@ti.func
def mgkfs_initial_reconstruction(i: int, j: int, face_id: int, dW_L, dW_R):
    return mgkfs_initial_reconstruction_L(i, j, face_id, dW_L), mgkfs_initial_reconstruction_R(i, j, face_id, dW_R)


@ti.func
def mgkfs_MUSCL_reconstruction(i: int, j: int, face_id: int):
    i_p, j_p = i + N[face_id][0], j + N[face_id][1]
    i_pp, j_pp = i + 2 * N[face_id][0], j + 2 * N[face_id][1]
    i_n, j_n = i - N[face_id][0], j - N[face_id][1]

    W_L = ti.Vector([0.0, 0.0, 0.0, 0.0])
    W_R = W_L
    if is_gas(i_p, j_p) and is_gas(i_n, j_n) and is_gas(i_pp, j_pp) and limiter == LM_VAN_ALBADA:
        # Upwind biased reconstruction
        kappa = 1.0 / 3.0

        W_pp = get_W_at_unsafe(i_pp, j_pp)
        W_p = get_W_at_unsafe(i_p, j_p)
        W = get_W_at_unsafe(i, j)
        W_n = get_W_at_unsafe(i_n, j_n)

        s_p = W_p - W
        s_n = W - W_n

        # van Albada limiter
        s = (2 * s_p * s_n + 1e-6) / (s_n**2 + s_p**2 + 1e-6)

        W_L = W + s * ((1 + s * kappa) * s_p / 4 + (1 - s * kappa) * s_n / 4)
        W_R = W_p - s * ((1 - s * kappa) * (W_pp - W_p) / 4 + (1 + s * kappa) * s_p / 4)
    else:
        W_L = get_W_at_unsafe(i, j)
        W_R = get_W_at(i_p, j_p, W_L)

    S_L, S_R = ensure_physical_state(W_to_S(W_L)), ensure_physical_state(W_to_S(W_R))
    return S_L, S_R


@ti.func
def mgkfs_rotate_frame(W, R):
    u = R @ W[1:3]
    return ti.Vector([W[0], u[0], u[1], W[3]])


@ti.func
def mgkfs_compute_gradient_coeffs(dW, rho, u1, u2, mfp, R, R_inv):
    dW = R_inv @ dW
    dSdX = mgkfs_rotate_frame(dW[0, :], R) / rho
    dSdY = mgkfs_rotate_frame(dW[1, :], R) / rho

    a = mgkfs_solve_for_coeff(dSdX[0], dSdX[1], dSdX[2], dSdX[3], u1, u2, mfp)
    b = mgkfs_solve_for_coeff(dSdY[0], dSdY[1], dSdY[2], dSdY[3], u1, u2, mfp)
    return a, b


@ti.func
def mgkfs_compute_flux(i: int, j: int, face_id: int):
    """
    Compute the flux on the face_id-th face
    """

    # -------------------------------------------------------------------------------------------------
    # Compute the normal vector and the rotation matrix
    # -------------------------------------------------------------------------------------------------
    n_i = N[face_id]
    R = ti.Matrix([[n_i[0], n_i[1]], [-n_i[1], n_i[0]]], dt=ti.i32)
    R_inv = R.transpose()

    i_L, j_L = i, j
    i_R, j_R = i + N[face_id][0], j + N[face_id][1]

    # -------------------------------------------------------------------------------------------------
    # Compute spatial gradient on the L and R side with
    # 1. Green-Gauss method
    # 2. Isotropic finite difference
    # 3. High-order finite difference
    # -------------------------------------------------------------------------------------------------
    dW_L = get_grad_W_MUSCL(i_L, j_L)
    dW_R = get_grad_W_MUSCL(i_R, j_R)
    if False:
        dW_L = get_grad_W_green_gauss(i_L, j_L)
        dW_R = get_grad_W_green_gauss(i_R, j_R)

    S_L, S_R = ti.Vector([0.0, 0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0, 0.0])
    if limiter == LM_NONE or limiter == LM_VAN_ALBADA:
        S_L, S_R = mgkfs_MUSCL_reconstruction(i_L, j_L, face_id)
    else:
        S_L, S_R = mgkfs_initial_reconstruction(i_L, j_L, face_id, dW_L, dW_R)

    S_L = mgkfs_rotate_frame(S_L, R)
    S_R = mgkfs_rotate_frame(S_R, R)
    rho_L, u1_L, u2_L, E_L = S_L
    rho_R, u1_R, u2_R, E_R = S_R
    T_L, T_R = E_to_T(E_L, u1_L, u2_L), E_to_T(E_R, u1_R, u2_R)
    rho_L, rho_R = ti.max(rho_L, EPS), ti.max(rho_R, EPS)
    T_L, T_R = ti.max(T_L, EPS), ti.max(T_R, EPS)
    mfp_L, mfp_R = 1.0 / (2 * Rg * T_L), 1.0 / (2 * Rg * T_R)

    # -------------------------------------------------------------------------------------------------
    # Compute the spatial gradient coefficients with flux limiter
    # -------------------------------------------------------------------------------------------------
    a_L, b_L = mgkfs_compute_gradient_coeffs(dW_L, rho_L, u1_L, u2_L, mfp_L, R, R_inv)
    a_R, b_R = mgkfs_compute_gradient_coeffs(dW_R, rho_R, u1_R, u2_R, mfp_R, R, R_inv)

    # -------------------------------------------------------------------------------------------------
    # Perform reconstruction on the interface
    # Validated by Mathematica
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
    T_i = E_to_T(E_i, u1, u2)
    mfp_i = 1.0 / (2 * e_i * (gamma - 1))

    M_CL = mgkfs_recursive_moments(1, u2_L, u2_L, mfp_L)
    M_CR = mgkfs_recursive_moments(1, u2_R, u2_R, mfp_R)

    Mz_L = mgkfs_zeta_moments(mfp_L)
    Mz_I = mgkfs_zeta_moments(mfp_i)
    Mz_R = mgkfs_zeta_moments(mfp_R)

    # -------------------------------------------------------------------------------------------------
    # Apply compatibility conditions to solve for temporal derivatives `B`
    # -------------------------------------------------------------------------------------------------
    h0_L = mgkfs_M_base(a_L, b_L, M_L, M_CL, 0, 0) + mgkfs_M_zeta_base(a_L, b_L, M_L, M_CL, Mz_L, 0, 0)
    h1_L = mgkfs_M_base(a_L, b_L, M_L, M_CL, 1, 0) + mgkfs_M_zeta_base(a_L, b_L, M_L, M_CL, Mz_L, 1, 0)
    h2_L = mgkfs_M_base(a_L, b_L, M_L, M_CL, 0, 1) + mgkfs_M_zeta_base(a_L, b_L, M_L, M_CL, Mz_L, 0, 1)
    h3_L = (
        mgkfs_M_base(a_L, b_L, M_L, M_CL, 2, 0) + mgkfs_M_base(a_L, b_L, M_L, M_CL, 0, 2)
    ) / 2.0 + mgkfs_M_zeta_residual(a_L, b_L, M_L, M_CL, Mz_L, 0)

    h0_R = mgkfs_M_base(a_R, b_R, M_R, M_CR, 0, 0) + mgkfs_M_zeta_base(a_R, b_R, M_R, M_CR, Mz_R, 0, 0)
    h1_R = mgkfs_M_base(a_R, b_R, M_R, M_CR, 1, 0) + mgkfs_M_zeta_base(a_R, b_R, M_R, M_CR, Mz_R, 1, 0)
    h2_R = mgkfs_M_base(a_R, b_R, M_R, M_CR, 0, 1) + mgkfs_M_zeta_base(a_R, b_R, M_R, M_CR, Mz_R, 0, 1)
    h3_R = (
        mgkfs_M_base(a_R, b_R, M_R, M_CR, 2, 0) + mgkfs_M_base(a_R, b_R, M_R, M_CR, 0, 2)
    ) / 2.0 + mgkfs_M_zeta_residual(a_R, b_R, M_R, M_CR, Mz_R, 0)

    h0 = -(rho_L * h0_L + rho_R * h0_R) / rho_i
    h1 = -(rho_L * h1_L + rho_R * h1_R) / rho_i
    h2 = -(rho_L * h2_L + rho_R * h2_R) / rho_i
    h3 = -(rho_L * h3_L + rho_R * h3_R) / rho_i

    B = mgkfs_solve_for_coeff(h0, h1, h2, h3, u1, u2, mfp_i)

    # -------------------------------------------------------------------------------------------------
    # Compensate for `tau` on the interface based on [eq (5.52), Yang et al. 2020]
    # -------------------------------------------------------------------------------------------------
    p_L, p_R, p_I = rho_L / (2 * mfp_L), rho_R / (2 * mfp_R), rho_i / (2 * mfp_i)
    mu_ref = viscosity[None] * rho_i
    mu = mu_ref * ti.pow(T_i, 1.5) * (T_ref + S_ref) / (T_i + S_ref)
    tau = mu / p_I + ti.abs((p_L - p_R) / (p_L + p_R)) * dt[None]

    # -------------------------------------------------------------------------------------------------
    # Reconstruct the flux based on [eq (5.71), Yang et al. 2020]
    # Validated by Mathematica
    # -------------------------------------------------------------------------------------------------
    MX = mgkfs_recursive_moments(1, u1, u1, mfp_i)
    MY = mgkfs_recursive_moments(1, u2, u2, mfp_i)

    F1_I = rho_i * (mgkfs_F0_base(B, MX, MY, tau, 1, 0) + mgkfs_F0_zeta_base(B, MX, MY, Mz_I, tau, 1, 0))
    F2_I = rho_i * (mgkfs_F0_base(B, MX, MY, tau, 0, 1) + mgkfs_F0_zeta_base(B, MX, MY, Mz_I, tau, 0, 1))
    F3_I = rho_i * (
        (mgkfs_F0_base(B, MX, MY, tau, 2, 0) + mgkfs_F0_base(B, MX, MY, tau, 0, 2)) / 2.0
        + Mz_I[2] * mgkfs_combined_moments(1, 0, MX, MY) / 2.0
        - tau
        * (
            0.0
            + B[0] * Mz_I[2] * mgkfs_combined_moments(1, 0, MX, MY) / 2.0
            + B[3] * Mz_I[4] * mgkfs_combined_moments(1, 0, MX, MY) / 4.0
            + B[1] * Mz_I[2] * mgkfs_combined_moments(2, 0, MX, MY) / 2.0
            + B[3] * Mz_I[2] * mgkfs_combined_moments(3, 0, MX, MY) / 2.0
            + B[2] * Mz_I[2] * mgkfs_combined_moments(1, 1, MX, MY) / 2.0
            + B[3] * Mz_I[2] * mgkfs_combined_moments(1, 2, MX, MY) / 2.0
        )
    )

    F1_L = rho_L * (mgkfs_M_base(a_L, b_L, M_L, M_CL, 2, 0) + mgkfs_M_zeta_base(a_L, b_L, M_L, M_CL, Mz_L, 2, 0))
    F2_L = rho_L * (mgkfs_M_base(a_L, b_L, M_L, M_CL, 1, 1) + mgkfs_M_zeta_base(a_L, b_L, M_L, M_CL, Mz_L, 1, 1))
    F3_L = rho_L * (
        (mgkfs_M_base(a_L, b_L, M_L, M_CL, 3, 0) + mgkfs_M_base(a_L, b_L, M_L, M_CL, 1, 2)) / 2.0
        + mgkfs_M_zeta_residual(a_L, b_L, M_L, M_CL, Mz_L, 1)
    )

    F1_R = rho_R * (mgkfs_M_base(a_R, b_R, M_R, M_CR, 2, 0) + mgkfs_M_zeta_base(a_R, b_R, M_R, M_CR, Mz_R, 2, 0))
    F2_R = rho_R * (mgkfs_M_base(a_R, b_R, M_R, M_CR, 1, 1) + mgkfs_M_zeta_base(a_R, b_R, M_R, M_CR, Mz_R, 1, 1))
    F3_R = rho_R * (
        (mgkfs_M_base(a_R, b_R, M_R, M_CR, 3, 0) + mgkfs_M_base(a_R, b_R, M_R, M_CR, 1, 2)) / 2.0
        + mgkfs_M_zeta_residual(a_R, b_R, M_R, M_CR, Mz_R, 1)
    )

    F0 = rho_i * u1
    F1 = F1_I - tau * (F1_L + F1_R)
    F2 = F2_I - tau * (F2_L + F2_R)
    F3 = F3_I - tau * (F3_L + F3_R)

    # -------------------------------------------------------------------------------------------------
    # Correct the flux with Pr, see [eq (5.51), Yang et al. 2020] and [eq (5.78), Yang et al. 2020]
    # -------------------------------------------------------------------------------------------------
    q = F3 - u1 * F1 - u2 * F2 - u1 * (rho_i * E_i - rho_i * u1**2 - rho_i * u2**2)
    F3 = F3 + (1.0 / Pr[None] - 1) * q
    return mgkfs_rotate_frame(ti.Vector([F0, F1, F2, F3]), R_inv)


@ti.kernel
def mgkfs_init_u():
    for i, j in ti.ndrange(Nx, Ny):
        if flag[i, j] == BC_GAS:
            u[i, j] = ti.Vector([u_ref[None], 0.0])


def mgkfs_init():
    # -------------------------------------------------------------------------------------------------
    # Initialize the domain flag
    # -------------------------------------------------------------------------------------------------
    flag_np = flag.to_numpy()
    flag_np[:, :] = BC_GAS
    flag_np[0, :] = BC_VELOCITY
    flag_np[:, 0] = BC_SLIP
    flag_np[:, Ny - 1] = BC_SLIP
    flag_np[Nx - 1, :] = BC_FORWARD

    hw = 200
    flag_np[Nx // 3 : -1, 0:-1] = BC_SLIP
    flag.from_numpy(flag_np)

    # -------------------------------------------------------------------------------------------------
    # Initialize other macroscopic variables
    # -------------------------------------------------------------------------------------------------
    rho.fill(1.0)
    T.fill(T_ref)
    mgkfs_init_u()


@ti.kernel
def mgkfs_update_dt():
    """
    Update the time step size with CFL condition.
    """
    for i, j in ti.ndrange(Nx, Ny):
        c_max = ti.max(ti.sqrt(gamma * Rg * T[i, j]), c_s)
        u_max = ti.max(u[i, j].norm(), u_ref[None])
        ti.atomic_min(dt[None], CFL / (u_max + c_max))


@ti.kernel
def mgkfs_boundary_condition():
    for i, j in ti.ndrange(Nx, Ny):
        if flag[i, j] == BC_GAS:
            continue

        rho_, u_, E_ = 0.0, ti.Vector([0.0, 0.0]), 0.0
        cnt_ = 0
        for face_id in range(4):
            i_n = i + N[face_id][0]
            j_n = j + N[face_id][1]

            if is_inside(i_n, j_n) and flag[i_n, j_n] == BC_GAS:
                rho_ += rho[i_n, j_n]
                E_ += get_E_at(i_n, j_n)
                cnt_ += 1
                if flag[i, j] == BC_NO_SLIP:
                    u_ += -u[i_n, j_n]
                elif flag[i, j] == BC_SLIP:
                    u_ += u[i_n, j_n] - 2 * (u[i_n, j_n] @ N[face_id]) * N[face_id]
                elif flag[i, j] == BC_VELOCITY:
                    u_ += ti.Vector([u_ref[None], 0.0])
                elif flag[i, j] == BC_FORWARD:
                    u_ += u[i_n, j_n]

        # -------------------------------------------------------------------------------------------------
        # According to Kun Xu, in using ghost cells, rho/T are reflected
        # -------------------------------------------------------------------------------------------------
        if cnt_ > 0:
            rho[i, j] = rho_ / cnt_
            u[i, j] = u_ / cnt_

            E_average = E_ / cnt_
            T[i, j] = ti.max(E_to_T(E_average, u_[0], u_[1]), 0)


@ti.func
def mgkfs_update_differential(i: int, j: int, dW, rho_k, u_k, T_k):
    """
    Update the macroscopic variables.
    """
    rho_new = rho[i, j] - dW[0]
    u_new = (rho[i, j] * u[i, j] - dW[1:3]) / (rho[i, j] - dW[0])
    E_new = (rho[i, j] * get_E_at(i, j) - dW[3]) / (rho[i, j] - dW[0])
    e_new = ti.max(E_new - ti.math.dot(u_new, u_new) / 2.0, 0)
    T_new = e_new * (gamma - 1) / Rg

    rho_k[i, j] = rho_new - rho[i, j]
    u_k[i, j] = u_new - u[i, j]
    T_k[i, j] = T_new - T[i, j]


@ti.kernel
def mgkfs_calc_differential(rho_k: ti.template(), u_k: ti.template(), T_k: ti.template()):
    """
    Update the variables on-cell through FVM.
    """
    for i, j in ti.ndrange(Nx, Ny):
        if flag[i, j] != BC_GAS:
            rho_k[i, j] = 0.0
            u_k[i, j] = ti.Vector([0.0, 0.0])
            T_k[i, j] = 0.0
            continue
        dW = ti.Vector([0.0, 0.0, 0.0, 0.0])
        for face_id in range(4):
            dW += mgkfs_compute_flux(i, j, face_id)
        mgkfs_update_differential(i, j, dW, rho_k, u_k, T_k)


@ti.kernel
def mgkfs_update_macroscopic(factor: dtype, rho_k: ti.template(), u_k: ti.template(), T_k: ti.template()):
    for i, j in ti.ndrange(Nx, Ny):
        rho[i, j] += factor * dt[None] * rho_k[i, j]
        u[i, j] += factor * dt[None] * u_k[i, j]
        T[i, j] += factor * dt[None] * T_k[i, j]


@ti.kernel
def mgkfs_update_macroscopic_rk2(
    rho_k1: ti.template(),
    rho_k2: ti.template(),
    u_k1: ti.template(),
    u_k2: ti.template(),
    T_k1: ti.template(),
    T_k2: ti.template(),
):
    for i, j in ti.ndrange(Nx, Ny):
        rho[i, j] += dt[None] * (-rho_k1[i, j] + rho_k2[i, j]) / 2
        u[i, j] += dt[None] * (-u_k1[i, j] + u_k2[i, j]) / 2
        T[i, j] += dt[None] * (-T_k1[i, j] + T_k2[i, j]) / 2


@ti.kernel
def mgkfs_update_macroscopic_rk4(
    rho_k1: ti.template(),
    rho_k2: ti.template(),
    rho_k3: ti.template(),
    rho_k4: ti.template(),
    u_k1: ti.template(),
    u_k2: ti.template(),
    u_k3: ti.template(),
    u_k4: ti.template(),
    T_k1: ti.template(),
    T_k2: ti.template(),
    T_k3: ti.template(),
    T_k4: ti.template(),
):
    for i, j in ti.ndrange(Nx, Ny):
        rho[i, j] += dt[None] * (rho_k1[i, j] + 2 * rho_k2[i, j] - 4 * rho_k3[i, j] + rho_k4[i, j]) / 6
        u[i, j] += dt[None] * (u_k1[i, j] + 2 * u_k2[i, j] - 4 * u_k3[i, j] + u_k4[i, j]) / 6
        T[i, j] += dt[None] * (T_k1[i, j] + 2 * T_k2[i, j] - 4 * T_k3[i, j] + T_k4[i, j]) / 6


def mgkfs_euler_step():
    mgkfs_boundary_condition()
    mgkfs_calc_differential(rho_k1, u_k1, T_k1)
    mgkfs_update_macroscopic(1.0, rho_k1, u_k1, T_k1)


def mgkfs_rk2_step():
    mgkfs_boundary_condition()
    mgkfs_calc_differential(rho_k1, u_k1, T_k1)
    mgkfs_update_macroscopic(1.0, rho_k1, u_k1, T_k1)
    mgkfs_boundary_condition()
    mgkfs_calc_differential(rho_k2, u_k2, T_k2)
    mgkfs_update_macroscopic_rk2(rho_k1, rho_k2, u_k1, u_k2, T_k1, T_k2)


def mgkfs_rk4_step():
    # This is quite special, as we need to calculate the differential for each stage
    # k1=f(yn)
    # k2=f(yn+0.5*dt*k1)
    # k3=f(yn+0.5*dt*k2)
    # k4=f(yn+dt*k3)
    def mgkfs_restore_state():
        rho.copy_from(rho_k4)
        u.copy_from(u_k4)
        T.copy_from(T_k4)

    rho_k4.copy_from(rho)
    u_k4.copy_from(u)
    T_k4.copy_from(T)

    mgkfs_boundary_condition()
    mgkfs_calc_differential(rho_k1, u_k1, T_k1)
    mgkfs_update_macroscopic(0.5, rho_k1, u_k1, T_k1)
    mgkfs_boundary_condition()
    mgkfs_calc_differential(rho_k2, u_k2, T_k2)
    mgkfs_restore_state()
    mgkfs_update_macroscopic(0.5, rho_k2, u_k2, T_k2)
    mgkfs_boundary_condition()
    mgkfs_calc_differential(rho_k3, u_k3, T_k3)
    mgkfs_restore_state()
    mgkfs_update_macroscopic(1.0, rho_k3, u_k3, T_k3)
    mgkfs_boundary_condition()
    mgkfs_calc_differential(rho_k4, u_k4, T_k4)
    mgkfs_update_macroscopic_rk4(rho_k1, rho_k2, rho_k3, rho_k4, u_k1, u_k2, u_k3, u_k4, T_k1, T_k2, T_k3, T_k4)


@ti.kernel
def mgkfs_calculate_gradient():
    for i, j in ti.ndrange(Nx, Ny):
        dW = get_grad_W_green_gauss(i, j)
        grad_rho[i, j] = ti.Vector([dW[0, 0], dW[1, 0]]).norm()


def save():
    with open("u_MGKFS_Re1000.pickle", "wb") as f:
        pickle.dump(u.to_numpy(), f, pickle.HIGHEST_PROTOCOL)
        col = cm.coolwarm(np.linalg.norm(u.to_numpy(), axis=-1) / (u_ref[None] * 1.2))
        ti.tools.imwrite(col, "u_MGKFS_Re1000.png")
        print("Saved 'u_MGKFS_Re1000.pickle' and 'u_MGKFS_Re1000.png'")
        exit(0)


def main():
    gui = ti.GUI("M-GKFS", (Nx, Ny), background_color=0x212121)
    mgkfs_init()

    frame_id = 0
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                gui.running = False
                save()

        # -------------------------------------------------------------------------------------------------
        # This is what you'd expect in a typical FVM solver
        # -------------------------------------------------------------------------------------------------
        mgkfs_boundary_condition()
        dt[None] = 1.0
        mgkfs_update_dt()

        if False:
            mgkfs_euler_step()
        elif True:
            mgkfs_rk2_step()
        else:
            mgkfs_rk4_step()

        if frame_id % stride == 0:
            if visualize == "velocity":
                col = cm.turbo(np.linalg.norm(u.to_numpy(), axis=-1) / (u_ref[None] * 1.2))
            elif visualize == "density":
                col = cm.jet(np.abs(rho.to_numpy() - 1.0) / 2.0)
            elif visualize == "shockwave":
                mgkfs_calculate_gradient()
                col = cm.binary(grad_rho.to_numpy())
            else:
                raise ValueError(f"Unknown visualization mode: {visualize}")

            if False:
                ti.tools.imwrite(col, f".gitignore.experiments/MGKFS_{frame_id:05d}.png")

            gui.set_image(col)
            gui.show()
        frame_id += 1


if __name__ == "__main__":
    main()
