#!/usr/bin/env python3

# From taichi's course
import taichi as ti
from typing import Callable
import numpy as np
import scipy.sparse.linalg as linalg

ti.init(arch=ti.cpu, kernel_profiler=False, debug=False)

# global control
paused = False
damping_toggle = ti.field(ti.i32, ())
curser = ti.Vector.field(2, ti.f32, ())
picking = ti.field(ti.i32, ())
using_auto_diff = False
using_implicit = True

# procedurally setting up the cantilever
init_x, init_y = 0.1, 0.6
N_x = 20
N_y = 4
N = N_x*N_y
N_edges = (N_x-1)*N_y + N_x*(N_y - 1) + (N_x-1) * \
    (N_y-1)  # horizontal + vertical + diagonal springs
N_triangles = 2 * (N_x-1) * (N_y-1)
spacing = 1/32
curser_radius = spacing/2

# physical quantities
m = 1
g = 9.8
YoungsModulus = ti.field(ti.f32, ())
PoissonsRatio = ti.field(ti.f32, ())
LameMu = ti.field(ti.f32, ())
LameLa = ti.field(ti.f32, ())

# 1 Corotated linear elasticity
# 2 St. Venant-Kirchhoff
# 3 Neohookean elasticity
using_model: int = 3

# time-step size (for simulation, 16.7ms)
h = 16.7e-3
# substepping
substepping = 50
# time-step size (for time integration)
dh = h/substepping

# simulation components
x = ti.Vector.field(2, ti.f32, N, needs_grad=True)
v = ti.Vector.field(2, ti.f32, N)
total_energy = ti.field(ti.f32, (), needs_grad=True)
force = ti.Vector.field(2, ti.f32, N)
elements_Dm_inv = ti.Matrix.field(2, 2, ti.f32, N_triangles)
elements_V0 = ti.field(ti.f32, N_triangles)

# implicit integration
dx = ti.Vector.field(2, ti.f32, N)
grad_f = ti.Vector.field(2, ti.f32, N)
displacement = ti.Vector.field(2, ti.f32, N)

# geometric components
triangles = ti.Vector.field(3, ti.i32, N_triangles)
edges = ti.Vector.field(2, ti.i32, N_edges)


@ti.func
def ij_2_index(i, j): return i * N_y + j


# -----------------------meshing and init----------------------------
@ti.kernel
def meshing():
    # setting up triangles
    for i, j in ti.ndrange(N_x - 1, N_y - 1):
        # triangle id
        tid = (i * (N_y - 1) + j) * 2
        triangles[tid][0] = ij_2_index(i, j)
        triangles[tid][1] = ij_2_index(i + 1, j)
        triangles[tid][2] = ij_2_index(i, j + 1)

        tid = (i * (N_y - 1) + j) * 2 + 1
        triangles[tid][0] = ij_2_index(i, j + 1)
        triangles[tid][1] = ij_2_index(i + 1, j + 1)
        triangles[tid][2] = ij_2_index(i + 1, j)

    # setting up edges
    # edge id
    eid_base = 0

    # horizontal edges
    for i in range(N_x-1):
        for j in range(N_y):
            eid = eid_base+i*N_y+j
            edges[eid] = [ij_2_index(i, j), ij_2_index(i+1, j)]

    eid_base += (N_x-1)*N_y
    # vertical edges
    for i in range(N_x):
        for j in range(N_y-1):
            eid = eid_base+i*(N_y-1)+j
            edges[eid] = [ij_2_index(i, j), ij_2_index(i, j+1)]

    eid_base += N_x*(N_y-1)
    # diagonal edges
    for i in range(N_x-1):
        for j in range(N_y-1):
            eid = eid_base+i*(N_y-1)+j
            edges[eid] = [ij_2_index(i+1, j), ij_2_index(i, j+1)]


@ti.kernel
def initialize():
    YoungsModulus[None] = 1e6
    paused = False
    # init position and velocity
    for i, j in ti.ndrange(N_x, N_y):
        index = ij_2_index(i, j)
        x[index] = ti.Vector([init_x + i * spacing, init_y + j * spacing])
        v[index] = ti.Vector([0.0, 0.0])


@ti.func
def compute_D(i):
    a = triangles[i][0]
    b = triangles[i][1]
    c = triangles[i][2]
    return ti.Matrix.cols([x[b] - x[a], x[c] - x[a]])


@ti.func
def compute_dD(i):
    a = triangles[i][0]
    b = triangles[i][1]
    c = triangles[i][2]
    return ti.Matrix.cols([dx[b] - dx[a], dx[c] - dx[a]])


@ti.kernel
def initialize_elements():
    for i in range(N_triangles):
        Dm = compute_D(i)
        elements_Dm_inv[i] = Dm.inverse()
        elements_V0[i] = ti.abs(Dm.determinant())/2

# ----------------------core-----------------------------


@ti.func
def compute_R_2D(F):
    R, _ = ti.polar_decompose(F, ti.f32)
    return R


@ti.kernel
def compute_gradient():
    Eye = ti.Matrix.cols([[1.0, 0.0], [0.0, 1.0]]).cast(ti.f32)

    # clear gradient
    for i in force:
        force[i] = ti.Vector([0, 0])

    # gradient of elastic potential
    for i in range(N_triangles):
        Ds = compute_D(i)
        F = Ds@elements_Dm_inv[i]
        P = ti.Matrix.identity(dt=ti.f32, n=2)

        # Compute P with different models
        if ti.static(using_model == 1):
            # co-rotated linear elasticity model
            R = compute_R_2D(F)
            # first Piola-Kirchhoff tensor
            P = 2*LameMu[None]*(F-R) + LameLa[None] * \
                ((R.transpose())@F-Eye).trace()*R
        elif ti.static(using_model == 2):
            # StVK model
            E = 0.5*(F.transpose()@F - Eye)
            P = F @ (2*LameMu[None]*E + LameLa[None]*E.trace()*Eye)
        elif ti.static(using_model == 3):
            # Neohookean model
            J, Fp = F.determinant(), F.transpose().inverse()
            P = LameMu[None]*F - LameMu[None] * \
                Fp + LameLa[None]*ti.log(J+1e-5)*Fp
        else:
            assert False, "model not implemented"

        # assemble to gradient
        H = -elements_V0[i] * P @ (elements_Dm_inv[i].transpose())
        a, b, c = triangles[i][0], triangles[i][1], triangles[i][2]
        gb = ti.Vector([H[0, 0], H[1, 0]])
        gc = ti.Vector([H[0, 1], H[1, 1]])
        ga = -gb-gc
        force[a] += ga
        force[b] += gb
        force[c] += gc


@ti.kernel
def compute_total_energy():
    Eye = ti.Matrix.cols([[1.0, 0.0], [0.0, 1.0]]).cast(ti.f32)

    for i in range(N_triangles):
        Ds = compute_D(i)
        F = Ds @ elements_Dm_inv[i]
        element_energy_density: ti.f32 = 0.0

        # populate element_energy_density with different models
        if ti.static(using_model == 1):
            # co-rotated linear elasticity model
            R = compute_R_2D(F)
            element_energy_density = LameMu[None]*((F-R)@(F-R).transpose()).trace(
            ) + 0.5*LameLa[None]*(R.transpose()@F-Eye).trace()**2
        elif ti.static(using_model == 2):
            # StVK model
            E = 0.5*(F.transpose()@F - Eye)
            element_energy_density = LameMu[None] * \
                (E.transpose()@E).trace() + LameLa[None]*E.trace()**2
        elif ti.static(using_model == 3):
            # Neohookean model
            Fp = F.transpose()@F
            I1, _, J = Fp.trace(), (Fp@Fp).trace(), F.determinant()
            element_energy_density = LameMu[None]/2*(
                I1 - 3) - LameMu[None]*ti.log(J) + LameLa[None]/2*ti.log(J)**2
        else:
            assert False, "model not implemented"

        total_energy[None] += element_energy_density * elements_V0[i]


@ti.kernel
def compute_force_differential():
    """Given delta_x, compute the force differential and write into grad_f"""
    Eye = ti.Matrix.cols([[1.0, 0.0], [0.0, 1.0]]).cast(ti.f32)

    # clear gradient
    for i in grad_f:
        grad_f[i] = ti.Vector([0, 0])

    for i in range(N):
        Ds, dDs = compute_D(i), compute_dD(i)
        F, dF = Ds@elements_Dm_inv[i], dDs@elements_Dm_inv[i]
        E = 0.5*(F.transpose()@F - Eye)
        dE = 0.5*(dF.transpose()@F + F.transpose()@dF)
        dP = ti.Matrix.identity(dt=ti.f32, n=2)

        if ti.static(using_model == 1):
            # co-rotated linear elasticity model
            assert False, "implicit integration cannot be used with co-rotated linear elasticity model"
        elif ti.static(using_model == 2):
            # StVK model
            dP = dF@(2*LameMu[None]*E + LameLa[None]*E.trace()*Eye) + \
                F@(2*LameMu[None]*dE + LameLa[None]*dE.trace()*Eye)
        elif ti.static(using_model == 3):
            # Neohookean model
            Fp = F.inverse()
            J = F.determinant()
            dP = LameMu[None]*dF + (LameMu[None] - LameLa[None]*ti.log(
                J)) * Fp @ dF @ Fp + LameLa[None] * (Fp @ dF).trace() * Fp.transpose()

        dH = -elements_V0[i] * dP @ elements_Dm_inv[i].transpose()
        a, b, c = triangles[i][0], triangles[i][1], triangles[i][2]
        gb = ti.Vector([dH[0, 0], dH[1, 0]])
        gc = ti.Vector([dH[0, 1], dH[1, 1]])
        ga = -(gb + gc)
        grad_f[a] += ga
        grad_f[b] += gb
        grad_f[c] += gc


def compute_displacement():
    def semi_implicit_matvec(x: np.ndarray) -> np.ndarray:
        dx.from_numpy(x.reshape(-1, 2))
        compute_force_differential()
        return m/dh**2 * x - grad_f.to_numpy().reshape(-1)

    b = (m/dh*v.to_numpy() + force.to_numpy() - np.array([0.0, g])).reshape(-1)
    op = linalg.LinearOperator((N*2, N*2), semi_implicit_matvec)
    x, _ = linalg.cg(op, b, maxiter=3, atol=1e-5)
    displacement.from_numpy(x.reshape(-1, 2))


@ti.kernel
def update():
    # perform time integration
    for i in range(N):
        # elastic force + gravitation force, divding mass to get the acceleration
        if ti.static(using_implicit):
            v[i] = displacement[i]/dh
        elif ti.static(using_auto_diff):
            acc = -x.grad[i]/m - ti.Vector([0.0, g])
            v[i] += dh*acc
        else:
            acc = force[i]/m - ti.Vector([0.0, g])
            v[i] += dh*acc
        x[i] += v[i]*dh

    # explicit damping (ether drag)
    for i in v:
        if damping_toggle[None]:
            v[i] *= ti.exp(-dh*5)

    # enforce boundary condition
    for i in range(N):
        if picking[None]:
            r = x[i]-curser[None]
            if r.norm() < curser_radius:
                x[i] = curser[None]
                v[i] = ti.Vector([0.0, 0.0])
                pass

    for j in range(N_y):
        ind = ij_2_index(0, j)
        v[ind] = ti.Vector([0, 0])
        # rest pose attached to the wall
        x[ind] = ti.Vector([init_x, init_y + j * spacing])

    for i in range(N):
        if x[i][0] < init_x:
            x[i][0] = init_x
            v[i][0] = 0


@ti.kernel
def updateLameCoeff():
    E = YoungsModulus[None]
    nu = PoissonsRatio[None]
    LameLa[None] = E*nu / ((1+nu)*(1-2*nu))
    LameMu[None] = E / (2*(1+nu))


# init once and for all
meshing()
initialize()
initialize_elements()
updateLameCoeff()

gui = ti.GUI('Linear FEM', (800, 800))
while gui.running:

    picking[None] = 0

    # key events
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == 'r':
            initialize()
        elif e.key == '0':
            YoungsModulus[None] *= 1.1
        elif e.key == '9':
            YoungsModulus[None] /= 1.1
            if YoungsModulus[None] <= 0:
                YoungsModulus[None] = 0
        elif e.key == '8':
            PoissonsRatio[None] = PoissonsRatio[None] * \
                0.9+0.05  # slowly converge to 0.5
            if PoissonsRatio[None] >= 0.499:
                PoissonsRatio[None] = 0.499
        elif e.key == '7':
            PoissonsRatio[None] = PoissonsRatio[None]*1.1-0.05
            if PoissonsRatio[None] <= 0:
                PoissonsRatio[None] = 0
        elif e.key == ti.GUI.SPACE:
            paused = not paused
        elif e.key == 'd' or e.key == 'D':
            damping_toggle[None] = not damping_toggle[None]
        elif e.key == 'p' or e.key == 'P':  # step-forward
            for i in range(substepping):
                if using_auto_diff:
                    total_energy[None] = 0
                    with ti.ad.Tape(total_energy):
                        compute_total_energy()
                else:
                    compute_gradient()
                if using_implicit:
                    compute_displacement()
                update()
        updateLameCoeff()

    if gui.is_pressed(ti.GUI.LMB):
        curser[None] = gui.get_cursor_pos()
        picking[None] = 1

    # numerical time integration
    if not paused:
        for i in range(substepping):
            if using_auto_diff:
                total_energy[None] = 0
                with ti.ad.Tape(total_energy):
                    compute_total_energy()
            else:
                compute_gradient()
            if using_implicit:
                compute_displacement()
            update()

    # render
    pos = x.to_numpy()
    for i in range(N_edges):
        a, b = edges[i][0], edges[i][1]
        gui.line((pos[a][0], pos[a][1]),
                 (pos[b][0], pos[b][1]),
                 radius=1,
                 color=0xFFFF00)
    gui.line((init_x, 0.0), (init_x, 1.0), color=0xFFFFFF, radius=4)

    if picking[None]:
        gui.circle((curser[None][0], curser[None][1]),
                   radius=curser_radius*800, color=0xFF8888)

    # text
    gui.text(
        content=f'9/0: (-/+) Young\'s Modulus {YoungsModulus[None]:.1f}', pos=(0.6, 0.9), color=0xFFFFFF)
    gui.text(
        content=f'7/8: (-/+) Poisson\'s Ratio {PoissonsRatio[None]:.3f}', pos=(0.6, 0.875), color=0xFFFFFF)
    if damping_toggle[None]:
        gui.text(
            content='D: Damping On', pos=(0.6, 0.85), color=0xFFFFFF)
    else:
        gui.text(
            content='D: Damping Off', pos=(0.6, 0.85), color=0xFFFFFF)
    gui.show()

ti.profiler.print_kernel_profiler_info()
