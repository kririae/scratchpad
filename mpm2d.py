#!/usr/bin/env python3
from typing import Any, Tuple

import warp as wp
import numpy as np
import taichi as ti  # I need taichi's OpenGL render

WATER = wp.constant(wp.uint8(0))
SOFT = wp.constant(wp.uint8(1))
SOLID = wp.constant(wp.uint8(2))
GAS = wp.constant(wp.uint8(3))


@wp.func
def is_inside(i: int, j: int, width: Any, height: Any) -> bool:
    return float(i) >= 0 and float(i) < float(width) and float(j) >= 0 and float(j) < float(height)


@wp.func
def get_at_position(
    i: int,
    j: int,
    f: wp.array2d(dtype=Any),
) -> Any:
    width, height = f.shape[0], f.shape[1]
    if is_inside(i, j, width, height):
        return f[i, j]
    else:
        return type(f[0, 0])(0.0)


@wp.func
def is_solid(i: int, j: int, f: wp.array2d(dtype=wp.uint8)) -> bool:
    if not is_inside(i, j, f.shape[0], f.shape[1]):
        return False
    return f[i, j] == SOLID


@wp.func
def is_water(i: int, j: int, f: wp.array2d(dtype=wp.uint8)) -> bool:
    if not is_inside(i, j, f.shape[0], f.shape[1]):
        return False
    return f[i, j] == WATER


@wp.func
def is_gas(i: int, j: int, f: wp.array2d(dtype=wp.uint8)) -> bool:
    if not is_inside(i, j, f.shape[0], f.shape[1]):
        return False
    return f[i, j] == GAS


@wp.func
def get_bspline_coeff(p: wp.vec2):
    # This implementation sticks to cell-centered grid, where both pressure and
    # velocity are stored at the center of the cell.
    # Calculate the corresponding grid position of this particle,
    base = wp.vec2(wp.floor(p.x), wp.floor(p.y)) + wp.vec2(0.5, 0.5)
    ox = wp.vec3(p.x - (base.x - 1.0), p.x - (base.x), p.x - (base.x + 1.0))
    oy = wp.vec3(p.y - (base.y - 1.0), p.y - (base.y), p.y - (base.y + 1.0))

    # Fuck WARP
    ox.x = wp.abs(ox.x)
    ox.y = wp.abs(ox.y)
    ox.z = wp.abs(ox.z)
    oy.x = wp.abs(oy.x)
    oy.y = wp.abs(oy.y)
    oy.z = wp.abs(oy.z)

    # Quadratic B-spline kernel
    nox = wp.vec3(0.5*(1.5-ox.x)*(1.5-ox.x), 0.75 -
                  ox.y*ox.y, 0.5*(1.5-ox.z)*(1.5-ox.z))
    noy = wp.vec3(0.5*(1.5-oy.x)*(1.5-oy.x), 0.75 -
                  oy.y*oy.y, 0.5*(1.5-oy.z)*(1.5-oy.z))

    return base, nox, noy


@wp.func
def svd2(A: wp.mat22):
    U3 = wp.mat33(0.0)
    S3 = wp.vec3(0.0)
    V3 = wp.mat33(0.0)
    A3 = wp.mat33(A[0, 0], A[0, 1], 0.0, A[1, 0], A[1, 1], 0.0, 0.0, 0.0, 0.0)
    wp.svd3(A3, U3, S3, V3)

    U = wp.mat22(U3[0, 0], U3[0, 1], U3[1, 0], U3[1, 1])
    S = wp.mat22(S3[0], 0.0, 0.0, S3[1])
    V = wp.mat22(V3[0, 0], V3[0, 1], V3[1, 0], V3[1, 1])
    return U, S, V


class MaterialPointSolver2D:
    def __init__(self, width: int, height: int) -> None:
        wp.init()

        self.width_ = width
        self.height_ = height
        self.shape_ = (width, height)
        self.boundary_ = 5

        self.dt_ = 1e-5

        # Particle positions and velocities
        # Incorrect AoS layout, fuck WARP again
        p, f = self._init_case()
        self.p_ = wp.from_numpy(p, dtype=wp.vec2, device="cuda")
        self.f_ = wp.from_numpy(f, dtype=wp.uint8, device="cuda")
        self.u_ = wp.zeros(self.p_.shape, dtype=wp.vec2)
        self.c_ = wp.zeros(self.p_.shape, dtype=wp.mat22)
        self.F_ = wp.zeros(self.p_.shape, dtype=wp.mat22)
        self.J_ = wp.ones(self.p_.shape, dtype=wp.float32)

        self.pn_ = self.p_.numpy()
        self.fn_ = self.f_.numpy()

        self.ug_ = wp.zeros((width, height), dtype=wp.vec2)
        self.mg_ = wp.zeros((width, height), dtype=wp.float32)
        self.fg_ = wp.zeros((width, height), dtype=wp.uint8)

        self._init()

        with wp.ScopedCapture() as cap:
            self._set_flag()
            self._p2g()
            self._grid_op()
            self._g2p()
        self.graph_ = cap.graph

    def step(self) -> None:
        # The solver is composed of
        # ----------------------------
        # (1) Particle to Grid Transfer(P2G)
        # (2.0, not implemented) Apply force and then boundary extrapolation
        #                        (note that free-surface extrapolation is automatically performed)
        # (2.x) Projection
        # (2.1, not implemented) Apply boundary extrapolation
        # (3) Grid to Particle Transfer(G2P)
        # (4) Particle Advection(by semi-lagrangian)
        with wp.ScopedTimer("step"):
            wp.capture_launch(self.graph_)
            wp.synchronize()

    def anim_frame_particle(self, n_steps=32):
        for _ in range(n_steps):
            self.step()
        np.copyto(self.pn_, self.p_.numpy())
        np.copyto(self.fn_, self.f_.numpy())

    def to_fluid(self):
        self.f_.fill_(WATER)

    def to_soft(self):
        self.u_.fill_(0.0)
        self.f_.fill_(SOFT)
        self.J_.fill_(1.0)
        wp.launch(self._init_particle_kernel, dim=self.p_.shape[0],
                  inputs=[self.F_])
        wp.synchronize()

    def _init_case(self) -> np.ndarray:
        def generate(xl, xr, yl, yr, type, density=1600):
            points_x = (np.linspace(xl, xr, int(
                (xr-xl)*density))) * self.width_
            points_y = np.linspace(yl, yr, int((yr-yl)*density)) * self.height_
            grid_x, grid_y = np.meshgrid(points_x, points_y)
            res = np.vstack((grid_x.flatten(), grid_y.flatten())).T
            f = np.zeros_like(res[:, 0])
            f.fill(type)

            return res, f

        comp = []
        comp.append(generate(0.05, 0.25, 0.02, 0.62, WATER))
        comp.append(generate(0.75, 0.95, 0.02, 0.62, SOFT, density=800))

        res = np.vstack([c[0] for c in comp])
        f = np.hstack([c[1] for c in comp])

        # concatenate the two cases
        return (res + np.random.rand(*res.shape) - 0.5), f

    def _init(self) -> None:
        self.fg_.fill_(GAS)
        wp.launch(self._init_grid_kernel, dim=self.shape_,
                  inputs=[self.fg_, self.boundary_])
        wp.launch(self._init_particle_kernel, dim=self.p_.shape[0],
                  inputs=[self.F_])
        wp.synchronize()

    def _set_flag(self) -> None:
        wp.launch(self._set_all_flag, dim=self.shape_, inputs=[self.fg_])
        wp.launch(self._set_fluid_flag,
                  dim=self.p_.shape[0], inputs=[self.p_, self.fg_])

    def _p2g(self) -> None:
        self.ug_.fill_(0.0)
        self.mg_.fill_(0.0)
        wp.launch(self._p2g_apic_kernel, dim=self.p_.shape[0],
                  inputs=[self.u_, self.p_, self.c_, self.F_, self.J_, self.f_,
                          self.ug_, self.mg_, self.dt_])
        wp.launch(self._p2g_postprocess_kernel, dim=self.shape_,
                  inputs=[self.ug_, self.mg_, self.fg_])

    def _grid_op(self) -> None:
        wp.launch(self._g_force_and_boundary_kernel, dim=self.shape_, inputs=[
            self.ug_, self.fg_])

    def _g2p(self) -> None:
        wp.launch(self._g2p_apic_kernel, dim=self.p_.shape[0],
                  inputs=[self.u_, self.p_, self.c_, self.F_, self.J_,
                          self.ug_,
                          self.boundary_, self.dt_])

    @wp.kernel
    def _init_grid_kernel(fg: wp.array2d(dtype=wp.uint8), boundary: int) -> None:
        i, j = wp.tid()
        width, height = fg.shape[0], fg.shape[1]
        if (i < boundary or i >= width - boundary) or (j < boundary or j >= height - boundary):
            fg[i, j] = SOLID

    @wp.kernel
    def _init_particle_kernel(F: wp.array1d(dtype=wp.mat22)) -> None:
        i = wp.tid()
        F[i] = wp.mat22(1.0, 0.0, 0.0, 1.0)

    @wp.kernel
    def _set_all_flag(
        fg: wp.array2d(dtype=wp.uint8)
    ) -> None:
        i, j = wp.tid()
        if fg[i, j] != SOLID:
            fg[i, j] = GAS

    @wp.kernel
    def _set_fluid_flag(
        p: wp.array1d(dtype=wp.vec2),
        fg: wp.array2d(dtype=wp.uint8),
    ) -> None:
        i = wp.tid()
        pp = p[i]
        base = wp.vec2(wp.floor(pp.x), wp.floor(pp.y))

        if fg[int(base.x), int(base.y)] != SOLID:
            fg[int(base.x), int(base.y)] = WATER

    @wp.kernel
    def _p2g_apic_kernel(
        u: wp.array1d(dtype=wp.vec2),
        p: wp.array1d(dtype=wp.vec2),
        c: wp.array1d(dtype=wp.mat22),
        F: wp.array1d(dtype=wp.mat22),
        J: wp.array1d(dtype=wp.float32),
        f: wp.array1d(dtype=wp.uint8),
        ug: wp.array2d(dtype=wp.vec2),
        mg: wp.array2d(dtype=wp.float32),
        dt: float,
    ) -> None:
        i = wp.tid()

        up = u[i]
        pp = p[i]
        aff = c[i]
        base, nox, noy = get_bspline_coeff(p[i])

        # Perform MLS-MPM particle advection
        F[i] = (wp.mat22(1.0, 0.0, 0.0, 1.0) + dt*aff)@F[i]
        J[i] = (1.0 + dt*wp.trace(aff)) * J[i]

        E = 4e9
        nu = 0.20
        LameLa = E*nu / ((1.0+nu)*(1.0-2.0*nu))
        LameMu = E / (2.0*(1.0+nu))

        if f[i] == WATER:
            w_factor = (dt * E) * (1.0 - J[i])
            stress = wp.mat22(w_factor, 0.0, 0.0, w_factor)
        elif f[i] == SOFT:
            TF = wp.transpose(F[i])
            U, _, V = svd2(F[i])
            R = U @ wp.transpose(V)
            P = 2.0*LameMu*(F[i] - R)@TF + LameLa*(J[i] - 1.0) * \
                J[i]*wp.mat22(1.0, 0.0, 0.0, 1.0)
            stress = -4.0*dt*P
        else:
            invTF = wp.inverse(wp.transpose(F[i]))
            stress = -dt * \
                ((LameMu * (F[i] - invTF)) + LameLa * wp.log(wp.clamp(J[i], 1e-3, 1e3))
                 * invTF @ wp.transpose(F[i]))

        aff = stress + aff
        for x in range(3):
            for y in range(3):
                bi = int(base.x) - 1 + x
                bj = int(base.y) - 1 + y

                if bi < 0 or bi >= ug.shape[0] or bj < 0 or bj >= ug.shape[1]:
                    continue

                dpos = wp.vec2(float(bi) + 0.5, float(bj) + 0.5) - pp
                coeff = nox[x] * noy[y]
                wp.atomic_add(ug, bi, bj, coeff * (up + aff @ dpos))
                wp.atomic_add(mg, bi, bj, coeff)

    @wp.kernel
    def _p2g_postprocess_kernel(
        ug: wp.array2d(dtype=wp.vec2),
        mg: wp.array2d(dtype=wp.float32),
        fg: wp.array2d(dtype=wp.uint8),
    ) -> None:
        i, j = wp.tid()

        if mg[i, j] > 0:
            ug[i, j] /= mg[i, j]

        if is_solid(i, j, fg):
            ug[i, j] = wp.vec2(0.0)

    @wp.kernel
    def _g_force_and_boundary_kernel(
        ug: wp.array2d(dtype=wp.vec2),
        fg: wp.array2d(dtype=wp.uint8),
    ) -> None:
        i, j = wp.tid()

        # Apply gravity
        if not is_solid(i, j, fg):
            ug[i, j] += wp.vec2(0.0, -5.0)

        # Can refer to
        # https://www.sciencedirect.com/science/article/pii/S0021999120300851
        # for more accurate boundary conditions.
        if is_water(i, j, fg):
            # Separate boundary condition:
            n = wp.vec2(0.0, 0.0)
            if is_solid(i+1, j, fg):
                n += wp.vec2(-1.0, 0.0)
            if is_solid(i-1, j, fg):
                n += wp.vec2(1.0, 0.0)
            if is_solid(i, j+1, fg):
                n += wp.vec2(0.0, -1.0)
            if is_solid(i, j-1, fg):
                n += wp.vec2(0.0, 1.0)
            ug[i, j] = ug[i, j] - n*wp.min(wp.dot(n, ug[i, j]), 0.0)

    @wp.kernel
    def _g2p_apic_kernel(
        u: wp.array1d(dtype=wp.vec2),
        p: wp.array1d(dtype=wp.vec2),
        c: wp.array1d(dtype=wp.mat22),
        F: wp.array1d(dtype=wp.mat22),
        J: wp.array1d(dtype=wp.float32),
        ug: wp.array2d(dtype=wp.vec2),
        boundary: int,
        dt: float,
    ) -> None:
        i = wp.tid()

        pp = p[i]
        base, nox, noy = get_bspline_coeff(pp)
        vel = wp.vec2(0.0, 0.0)
        aff = wp.mat22(0.0)

        for x in range(3):
            for y in range(3):
                bi = int(base.x) - 1 + x
                bj = int(base.y) - 1 + y

                if bi < 0 or bi >= ug.shape[0] or bj < 0 or bj >= ug.shape[1]:
                    continue

                ug_temp = ug[bi, bj]
                dpos = wp.vec2(float(bi) + 0.5, float(bj) + 0.5) - pp
                coeff = nox[x] * noy[y]

                vel += coeff * ug_temp
                aff += coeff * 4.0 * wp.outer(ug_temp, dpos)

        pos = p[i] + vel * dt
        pos.x = wp.clamp(pos.x, 0.0, float(ug.shape[0]-1))
        pos.y = wp.clamp(pos.y, 0.0, float(ug.shape[1]-1))
        p[i] = pos
        u[i] = vel
        c[i] = aff


if __name__ == '__main__':
    res = 512
    solver = MaterialPointSolver2D(res, res)

    gui = ti.GUI("MPM2D",
                 (solver.width_*2, solver.height_*2),
                 background_color=0x212121)
    frame_id = 0
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == 'f':
                solver.to_fluid()
            if e.key == 's':
                solver.to_soft()
        solver.anim_frame_particle(n_steps=32)
        gui.circles(solver.pn_ / res, radius=1.4,
                    palette=[0x0288D2, 0xE2943B, 0xffffff, 0xffffff], palette_indices=solver.fn_)
        # gui.show()
        gui.show('images/mpm2d_{:04d}.png'.format(frame_id))
        frame_id += 1
