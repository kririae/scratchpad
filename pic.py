#!/usr/bin/env python3
from typing import Any

import warp as wp
import numpy as np
import taichi as ti  # I need taichi's OpenGL render

import matplotlib
import matplotlib.animation as anim
import matplotlib.pyplot as plt
from matplotlib import cm

SOLID = wp.constant(wp.uint8(0))
WATER = wp.constant(wp.uint8(1))
GAS = wp.constant(wp.uint8(2))


@wp.func
def clamp_to_domain(p: wp.vec2, width: float, height: float) -> wp.vec2:
    return wp.vec2(
        wp.clamp(p.x, 0.0, width - 1.0),
        wp.clamp(p.y, 0.0, height - 1.0)
    )


@wp.func
def get_at_position(
        p: wp.vec2,
        f: wp.array2d(dtype=Any),
) -> Any:
    width, height = float(f.shape[0]), float(f.shape[1])
    p = clamp_to_domain(p, width, height)
    return f[int(p.x), int(p.y)]


@wp.func
def get_q_at_position(
    p: wp.vec2,
    q: wp.array2d(dtype=wp.float32),
) -> wp.float32:
    width, height = float(q.shape[0]), float(q.shape[1])
    p = clamp_to_domain(p, width, height)
    return q[int(p.x), int(p.y)]


@wp.func
def is_inside(i: int, j: int, width: Any, height: Any) -> bool:
    return float(i) >= 0 and float(i) < float(width) and float(j) >= 0 and float(j) < float(height)


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


class ParticleInCellSolver2D:
    """
    The fluid solver that replaces Stable Fluids' advection process with Particle-In-Cell (PIC) in 2D.
    """

    def __init__(self, width: int, height: int) -> None:
        wp.init()

        self.width_ = width
        self.height_ = height
        self.shape_ = (width, height)

        self.n_project_ = 8
        self.damping_ = 0.64

        # Particle positions and velocities
        self.p_ = wp.from_numpy(self._init(), dtype=wp.vec2, device="cuda")
        self.u_ = wp.zeros(self.p_.shape, dtype=wp.vec2)

        self.pn_ = self.p_.numpy()

        # Grid velocities
        self.ug_ = wp.zeros((width, height), dtype=wp.vec2)
        self.mg_ = wp.zeros((width, height), dtype=wp.float32)
        self.um_ = wp.zeros((width, height), dtype=wp.float32)
        self.fg_ = wp.zeros((width, height), dtype=wp.uint8)  # flag field

        # Projected grid velocities
        self.div_ = wp.zeros((width, height), dtype=wp.float32)
        self.j0_ = wp.zeros((width, height), dtype=wp.float32)
        self.j1_ = wp.zeros((width, height), dtype=wp.float32)

        self._init_grid()

    def step(self) -> None:
        # The solver is composed of
        # ----------------------------
        # (1) Particle to Grid Transfer(P2G)
        # (2) Projection
        # (3) Grid to Particle Transfer(G2P)
        # (4) Particle Advection(by semi-lagrangian)
        with wp.ScopedTimer("step"):
            self._set_flag()
            self._p2g()
            self._g_project()
            self._g2p()
            wp.synchronize()

    def anim_frame_particle(self, n_steps=32):
        for _ in range(n_steps):
            self.step()
        np.copyto(self.pn_, self.p_.numpy())

    def _init(self) -> np.ndarray:
        points_x = np.linspace(0.25, 0.75, 500) * self.width_
        points_y = np.linspace(0.10, 0.60, 500) * self.height_
        density = (500 / (np.max(points_y) - np.min(points_y)))**2
        print(f'average density: {density}')
        grid_x, grid_y = np.meshgrid(points_x, points_y)
        return np.vstack((grid_x.flatten(), grid_y.flatten())).T

    def _init_grid(self) -> None:
        self.fg_.fill_(GAS)
        wp.launch(self._init_grid_kernel, dim=self.shape_, inputs=[self.fg_])

    def _set_flag(self) -> None:
        wp.launch(self._set_all_flag, dim=self.shape_, inputs=[self.fg_])
        wp.launch(self._set_fluid_flag,
                  dim=self.p_.shape[0], inputs=[self.p_, self.fg_])

    def _p2g(self) -> None:
        self.ug_.fill_(0.0)
        self.mg_.fill_(0.0)
        wp.launch(self._p2g_kernel, dim=self.p_.shape[0],
                  inputs=[self.u_,
                          self.p_,
                          self.ug_,
                          self.mg_])
        wp.launch(self._p2g_postprocess_kernel, dim=self.shape_,
                  inputs=[self.ug_, self.mg_, self.fg_])

    def _g_project(self) -> None:
        self.div_.fill_(0.0)
        wp.launch(self._g_calc_divergence, dim=self.shape_, inputs=[
                  self.ug_, self.div_, self.fg_])
        wp.launch(self._g_precondition, dim=self.shape_,
                  inputs=[self.j0_, self.fg_])
        for _ in range(self.n_project_):
            wp.launch(self._g_project_kernel, dim=self.shape_,
                      inputs=[self.j0_,
                              self.j1_,
                              self.div_,
                              self.fg_,
                              self.damping_,])
            (self.j0_, self.j1_) = (self.j1_, self.j0_)
        wp.launch(self._g_subtract_gradient_q_kernel, dim=self.shape_,
                  inputs=[self.ug_,
                          self.j0_,
                          self.fg_,])

    def _g2p(self) -> None:
        wp.launch(self._g2p_kernel, dim=self.p_.shape[0],
                  inputs=[self.u_,
                          self.p_,
                          self.ug_,])

    @wp.kernel
    def _init_grid_kernel(fg: wp.array2d(dtype=wp.uint8)) -> None:
        i, j = wp.tid()
        width, height = fg.shape[0], fg.shape[1]
        if (i <= 5 or i >= width - 6) or (j <= 5 or j >= height - 6):
            fg[i, j] = SOLID

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
        pp = wp.vec2(p[i].x, p[i].y)
        base = wp.vec2(wp.floor(pp.x), wp.floor(pp.y))

        if fg[int(base.x), int(base.y)] != SOLID:
            fg[int(base.x), int(base.y)] = WATER

    @wp.kernel
    def _p2g_kernel(
        u: wp.array1d(dtype=wp.vec2),
        p: wp.array1d(dtype=wp.vec2),
        ug: wp.array2d(dtype=wp.vec2),
        mg: wp.array2d(dtype=wp.float32),
    ) -> None:
        i = wp.tid()

        up = wp.vec2(u[i].x, u[i].y)
        pp = wp.vec2(p[i].x, p[i].y)

        # Calculate the corresponding grid position of this particle,
        base = wp.vec2(wp.floor(pp.x), wp.floor(pp.y))
        ox = wp.vec3(pp.x - (base.x - 0.5), pp.x -
                     (base.x + 0.5), (base.x + 1.5) - pp.x)
        oy = wp.vec3(pp.y - (base.y - 0.5), pp.y -
                     (base.y + 0.5), (base.y + 1.5) - pp.y)

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

        # Accumulate the velocity from particles to grid
        for x in range(3):
            for y in range(3):
                bi = int(base.x) - 1 + x
                bj = int(base.y) - 1 + y

                if bi < 0 or bi >= ug.shape[0] or bj < 0 or bj >= ug.shape[1]:
                    continue

                coeff = nox[x] * noy[y]
                wp.atomic_add(ug, bi, bj, coeff * up)
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

            # Apply gravity
            ug[i, j] += wp.vec2(0.0, -30.0)

        if is_solid(i, j, fg):
            ug[i, j] = wp.vec2(0.0, 0.0)

    @wp.kernel
    def _g_calc_divergence(
        u: wp.array2d(dtype=wp.vec2),
        div: wp.array2d(dtype=wp.float32),
        f: wp.array2d(dtype=wp.uint8),
    ) -> None:
        i, j = wp.tid()
        if not is_water(i, j, f):
            return

        ####################################################################
        # Equations from [Bridson et al. 2015, Fluid Simulation for Computer
        # Graphics, Figure 5.4].
        #
        # Divergence of the velocity field with boundary correction.
        d = float(0.0)
        if not is_solid(i-1, j, f):
            d -= get_at_position(wp.vec2(float(i),  float(j)), u).x
        if not is_solid(i+1, j, f):
            d += get_at_position(wp.vec2(float(i+1), float(j)), u).x
        if not is_solid(i, j-1, f):
            d -= get_at_position(wp.vec2(float(i),  float(j)), u).y
        if not is_solid(i, j+1, f):
            d += get_at_position(wp.vec2(float(i), float(j+1)), u).y

        div[i, j] = d

    @wp.kernel
    def _g_precondition(
        q: wp.array2d(dtype=wp.float32),
        f: wp.array2d(dtype=wp.uint8),
    ) -> None:
        i, j = wp.tid()

        if is_gas(i, j, f):
            q[i, j] = 0.0

    @wp.kernel
    def _g_project_kernel(
            q_src: wp.array2d(dtype=wp.float32),
            q_dst: wp.array2d(dtype=wp.float32),
            div: wp.array2d(dtype=wp.float32),
            f: wp.array2d(dtype=wp.uint8),
            damping: float,
    ) -> None:
        i, j = wp.tid()

        if not is_water(i, j, f):
            return

        ####################################################################
        # Equations from [Bridson et al. 2015, Fluid Simulation for Computer
        # Graphics, Figure 5.5].
        #
        # Jacobi iteration for Poisson pressure.
        q_nearby = float(0.0)
        valid_counter = int(0)

        if not is_solid(i+1, j, f):
            q_nearby += get_q_at_position(wp.vec2(float(i+1), float(j)), q_src)
            valid_counter += 1
        if not is_solid(i-1, j, f):
            q_nearby += get_q_at_position(wp.vec2(float(i-1), float(j)), q_src)
            valid_counter += 1
        if not is_solid(i, j+1, f):
            q_nearby += get_q_at_position(wp.vec2(float(i), float(j+1)), q_src)
            valid_counter += 1
        if not is_solid(i, j-1, f):
            q_nearby += get_q_at_position(wp.vec2(float(i), float(j-1)), q_src)
            valid_counter += 1

        if valid_counter > 0:
            q_dst[i, j] = (1.0 - damping)*q_src[i, j] + \
                (1.0/float(valid_counter))*damping*(q_nearby - div[i, j])
        else:
            q_dst[i, j] = 0.0

    @wp.kernel
    def _g_subtract_gradient_q_kernel(
        u: wp.array2d(dtype=wp.vec2),
        q: wp.array2d(dtype=wp.float32),
        f: wp.array2d(dtype=wp.uint8),
    ) -> None:
        i, j = wp.tid()

        ####################################################################
        # Equations from [Bridson et al. 2015, Fluid Simulation for Computer
        # Graphics, Figure 5.2].
        #
        # Poisson pressure gradient update.
        if is_water(i-1, j, f) or is_water(i, j, f):
            if is_solid(i-1, j, f) or is_solid(i, j, f):
                u[i, j].x = 0.0
            else:
                u[i, j].x -= (q[i, j] - q[i-1, j])

        if is_water(i, j-1, f) or is_water(i, j, f):
            if is_solid(i, j-1, f) or is_solid(i, j, f):
                u[i, j].y = 0.0
            else:
                u[i, j].y -= (q[i, j] - q[i, j-1])

    @wp.kernel
    def _g2p_kernel(
        u: wp.array1d(dtype=wp.vec2),
        p: wp.array1d(dtype=wp.vec2),
        ug: wp.array2d(dtype=wp.vec2),
    ) -> None:
        i = wp.tid()
        pp = wp.vec2(p[i].x, p[i].y)

        # Calculate the corresponding grid position of this particle,
        base = wp.vec2(wp.floor(pp.x), wp.floor(pp.y))
        ox = wp.vec3(pp.x - (base.x - 0.5), pp.x -
                     (base.x + 0.5), (base.x + 1.5) - pp.x)
        oy = wp.vec3(pp.y - (base.y - 0.5), pp.y -
                     (base.y + 0.5), (base.y + 1.5) - pp.y)

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

        # Create the accumulator for the velocity
        vel = wp.vec2(0.0, 0.0)

        # Accumulate the velocity from particles to grid
        for x in range(3):
            for y in range(3):
                bi = int(base.x) - 1 + x
                bj = int(base.y) - 1 + y

                if bi < 0 or bi >= ug.shape[0] or bj < 0 or bj >= ug.shape[1]:
                    continue

                coeff = nox[x] * noy[y]
                vel += coeff * ug[bi, bj]

        # Update the velocity of this particle
        u[i] = vel

        p[i] += u[i] * 2e-5
        p[i].x = wp.clamp(p[i].x, 0.0, float(ug.shape[0]) - 1.0)
        p[i].y = wp.clamp(p[i].y, 0.0, float(ug.shape[1]) - 1.0)

    @wp.kernel
    def _render_velocity(
        u: wp.array2d(dtype=wp.vec2),
        um: wp.array2d(dtype=wp.float32)
    ) -> None:
        i, j = wp.tid()
        um[i, j] = wp.length(u[i, j])


if __name__ == '__main__':
    res = 512
    solver = ParticleInCellSolver2D(res, res)

    gui = ti.GUI("Particle-In-Cell", (solver.width_,
                 solver.height_), background_color=0x212121)
    while gui.running:
        solver.anim_frame_particle(n_steps=32)
        gui.circles(solver.pn_ / res, radius=0.6, color=0x0288D1)
        gui.show()
