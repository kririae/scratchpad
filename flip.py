#!/usr/bin/env python3
from typing import Any, Tuple

import warp as wp
import numpy as np
import taichi as ti  # I need taichi's OpenGL render

SOLID = wp.constant(wp.uint8(0))
WATER = wp.constant(wp.uint8(1))
GAS = wp.constant(wp.uint8(2))


@wp.func
def get_at_position(
    i: int,
    j: int,
    f: wp.array2d(dtype=Any),
) -> Any:
    width, height = f.shape[0], f.shape[1]
    i = wp.clamp(i, 0, width - 1)
    j = wp.clamp(j, 0, height - 1)
    return f[i, j]


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


@wp.func
def get_bspline_coeff(p: wp.vec2):
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


class HybridSolver2D:
    """
    The fluid solver that replaces Stable Fluids' advection process with PIC/APIC in 2D.
    """

    def __init__(self, width: int, height: int) -> None:
        wp.init()

        self.width_ = width
        self.height_ = height
        self.shape_ = (width, height)
        self.boundary_ = 2

        # FLIP/PIC blend rate
        self.blend_ = 0.95

        # The number of iterations for the projection step
        self.n_project_ = 64

        # The damping factor for the damped Jacobi iteration
        self.damping_ = 0.64

        # Particle positions and velocities
        self.p_ = wp.from_numpy(self._init(), dtype=wp.vec2, device="cuda")
        self.u_ = wp.zeros(self.p_.shape, dtype=wp.vec2)

        # Particles positions as a numpy array for rendering
        self.pn_ = self.p_.numpy()

        # Grid velocities
        self.ug_ = wp.zeros((width, height), dtype=wp.vec2)
        self.ug_prev_ = wp.zeros((width, height), dtype=wp.vec2)

        self.mg_ = wp.zeros((width, height), dtype=wp.float32)
        self.fg_ = wp.zeros((width, height), dtype=wp.uint8)  # flag field

        # Projected grid velocities
        self.div_ = wp.zeros((width, height), dtype=wp.float32)
        self.j0_ = wp.zeros((width, height), dtype=wp.float32)
        self.j1_ = wp.zeros((width, height), dtype=wp.float32)

        self._init_grid()
        with wp.ScopedCapture() as cap:
            self._g_project_iter()
        self.project_graph_ = cap.graph

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
            self.ug_prev_.assign(self.ug_)
            self._g_project()
            self._g2p()
            wp.synchronize()

    def anim_frame_particle(self, n_steps=32):
        for _ in range(n_steps):
            self.step()
        np.copyto(self.pn_, self.p_.numpy())

    def _init(self) -> np.ndarray:
        points_x = np.linspace(0.25, 0.75, 500) * self.width_
        points_y = np.linspace(0.30, 0.80, 500) * self.height_
        density = (500 / (np.max(points_y) - np.min(points_y)))**2
        print(f'average density: {density}')
        grid_x, grid_y = np.meshgrid(points_x, points_y)
        return np.vstack((grid_x.flatten(), grid_y.flatten())).T

    def _init_grid(self) -> None:
        self.fg_.fill_(GAS)
        wp.launch(self._init_grid_kernel, dim=self.shape_,
                  inputs=[self.fg_, self.boundary_])

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
                  inputs=[self.ug_, self.mg_])

    def _g_project_iter(self) -> None:
        for _ in range(self.n_project_):
            wp.launch(self._g_project_kernel, dim=self.shape_,
                      inputs=[self.j0_,
                              self.j1_,
                              self.div_,
                              self.fg_,
                              self.damping_,])
            (self.j0_, self.j1_) = (self.j1_, self.j0_)

    def _g_project(self) -> None:
        self.div_.fill_(0.0)
        wp.launch(self._g_force_and_boundary_kernel,
                  dim=self.shape_, inputs=[self.ug_, self.fg_])
        wp.launch(self._g_calc_divergence, dim=self.shape_, inputs=[
                  self.ug_, self.div_, self.fg_])
        wp.launch(self._g_precondition, dim=self.shape_,
                  inputs=[self.j0_, self.fg_])
        wp.capture_launch(self.project_graph_)
        wp.launch(self._g_subtract_gradient_q_kernel, dim=self.shape_,
                  inputs=[self.ug_,
                          self.j0_,
                          self.fg_,])

    def _g2p(self) -> None:
        wp.launch(self._g2p_kernel, dim=self.p_.shape[0],
                  inputs=[self.u_,
                          self.p_,
                          self.ug_,
                          self.ug_prev_,
                          self.blend_,
                          self.boundary_,])

    @wp.kernel
    def _init_grid_kernel(fg: wp.array2d(dtype=wp.uint8), boundary: int) -> None:
        i, j = wp.tid()
        width, height = fg.shape[0], fg.shape[1]
        if (i <= boundary or i >= width - boundary - 1) or (j <= boundary or j >= height - boundary - 1):
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
        base, nox, noy = get_bspline_coeff(p[i])

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
    ) -> None:
        i, j = wp.tid()

        if mg[i, j] > 0:
            ug[i, j] /= mg[i, j]

    @wp.kernel
    def _g_force_and_boundary_kernel(
        ug: wp.array2d(dtype=wp.vec2),
        fg: wp.array2d(dtype=wp.uint8),
    ) -> None:
        i, j = wp.tid()

        # Apply gravity
        ug[i, j] += wp.vec2(0.0, -9.8)

        # Apply slip boundary conditions
        if is_solid(i, j, fg) or is_solid(i-1, j, fg) or is_solid(i+1, j, fg):
            ug[i, j].x = 0.0
        if is_solid(i, j, fg) or is_solid(i, j-1, fg) or is_solid(i, j+1, fg):
            ug[i, j].y = 0.0

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
        # Since boundary cells are set to zero, we can skip them.
        d = get_at_position(i+1, j, u).x -  \
            get_at_position(i-1, j, u).x + \
            get_at_position(i, j+1, u).y -  \
            get_at_position(i, j-1, u).y

        div[i, j] = d / 2.0

    @wp.kernel
    def _g_precondition(
        q: wp.array2d(dtype=wp.float32),
        f: wp.array2d(dtype=wp.uint8),
    ) -> None:
        i, j = wp.tid()

        if is_gas(i, j, f) or is_solid(i, j, f):
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
            q_dst[i, j] = q_src[i, j]
            return

        ####################################################################
        # Equations from [Bridson et al. 2015, Fluid Simulation for Computer
        # Graphics, Figure 5.5].
        #
        # Jacobi iteration for Poisson pressure.
        q_nearby = float(0.0)
        valid_counter = int(0)

        if not is_solid(i+1, j, f):
            q_nearby += get_at_position(i+1, j, q_src)
            valid_counter += 1
        if not is_solid(i-1, j, f):
            q_nearby += get_at_position(i-1, j, q_src)
            valid_counter += 1
        if not is_solid(i, j+1, f):
            q_nearby += get_at_position(i, j+1, q_src)
            valid_counter += 1
        if not is_solid(i, j-1, f):
            q_nearby += get_at_position(i, j-1, q_src)
            valid_counter += 1

        q_dst[i, j] = (1.0 - damping)*q_src[i, j] + \
            (1.0/float(valid_counter))*damping*(q_nearby - div[i, j])

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
        if is_water(i+1, j, f) or is_water(i-1, j, f):
            if is_solid(i+1, j, f) or is_solid(i-1, j, f):
                u[i, j].x = 0.0
            else:
                u[i, j].x -= (q[i+1, j] - q[i-1, j]) / 2.0

        if is_water(i, j+1, f) or is_water(i, j-1, f):
            if is_solid(i, j+1, f) or is_solid(i, j-1, f):
                u[i, j].y = 0.0
            else:
                u[i, j].y -= (q[i, j+1] - q[i, j-1]) / 2.0

    @wp.kernel
    def _g2p_kernel(
        u: wp.array1d(dtype=wp.vec2),
        p: wp.array1d(dtype=wp.vec2),
        ug: wp.array2d(dtype=wp.vec2),
        ug_prev: wp.array2d(dtype=wp.vec2),
        blend: float,
        boundary: int,
    ) -> None:
        i = wp.tid()

        # Calculate the corresponding grid position of this particle,
        base, nox, noy = get_bspline_coeff(p[i])

        # Create the accumulator for the velocity
        vel_pic = wp.vec2(0.0, 0.0)
        vel_flip = u[i]

        # Accumulate the velocity from particles to grid
        for x in range(3):
            for y in range(3):
                bi = int(base.x) - 1 + x
                bj = int(base.y) - 1 + y

                if bi < 0 or bi >= ug.shape[0] or bj < 0 or bj >= ug.shape[1]:
                    continue

                coeff = nox[x] * noy[y]
                vel_pic += coeff * ug[bi, bj]
                vel_flip += coeff * (ug[bi, bj] - ug_prev[bi, bj])

        # Particle boundary conditions
        vel = (1.0 - blend) * vel_pic + blend * vel_flip

        # Dimensionless time scaling, T* = L* / 1e5 (this, and the gravity
        # constant, are the only two parameters in the simulation).
        # Suppose we are simulating a 1m x 1m box, while L* = 500, then T* = 5e-3.
        # 64 frames gives us 0.32s of physical time.
        pos = p[i] + vel * 1e-5
        for j in range(2):
            if pos[j] < float(boundary):
                pos[j] = float(boundary) + 1e-3
                vel[j] = 0.0
            if pos[j] > float(ug.shape[j] - boundary):
                pos[j] = float(ug.shape[j] - boundary) - 1e-3
                vel[j] = 0.0

        # Update the position/velocity of this particle
        p[i] = pos
        u[i] = vel


if __name__ == '__main__':
    res = 512
    solver = HybridSolver2D(res, res)

    gui = ti.GUI("PIC/FLIP", (solver.width_*2,
                 solver.height_*2), background_color=0x212121)
    while gui.running:
        solver.anim_frame_particle(n_steps=32)
        gui.circles(solver.pn_ / res, radius=1.6, color=0x0288D1)
        gui.show()
