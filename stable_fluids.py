#!/usr/bin/env python3

from typing import Any

import numpy as np
import warp as wp

import matplotlib
import matplotlib.animation as anim
import matplotlib.pyplot as plt
from matplotlib import cm


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
        width: float,
        height: float
) -> Any:
    p = clamp_to_domain(p, width, height)
    return f[int(p.x), int(p.y)]


class StableFluidSolver2D:
    """
    The fluid solver that implements [Stam 1999, Stable Fluids] in 2D.
    """

    def __init__(self, width: int, height: int, viscosity: float) -> None:
        wp.init()

        self.shape_ = (width, height)
        self.u0_ = wp.zeros(self.shape_, dtype=wp.vec2)
        self.u1_ = wp.zeros(self.shape_, dtype=wp.vec2)
        self.dx_ = 1.0
        self.viscosity_ = viscosity

        self.n_diffuse_ = 11
        self.n_project_ = 128

        # Damped jacobi iteration
        self.damping_ = 0.64

        # Magnitude of velocity
        self.um_ = wp.zeros(self.shape_, dtype=wp.float32)
        self.umax_ = 8

        # Cache the intermediate fields
        self.div_ = wp.zeros(self.shape_, dtype=wp.float32)
        self.d0_ = wp.zeros(self.shape_, dtype=wp.vec2)
        self.d1_ = wp.zeros(self.shape_, dtype=wp.vec2)
        self.j0_ = wp.zeros(self.shape_, dtype=wp.float32)
        self.j1_ = wp.zeros(self.shape_, dtype=wp.float32)

        self._init()
        with wp.ScopedCapture() as diffuse_capture:
            self._diffuse_iter()
        self.diffuse_graph_ = diffuse_capture.graph

        with wp.ScopedCapture() as project_capture:
            self._project_iter()
        self.project_graph_ = project_capture.graph

    def step(self) -> None:
        with wp.ScopedTimer("step"):
            self._init()
            self._force()
            self._advect()
            self._diffuse()
            self._project()
            wp.synchronize()

        # Swap the velocity fields
        (self.u0_, self.u1_) = (self.u1_, self.u0_)

    def get_velocity(self) -> wp.array2d:
        return self.u0_

    def render_velocity(self) -> None:
        wp.launch(self._render_velocity, dim=self.shape_,
                  inputs=[self.u0_, self.um_])
        wp.synchronize()

    def anim_frame(self, frame_num=None, img=None):
        self.step()
        self.render_velocity()
        if img:
            img.set_array(self.um_.numpy())
        return (img,)

    def _init(self) -> None:
        wp.launch(self._init_jetflow, dim=self.shape_, inputs=[
                  self.u0_, self.shape_[0], self.shape_[1]])
        wp.synchronize()

    def _force(self) -> None:
        pass

    def _advect(self) -> None:
        wp.launch(self._advect_kernel, dim=self.shape_,
                  inputs=[self.u0_,
                          self.u1_,
                          self.shape_[0],
                          self.shape_[1]])

    def _diffuse_iter(self) -> None:
        for _ in range(self.n_diffuse_):
            wp.launch(self._jacobi_iter_diffuse, dim=self.shape_,
                      inputs=[self.d0_,
                              self.d1_,
                              self.u1_,  # w2
                              self.shape_[0],
                              self.shape_[1],
                              self.viscosity_])
            (self.d0_, self.d1_) = (self.d1_, self.d0_)

    def _project_iter(self) -> None:
        for _ in range(self.n_project_):
            wp.launch(self._jacobi_iter_project, dim=self.shape_,
                      inputs=[self.j0_,
                              self.j1_,
                              self.div_,
                              self.damping_,
                              self.shape_[0],
                              self.shape_[1]])
            (self.j0_, self.j1_) = (self.j1_, self.j0_)

    def _diffuse(self) -> None:
        wp.capture_launch(self.diffuse_graph_)

        # Copy the result back to the velocity field
        self.u1_.assign(self.d0_)

    def _project(self) -> None:
        wp.launch(self._calc_divergence, dim=self.shape_, inputs=[
                  self.u1_, self.div_])
        wp.capture_launch(self.project_graph_)

        # Subtract the gradient of `q`
        wp.launch(self._subtract_gradient_q, dim=self.shape_,
                  inputs=[self.u1_,
                          self.j0_,
                          self.shape_[0],
                          self.shape_[1]])

    @wp.kernel
    def _init_jetflow(
        u: wp.array2d(dtype=wp.vec2),
        width: float,
        height: float
    ) -> None:
        i, j = wp.tid()
        if i == 0:
            u[i, j] = wp.vec2(0.0, 0.0)
        if i == 0 and j >= 256-20 and j <= 256+20:
            u[i, j] = wp.vec2(8.0, 0.0)

    @wp.kernel
    def _advect_kernel(
            u0: wp.array2d(dtype=wp.vec2),
            u1: wp.array2d(dtype=wp.vec2),
            width: float,
            height: float
    ) -> None:
        i, j = wp.tid()

        # Perform advection by semi-lagrangian method to determine the original position
        p = wp.vec2(float(i), float(j))
        u = u0[i, j]
        p_src = p - u

        # Sample the velocity at the source position
        p_p0 = wp.vec2(wp.floor(p_src.x), wp.floor(p_src.y))
        p_p1 = p_p0 + wp.vec2(1.0, 0.0)
        p_p2 = p_p0 + wp.vec2(0.0, 1.0)
        p_p3 = p_p0 + wp.vec2(1.0, 1.0)

        w_p0 = (1.0 - (p_src.x - p_p0.x)) * (1.0 - (p_src.y - p_p0.y))
        w_p1 = (p_src.x - p_p0.x) * (1.0 - (p_src.y - p_p0.y))
        w_p2 = (1.0 - (p_src.x - p_p0.x)) * (p_src.y - p_p0.y)
        w_p3 = (p_src.x - p_p0.x) * (p_src.y - p_p0.y)

        u_interp = w_p0 * get_at_position(p_p0, u0, width, height) + \
            w_p1 * get_at_position(p_p1, u0, width, height) + \
            w_p2 * get_at_position(p_p2, u0, width, height) + \
            w_p3 * get_at_position(p_p3, u0, width, height)

        # Write the interpolated velocity to the current position
        u1[i, j] = u_interp

    @wp.kernel
    def _jacobi_iter_diffuse(
            u_src: wp.array2d(dtype=wp.vec2),
            u_dst: wp.array2d(dtype=wp.vec2),
            b: wp.array2d(dtype=wp.vec2),
            width: float,
            height: float,
            viscosity: float
    ) -> None:
        # Perform the Jacobi iteration to implicitly solve the diffusion equation
        i, j = wp.tid()

        # Implement: x_post = D^{-1}b - D^{-1}(L + U)x
        # where D = (1 + 4*viscosity), so the equation is
        #   x_dst = inv_coeff*b[i, j] + inv_coeff*viscosity*x[nearbys...]
        x_dst = (b[i, j] +
                 (get_at_position(wp.vec2(float(i-1), float(j)),   u_src, width, height) +
                  get_at_position(wp.vec2(float(i+1), float(j)),   u_src, width, height) +
                  get_at_position(wp.vec2(float(i),   float(j-1)), u_src, width, height) +
                  get_at_position(wp.vec2(float(i),   float(j+1)), u_src, width, height)) * viscosity) / (1.0 + 4.0*viscosity)
        u_dst[i, j] = x_dst

    @wp.kernel
    def _calc_divergence(
        u: wp.array2d(dtype=wp.vec2),
        div: wp.array2d(dtype=wp.float32),
    ) -> None:
        i, j = wp.tid()
        width, height = float(u.shape[0]), float(u.shape[1])

        div[i, j] = (get_at_position(wp.vec2(float(i+1), float(j)), u, width, height).x -
                     get_at_position(wp.vec2(float(i-1), float(j)), u, width, height).x +
                     get_at_position(wp.vec2(float(i), float(j+1)), u, width, height).y -
                     get_at_position(wp.vec2(float(i), float(j-1)), u, width, height).y) / 2.0

    @wp.kernel
    def _jacobi_iter_project(
            q_src: wp.array2d(dtype=wp.float32),
            q_dst: wp.array2d(dtype=wp.float32),
            div: wp.array2d(dtype=wp.float32),
            damping: float,
            width: float,
            height: float,
    ) -> None:
        # Perform the Jacobi iteration to solve the poisson equation
        i, j = wp.tid()

        # Implement: x_post = D^{-1}b - D^{-1}(L + U)x
        # where D = -4, so the equation is
        #   x_dst = 1/4*(q_src[nearbys...] - div)
        q_nearby = (get_at_position(wp.vec2(float(i-1), float(j)),   q_src, width, height) +
                    get_at_position(wp.vec2(float(i+1), float(j)),   q_src, width, height) +
                    get_at_position(wp.vec2(float(i),   float(j-1)), q_src, width, height) +
                    get_at_position(wp.vec2(float(i),   float(j+1)), q_src, width, height))
        # x_dst = (q_nearby - div) / 4.0
        # alternatively, we can use the damping factor
        #   x_dst = (1 - damping)*q_src[i, j]
        #         + (0.25*damping + q_src[nearbys...])
        #         + -0.25*damping*div
        x_dst = (1.0 - damping)*q_src[i, j] + \
            0.25*damping*(q_nearby - div[i, j])
        q_dst[i, j] = x_dst

    @wp.kernel
    def _subtract_gradient_q(
        w3: wp.array2d(dtype=wp.vec2),
        q: wp.array2d(dtype=wp.float32),
        width: float,
        height: float
    ) -> None:
        # w4 = w3 - grad(q)
        i, j = wp.tid()

        grad_q_x = (get_at_position(wp.vec2(float(i+1), float(j)), q, width, height) -
                    get_at_position(wp.vec2(float(i-1), float(j)), q, width, height)) / 2.0
        grad_q_y = (get_at_position(wp.vec2(float(i), float(j+1)), q, width, height) -
                    get_at_position(wp.vec2(float(i), float(j-1)), q, width, height)) / 2.0
        w3[i, j] = w3[i, j] - wp.vec2(grad_q_x, grad_q_y)

    @wp.kernel
    def _render_velocity(
        u: wp.array2d(dtype=wp.vec2),
        um: wp.array2d(dtype=wp.float32)
    ) -> None:
        i, j = wp.tid()
        um[i, j] = wp.length(u[i, j])


if __name__ == '__main__':
    solver = StableFluidSolver2D(512, 512, 0.00001)

    fig = plt.figure()
    img = plt.imshow(solver.um_.numpy(), origin='lower',
                     animated=True, interpolation="antialiased")
    img.set_norm(matplotlib.colors.Normalize(0, solver.umax_))
    seq = anim.FuncAnimation(
        fig,
        solver.anim_frame,
        fargs=(img,),
        frames=16384,
        blit=True,
        interval=32,
        repeat=False,
    )

    plt.show()
