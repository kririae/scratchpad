#!/usr/bin/env python3

# Just a show-off... for reference purposes only.
from typing import Any

import warp as wp
import warp.render
import numpy as np
import taichi as ti

SOLID = wp.constant(wp.uint8(0))
WATER = wp.constant(wp.uint8(1))
GAS = wp.constant(wp.uint8(2))


@wp.func
def is_inside(i: int, j: int, k: int, length: Any, width: Any, height: Any) -> bool:
    return float(i) >= 0 and float(i) < float(length) and  \
        float(j) >= 0 and float(j) < float(width) and \
        float(k) >= 0 and float(k) < float(height)


@wp.func
def get_at_position(
    i: int,
    j: int,
    k: int,
    f: wp.array3d(dtype=Any),
) -> Any:
    length, width, height = f.shape[0], f.shape[1], f.shape[2]
    i = wp.clamp(i, 0, length - 1)
    j = wp.clamp(j, 0, width - 1)
    k = wp.clamp(k, 0, height - 1)
    return f[i, j, k]


@wp.func
def is_solid(i: int, j: int, k: int, f: wp.array3d(dtype=wp.uint8)) -> bool:
    if not is_inside(i, j, k, f.shape[0], f.shape[1], f.shape[2]):
        return True
    return f[i, j, k] == SOLID


@wp.func
def is_water(i: int, j: int, k: int, f: wp.array3d(dtype=wp.uint8)) -> bool:
    if not is_inside(i, j, k, f.shape[0], f.shape[1], f.shape[2]):
        return False
    return f[i, j, k] == WATER


@wp.func
def is_gas(i: int, j: int, k: int, f: wp.array3d(dtype=wp.uint8)) -> bool:
    if not is_inside(i, j, k, f.shape[0], f.shape[1], f.shape[2]):
        return False
    return f[i, j, k] == GAS


@wp.func
def get_bspline_coeff(p: wp.vec3):
    base = wp.vec3(wp.floor(p.x), wp.floor(
        p.y), wp.floor(p.z)) + wp.vec3(0.5, 0.5, 0.5)
    ox = wp.vec3(p.x - (base.x - 1.0), p.x - (base.x), p.x - (base.x + 1.0))
    oy = wp.vec3(p.y - (base.y - 1.0), p.y - (base.y), p.y - (base.y + 1.0))
    oz = wp.vec3(p.z - (base.z - 1.0), p.z - (base.z), p.z - (base.z + 1.0))

    # Fuck WARP
    ox.x = wp.abs(ox.x)
    ox.y = wp.abs(ox.y)
    ox.z = wp.abs(ox.z)
    oy.x = wp.abs(oy.x)
    oy.y = wp.abs(oy.y)
    oy.z = wp.abs(oy.z)
    oz.x = wp.abs(oz.x)
    oz.y = wp.abs(oz.y)
    oz.z = wp.abs(oz.z)

    # Quadratic B-spline kernel
    nox = wp.vec3(0.5*(1.5-ox.x)*(1.5-ox.x), 0.75 -
                  ox.y*ox.y, 0.5*(1.5-ox.z)*(1.5-ox.z))
    noy = wp.vec3(0.5*(1.5-oy.x)*(1.5-oy.x), 0.75 -
                  oy.y*oy.y, 0.5*(1.5-oy.z)*(1.5-oy.z))
    noz = wp.vec3(0.5*(1.5-oz.x)*(1.5-oz.x), 0.75 -
                  oz.y*oz.y, 0.5*(1.5-oz.z)*(1.5-oz.z))

    return base, nox, noy, noz


class AffineParticleInCellSolver3D:
    def __init__(self, length: int, width: int, height: int, viscosity: float = 0.0) -> None:
        wp.init()

        self.length_ = length
        self.width_ = width
        self.height_ = height
        self.shape_ = (length, width, height)
        self.boundary_ = 5

        self.viscosity_ = viscosity
        self.dt_ = 1e-5

        self.n_diffuse_ = 32  # not used
        self.n_project_ = 64
        self.damping_ = 0.64

        self.p_ = wp.from_numpy(self._init(), dtype=wp.vec3, device="cuda")
        self.u_ = wp.zeros(self.p_.shape, dtype=wp.vec3)
        self.c_ = wp.zeros(self.p_.shape, dtype=wp.mat33)

        self.pn_ = self.p_.numpy()

        self.ug_ = wp.zeros(self.shape_, dtype=wp.vec3)
        self.ug0_ = wp.zeros(self.shape_, dtype=wp.vec3)
        self.ug1_ = wp.zeros(self.shape_, dtype=wp.vec3)

        self.mg_ = wp.zeros(self.shape_, dtype=wp.float32)
        self.fg_ = wp.zeros(self.shape_, dtype=wp.uint8)

        self.div_ = wp.zeros(self.shape_, dtype=wp.float32)
        self.j0_ = wp.zeros(self.shape_, dtype=wp.float32)
        self.j1_ = wp.zeros(self.shape_, dtype=wp.float32)

        self._init_grid()
        with wp.ScopedCapture() as prj_cap:
            self._g_project_iter()
        self.project_graph_ = prj_cap.graph

        with wp.ScopedCapture() as dif_cap:
            self._g_diffuse_iter()
        self.diffuse_graph_ = dif_cap.graph

    def step(self) -> None:
        with wp.ScopedTimer("step"):
            self._set_flag()
            self._p2g()
            self._g_project()
            self._g2p()
            wp.synchronize()

    def anim_frame(self, n_steps=32):
        for _ in range(n_steps):
            self.step()
        np.copyto(self.pn_, self.p_.numpy())

    def _init(self) -> np.ndarray:
        points_x = np.linspace(0.04, 0.34, int(120*1.2)) * self.length_
        points_y = np.linspace(0.05, 0.40, int(70*1.2)) * self.width_
        points_z = np.linspace(0.05, 0.95, int(180*1.2)) * self.height_
        grid_x, grid_y, grid_z = np.meshgrid(points_x, points_y, points_z)
        res = np.vstack(
            (grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T
        return res + np.random.rand(*res.shape) - 0.5

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
        wp.launch(self._p2g_apic_kernel, dim=self.p_.shape[0],
                  inputs=[self.u_,
                          self.p_,
                          self.c_,
                          self.ug_,
                          self.mg_])
        wp.launch(self._p2g_postprocess_kernel, dim=self.shape_,
                  inputs=[self.ug_, self.mg_, self.fg_])

    def _g_diffuse_iter(self) -> None:
        for _ in range(self.n_diffuse_):
            wp.launch(self._g_diffuse_kernel, dim=self.shape_,
                      inputs=[self.ug0_,
                              self.ug1_,
                              self.ug_,
                              self.fg_,
                              self.viscosity_])
            (self.ug0_, self.ug1_) = (self.ug1_, self.ug0_)

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
                  dim=self.shape_, inputs=[self.ug_, self.fg_, self.boundary_, self.dt_])
        if False:
            # IDK if this is correct, so disable it for now
            wp.capture_launch(self.diffuse_graph_)
        else:
            self.ug0_.assign(self.ug_)
        wp.launch(self._g_calc_divergence, dim=self.shape_, inputs=[
                  self.ug0_, self.div_, self.fg_])
        wp.launch(self._g_precondition, dim=self.shape_,
                  inputs=[self.j0_, self.fg_])
        wp.capture_launch(self.project_graph_)
        wp.launch(self._g_subtract_gradient_q_kernel, dim=self.shape_,
                  inputs=[self.ug0_,
                          self.j0_,
                          self.fg_,])
        self.ug_.assign(self.ug0_)

    def _g2p(self) -> None:
        wp.launch(self._g2p_apic_kernel, dim=self.p_.shape[0],
                  inputs=[self.u_, self.p_, self.c_,
                          self.ug_,
                          self.boundary_, self.dt_,])

    @wp.kernel
    def _init_grid_kernel(fg: wp.array3d(dtype=wp.uint8), boundary: int) -> None:
        i, j, k = wp.tid()
        length, width, height = fg.shape[0], fg.shape[1], fg.shape[2]

        if (i < boundary or i >= length - boundary) or  \
            (j < boundary or j >= width - boundary) or \
                (k < boundary or k >= height - boundary):
            fg[i, j, k] = SOLID

        xl = 160
        xr = 180

        width = 18
        if i >= xl and i < xr and j <= 20:
            if k >= 0 and k < width:
                fg[i, j, k] = SOLID
            if k >= 2*width and k < 3*width:
                fg[i, j, k] = SOLID
            if k >= 4*width and k < 5*width:
                fg[i, j, k] = SOLID
            if k >= 6*width and k < 7*width:
                fg[i, j, k] = SOLID
            if k >= 8*width and k < 9*width:
                fg[i, j, k] = SOLID

    @wp.kernel
    def _set_all_flag(
        fg: wp.array3d(dtype=wp.uint8)
    ) -> None:
        i, j, k = wp.tid()
        if fg[i, j, k] != SOLID:
            fg[i, j, k] = GAS

    @wp.kernel
    def _set_fluid_flag(
        p: wp.array1d(dtype=wp.vec3),
        fg: wp.array3d(dtype=wp.uint8),
    ) -> None:
        i = wp.tid()
        pp = p[i]
        base = wp.vec3(wp.floor(pp.x), wp.floor(pp.y), wp.floor(pp.z))

        if fg[int(base.x), int(base.y), int(base.z)] != SOLID:
            fg[int(base.x), int(base.y), int(base.z)] = WATER

    @wp.kernel
    def _p2g_apic_kernel(
        u: wp.array1d(dtype=wp.vec3),
        p: wp.array1d(dtype=wp.vec3),
        c: wp.array1d(dtype=wp.mat33),
        ug: wp.array3d(dtype=wp.vec3),
        mg: wp.array3d(dtype=wp.float32),
    ) -> None:
        i = wp.tid()

        up = u[i]
        pp = p[i]
        affine = c[i]
        base, nox, noy, noz = get_bspline_coeff(p[i])

        for x in range(3):
            for y in range(3):
                for z in range(3):
                    bi = int(base.x) - 1 + x
                    bj = int(base.y) - 1 + y
                    bk = int(base.z) - 1 + z

                    if bi < 0 or bi >= ug.shape[0] or bj < 0 or bj >= ug.shape[1] or bk < 0 or bk >= ug.shape[2]:
                        continue

                    dpos = wp.vec3(float(bi) + 0.5, float(bj) +
                                   0.5, float(bk) + 0.5) - pp
                    coeff = nox[x] * noy[y] * noz[z]
                    wp.atomic_add(ug, bi, bj, bk, coeff * (up + affine @ dpos))
                    wp.atomic_add(mg, bi, bj, bk, coeff)

    @wp.kernel
    def _p2g_postprocess_kernel(
        ug: wp.array3d(dtype=wp.vec3),
        mg: wp.array3d(dtype=wp.float32),
        fg: wp.array3d(dtype=wp.uint8),
    ) -> None:
        i, j, k = wp.tid()

        if mg[i, j, k] > 0:
            ug[i, j, k] /= mg[i, j, k]

        if is_solid(i, j, k, fg):
            ug[i, j, k] = wp.vec3(0.0)

    @wp.kernel
    def _g_force_and_boundary_kernel(
        ug: wp.array3d(dtype=wp.vec3),
        fg: wp.array3d(dtype=wp.uint8),
        boundary: int,
        dt: float,
    ) -> None:
        i, j, k = wp.tid()

        if not is_solid(i, j, k, fg):
            ug[i, j, k] += wp.vec3(0.0, -98000.0, 0.0) * dt

        if is_water(i, j, k, fg):
            n = wp.vec3(0.0)
            if is_solid(i+1, j, k, fg):
                n += wp.vec3(-1.0, 0.0, 0.0)
            if is_solid(i-1, j, k, fg):
                n += wp.vec3(1.0, 0.0, 0.0)
            if is_solid(i, j+1, k, fg):
                n += wp.vec3(0.0, -1.0, 0.0)
            if is_solid(i, j-1, k, fg):
                n += wp.vec3(0.0, 1.0, 0.0)
            if is_solid(i, j, k+1, fg):
                n += wp.vec3(0.0, 0.0, -1.0)
            if is_solid(i, j, k-1, fg):
                n += wp.vec3(0.0, 0.0, 1.0)
            ug[i, j, k] = ug[i, j, k] - n*wp.min(wp.dot(n, ug[i, j, k]), 0.0)

    @wp.kernel
    def _g_diffuse_kernel(
        u_src: wp.array3d(dtype=wp.vec3),
        u_dst: wp.array3d(dtype=wp.vec3),
        b: wp.array3d(dtype=wp.vec3),
        f: wp.array3d(dtype=wp.uint8),
        viscosity: float,
    ) -> None:
        i, j, k = wp.tid()

        if not is_water(i, j, k, f):
            u_dst[i, j, k] = b[i, j, k]
            return

        u_nearby = wp.vec3(0.0)
        valid_counter = int(0)

        if not is_solid(i-1, j, k, f):
            u_nearby += get_at_position(i-1, j, k, u_src)
            valid_counter += 1
        if not is_solid(i+1, j, k, f):
            u_nearby += get_at_position(i+1, j, k, u_src)
            valid_counter += 1
        if not is_solid(i, j-1, k, f):
            u_nearby += get_at_position(i, j-1, k, u_src)
            valid_counter += 1
        if not is_solid(i, j+1, k, f):
            u_nearby += get_at_position(i, j+1, k, u_src)
            valid_counter += 1
        if not is_solid(i, j, k-1, f):
            u_nearby += get_at_position(i, j, k-1, u_src)
            valid_counter += 1
        if not is_solid(i, j, k+1, f):
            u_nearby += get_at_position(i, j, k+1, u_src)
            valid_counter += 1

        u_dst[i, j, k] = (b[i, j, k] + u_nearby * viscosity) / \
            (1.0 + float(valid_counter)*viscosity)

    @wp.kernel
    def _g_calc_divergence(
        u: wp.array3d(dtype=wp.vec3),
        div: wp.array3d(dtype=wp.float32),
        f: wp.array3d(dtype=wp.uint8),
    ) -> None:
        i, j, k = wp.tid()
        if not is_water(i, j, k, f):
            return

        # Central-difference calculation
        d = get_at_position(i+1, j, k, u).x -  \
            get_at_position(i-1, j, k, u).x + \
            get_at_position(i, j+1, k, u).y -  \
            get_at_position(i, j-1, k, u).y + \
            get_at_position(i, j, k+1, u).z -  \
            get_at_position(i, j, k-1, u).z

        div[i, j, k] = d / 2.0

    @wp.kernel
    def _g_precondition(
        q: wp.array3d(dtype=wp.float32),
        f: wp.array3d(dtype=wp.uint8),
    ) -> None:
        i, j, k = wp.tid()

        if is_gas(i, j, k, f) or is_solid(i, j, k, f):
            q[i, j, k] = 0.0

    @wp.kernel
    def _g_project_kernel(
            q_src: wp.array3d(dtype=wp.float32),
            q_dst: wp.array3d(dtype=wp.float32),
            div: wp.array3d(dtype=wp.float32),
            f: wp.array3d(dtype=wp.uint8),
            damping: float,
    ) -> None:
        i, j, k = wp.tid()

        if not is_water(i, j, k, f):
            q_dst[i, j, k] = q_src[i, j, k]
            return

        q_nearby = float(0.0)
        valid_counter = int(0)

        if not is_solid(i+1, j, k, f):
            q_nearby += get_at_position(i+1, j, k, q_src)
            valid_counter += 1
        if not is_solid(i-1, j, k, f):
            q_nearby += get_at_position(i-1, j, k, q_src)
            valid_counter += 1
        if not is_solid(i, j+1, k, f):
            q_nearby += get_at_position(i, j+1, k, q_src)
            valid_counter += 1
        if not is_solid(i, j-1, k, f):
            q_nearby += get_at_position(i, j-1, k, q_src)
            valid_counter += 1
        if not is_solid(i, j, k+1, f):
            q_nearby += get_at_position(i, j, k+1, q_src)
            valid_counter += 1
        if not is_solid(i, j, k-1, f):
            q_nearby += get_at_position(i, j, k-1, q_src)
            valid_counter += 1

        q_dst[i, j, k] = (1.0 - damping)*q_src[i, j, k] + \
            (1.0/float(valid_counter))*damping*(q_nearby - div[i, j, k])

    @wp.kernel
    def _g_subtract_gradient_q_kernel(
        u: wp.array3d(dtype=wp.vec3),
        q: wp.array3d(dtype=wp.float32),
        f: wp.array3d(dtype=wp.uint8),
    ) -> None:
        i, j, k = wp.tid()

        if is_water(i+1, j, k, f) or is_water(i-1, j, k, f):
            if is_solid(i+1, j, k, f) or is_solid(i-1, j, k, f):
                u[i, j, k].x = 0.0
            else:
                u[i, j, k].x -= (q[i+1, j, k] - q[i-1, j, k]) / 2.0

        if is_water(i, j+1, k, f) or is_water(i, j-1, k, f):
            if is_solid(i, j+1, k, f) or is_solid(i, j-1, k, f):
                u[i, j, k].y = 0.0
            else:
                u[i, j, k].y -= (q[i, j+1, k] - q[i, j-1, k]) / 2.0

        if is_water(i, j, k+1, f) or is_water(i, j, k-1, f):
            if is_solid(i, j, k+1, f) or is_solid(i, j, k-1, f):
                u[i, j, k].z = 0.0
            else:
                u[i, j, k].z -= (q[i, j, k+1] - q[i, j, k-1]) / 2.0

    @wp.kernel
    def _g2p_apic_kernel(
        u: wp.array1d(dtype=wp.vec3),
        p: wp.array1d(dtype=wp.vec3),
        c: wp.array1d(dtype=wp.mat33),
        ug: wp.array3d(dtype=wp.vec3),
        boundary: int,
        dt: float,
    ) -> None:
        i = wp.tid()

        pp = p[i]
        base, nox, noy, noz = get_bspline_coeff(pp)
        vel = wp.vec3(0.0)
        aff = wp.mat33(0.0)

        for x in range(3):
            for y in range(3):
                for z in range(3):
                    bi = int(base.x) - 1 + x
                    bj = int(base.y) - 1 + y
                    bk = int(base.z) - 1 + z

                    if bi < 0 or bi >= ug.shape[0] or bj < 0 or bj >= ug.shape[1] or bk < 0 or bk >= ug.shape[2]:
                        continue

                    ug_temp = ug[bi, bj, bk]
                    dpos = wp.vec3(float(bi) + 0.5, float(bj) +
                                   0.5, float(bk) + 0.5) - pp
                    coeff = nox[x] * noy[y] * noz[z]

                    vel += coeff * ug_temp
                    aff += coeff * 4.0 * wp.outer(ug_temp, dpos)

        pos = p[i] + vel * dt
        for j in range(3):
            pos[j] = wp.clamp(pos[j], float(boundary),
                              float(ug.shape[j] - boundary))
        p[i] = pos
        u[i] = vel
        c[i] = aff


if __name__ == '__main__':
    res = 160
    solver = AffineParticleInCellSolver3D(250, 160, 160)

    if False:
        renderer = wp.render.OpenGLRenderer(
            camera_pos=(1.0, 0.8, 3.5),
            draw_sky=False,
            draw_axis=False,
            vsync=False)

    ti.init(arch=ti.cuda)
    particle_pos = ti.Vector.field(3, dtype=ti.f32, shape=solver.pn_.shape[0])

    window = ti.ui.Window('APIC 3D', res=(1280, 720), vsync=True, fps_limit=60)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.8, 0.8, 3.5)
    camera.lookat(0.8, 0.5, 0)
    camera.up(0, 1, 0)
    camera.fov(30)

    for _ in range(1024):
        solver.anim_frame(n_steps=32)
        particle_pos.from_numpy(solver.pn_ / res)
        scene.set_camera(camera)
        scene.ambient_light((0.6, 0.6, 0.6))
        scene.point_light(pos=(-1.5, 1.5, 1.5), color=(1, 1, 1))
        scene.particles(particle_pos, color=(
            2/255, 136/255, 210/255), radius=0.002)
        vertices = np.array(
            [
                [-1.000000, -1.000000, 1.000000],
                [-1.000000, 1.000000, 1.000000],
                [-1.000000, -1.000000, -1.000000],
                [-1.000000, 1.000000, -1.000000],
                [1.000000, -1.000000, 1.000000],
                [1.000000, 1.000000, 1.000000],
                [1.000000, -1.000000, -1.000000],
                [1.000000, 1.000000, -1.000000]
            ]
        )
        indices = np.array(
            [
                [2, 3, 1],
                [4, 7, 3],
                [8, 5, 7],
                [6, 1, 5],
                [7, 1, 3],
                [4, 6, 8],
                [2, 4, 3],
                [4, 8, 7],
                [8, 6, 5],
                [6, 2, 1],
                [7, 5, 1],
                [4, 2, 6],
            ]
        )
        canvas.scene(scene)
        canvas.set_background_color((0.1, 0.1, 0.1))
        window.save_image('images/apic3d/apic3d_{:04d}.png'.format(_))
        window.show()

        if False:
            time = renderer.clock_time
            renderer.begin_frame(time)
            arr = solver.pn_ / res
            renderer.render_points(
                name='particles', points=arr, radius=0.002)
            renderer.end_frame()

        # Save numpy array
        if False:
            with open(f'archive4/p_{_}.npz', 'wb') as f:
                print(f'saving frame {_}...')
                np.savez(f, arr)
