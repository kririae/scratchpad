#!/usr/bin/env python3
from typing import Any, Tuple

import warp as wp
import numpy as np
import taichi as ti  # I need taichi's OpenGL render

WATER = wp.constant(wp.uint8(0))
SOFT = wp.constant(wp.uint8(1))
SNOW = wp.constant(wp.uint8(2))
SOLID = wp.constant(wp.uint8(3))
GAS = wp.constant(wp.uint8(4))


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

    ox.x = wp.abs(ox.x)
    ox.y = wp.abs(ox.y)
    ox.z = wp.abs(ox.z)
    oy.x = wp.abs(oy.x)
    oy.y = wp.abs(oy.y)
    oy.z = wp.abs(oy.z)
    oz.x = wp.abs(oz.x)
    oz.y = wp.abs(oz.y)
    oz.z = wp.abs(oz.z)

    nox = wp.vec3(0.5*(1.5-ox.x)*(1.5-ox.x), 0.75 -
                  ox.y*ox.y, 0.5*(1.5-ox.z)*(1.5-ox.z))
    noy = wp.vec3(0.5*(1.5-oy.x)*(1.5-oy.x), 0.75 -
                  oy.y*oy.y, 0.5*(1.5-oy.z)*(1.5-oy.z))
    noz = wp.vec3(0.5*(1.5-oz.x)*(1.5-oz.x), 0.75 -
                  oz.y*oz.y, 0.5*(1.5-oz.z)*(1.5-oz.z))

    return base, nox, noy, noz


@wp.func
def svd3(A: wp.mat33):
    U3 = wp.mat33(0.0)
    S3 = wp.vec3(0.0)
    V3 = wp.mat33(0.0)
    wp.svd3(A, U3, S3, V3)

    S = wp.mat33(S3[0], 0.0, 0.0, 0.0, S3[1], 0.0, 0.0, 0.0, S3[2])
    return U3, S, V3


class MaterialPointSolver3D:
    def __init__(self, length: int, width: int, height: int) -> None:
        wp.init()

        self.length_ = length
        self.width_ = width
        self.height_ = height
        self.shape_ = (length, width, height)
        self.boundary_ = 5

        self.dt_ = 1e-5

        # Particle positions and velocities
        # Incorrect AoS layout, fuck WARP again
        p, f = self._init_case()
        self.p_ = wp.from_numpy(p, dtype=wp.vec3, device="cuda")
        self.f_ = wp.from_numpy(f, dtype=wp.uint8, device="cuda")

        print(self.p_.shape, self.f_.shape)
        self.u_ = wp.zeros(self.p_.shape, dtype=wp.vec3)
        self.c_ = wp.zeros(self.p_.shape, dtype=wp.mat33)
        self.F_ = wp.zeros(self.p_.shape, dtype=wp.mat33)
        self.J_ = wp.ones(self.p_.shape, dtype=wp.float32)

        # plastic deformation
        self.Jp_ = wp.ones(self.p_.shape, dtype=wp.float32)

        self.pn_ = self.p_.numpy()
        self.fn_ = self.f_.numpy()

        self.ug_ = wp.zeros(self.shape_, dtype=wp.vec3)
        self.mg_ = wp.zeros(self.shape_, dtype=wp.float32)
        self.fg_ = wp.zeros(self.shape_, dtype=wp.uint8)

        self._init()
        wp.synchronize()

    def step(self) -> None:
        with wp.ScopedTimer("step"):
            self._set_flag()
            self._p2g()
            self._grid_op()
            self._g2p()
            wp.synchronize()

    def anim_frame(self, n_steps=32):
        for _ in range(n_steps):
            self.step()
        np.copyto(self.pn_, self.p_.numpy())
        np.copyto(self.fn_, self.f_.numpy())
        wp.synchronize()

    def _init_case(self):
        points_x = np.linspace(0.04, 0.34, int(120*1.2)) * self.length_
        points_y = np.linspace(0.05, 0.40, int(70*1.2)) * self.width_
        points_z = np.linspace(0.05, 0.95, int(180*1.2)) * self.height_
        grid_x, grid_y, grid_z = np.meshgrid(points_x, points_y, points_z)
        res = np.vstack(
            (grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T
        f = np.zeros_like(res[:, 0])
        f.fill(WATER)
        return res + np.random.rand(*res.shape) - 0.5, f

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
                  inputs=[self.u_, self.p_, self.c_, self.F_, self.J_, self.Jp_, self.f_,
                          self.ug_, self.mg_, self.dt_])
        wp.launch(self._p2g_postprocess_kernel, dim=self.shape_,
                  inputs=[self.ug_, self.mg_, self.fg_])

    def _grid_op(self) -> None:
        wp.launch(self._g_force_and_boundary_kernel, dim=self.shape_, inputs=[
            self.ug_, self.fg_,])

    def _g2p(self) -> None:
        wp.launch(self._g2p_apic_kernel, dim=self.p_.shape[0],
                  inputs=[self.u_, self.p_, self.c_, self.F_, self.J_,
                          self.ug_,
                          self.boundary_, self.dt_])

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
    def _init_particle_kernel(F: wp.array1d(dtype=wp.mat33)) -> None:
        i = wp.tid()
        F[i] = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

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
        F: wp.array1d(dtype=wp.mat33),
        J: wp.array1d(dtype=wp.float32),
        Jp: wp.array1d(dtype=wp.float32),
        f: wp.array1d(dtype=wp.uint8),
        ug: wp.array3d(dtype=wp.vec3),
        mg: wp.array3d(dtype=wp.float32),
        dt: float,
    ) -> None:
        i = wp.tid()

        up = u[i]
        pp = p[i]
        aff = c[i]
        base, nox, noy, noz = get_bspline_coeff(p[i])

        # Perform MLS-MPM particle advection
        F[i] = (wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 1.0) + dt*aff)@F[i]
        J[i] = (1.0 + dt*wp.trace(aff)) * J[i]

        E = 4e9
        nu = 0.36
        LameLa = E*nu / ((1.0+nu)*(1.0-2.0*nu))
        LameMu = E / (2.0*(1.0+nu))

        stress = wp.mat33(0.0)
        if f[i] == WATER:
            w = (dt * E) * (1.0 - J[i]) * J[i]
            stress += wp.mat33(w, 0.0, 0.0, 0.0, w, 0.0, 0.0, 0.0, w)
        elif f[i] == SNOW or f[i] == SOFT:
            U, S, V = svd3(F[i])

            if f[i] == SNOW:
                h = wp.clamp(wp.exp(8.0 * (1.0 - Jp[i])), 0.01, 50.0)
                LameLa *= (h * 0.1)
                LameMu *= (h * 0.1)
                for j in range(3):
                    new_S = wp.clamp(S[j, j], 1.0 - 2e-2, 1.0 + 8e-3)
                    Jp[i] *= S[j, j] / new_S
                    S[j, j] = new_S
                F[i] = U @ S @ wp.transpose(V)

            R = U @ wp.transpose(V)

            # We calculate the new jacobian on-the-fly for better numerical stability
            new_J = wp.determinant(S)
            P = 2.0*LameMu*(F[i] - R)@wp.transpose(F[i]) + LameLa*(new_J - 1.0) * \
                new_J*wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            stress += -4.0*dt*P
        elif False:
            invTF = wp.inverse(wp.transpose(F[i]))
            stress += -dt * \
                ((LameMu * (F[i] - invTF)) + LameLa * wp.log(wp.clamp(J[i], 1e-3, 1e3))
                 * invTF @ wp.transpose(F[i]))

        aff = stress + aff
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
                    wp.atomic_add(ug, bi, bj, bk, coeff * (up + aff @ dpos))
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
    ) -> None:
        i, j, k = wp.tid()

        if not is_solid(i, j, k, fg):
            ug[i, j, k] += wp.vec3(0.0, -0.9800, 0.0)

        if True:
            # Bounce-back
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
                ug[i, j, k] = ug[i, j, k] - n * \
                    wp.min(wp.dot(n, ug[i, j, k]), 0.0)
        else:
            if is_water(i, j, k, fg):
                if is_solid(i+1, j, k, fg) or is_solid(i-1, j, k, fg):
                    ug[i, j, k].x = 0.0
                if is_solid(i, j+1, k, fg) or is_solid(i, j-1, k, fg):
                    ug[i, k, k].y = 0.0
                if is_solid(i, j, k+1, fg) or is_solid(i, j, k-1, fg):
                    ug[i, j, k].z = 0.0

    @wp.kernel
    def _g2p_apic_kernel(
        u: wp.array1d(dtype=wp.vec3),
        p: wp.array1d(dtype=wp.vec3),
        c: wp.array1d(dtype=wp.mat33),
        F: wp.array1d(dtype=wp.mat33),
        J: wp.array1d(dtype=wp.float32),
        ug: wp.array3d(dtype=wp.vec3),
        boundary: int,
        dt: float,
    ) -> None:
        i = wp.tid()

        pp = p[i]
        base, nox, noy, noz = get_bspline_coeff(pp)
        vel = wp.vec3(0.0, 0.0, 0.0)
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
            pos[j] = wp.clamp(pos[j], 0.0, float(ug.shape[j]-1))
        p[i] = pos
        u[i] = vel
        c[i] = aff


if __name__ == '__main__':
    res = 160
    solver = MaterialPointSolver3D(250, 160, 160)

    ti.init(arch=ti.cuda)
    particle_pos = ti.Vector.field(3, dtype=ti.f32, shape=solver.pn_.shape[0])

    window = ti.ui.Window('APIC 3D', res=(1280, 720), vsync=True, fps_limit=30)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.8, 0.8, 3.5)
    camera.lookat(0.8, 0.5, 0)
    camera.up(0, 1, 0)
    camera.fov(30)

    for _ in range(689):
        for e in window.get_events():
            if e.key == 's':
                with open('mpm3d.npz', 'wb') as f:
                    np.savez(f, pn=solver.pn_ / res)
        solver.anim_frame(n_steps=32)
        particle_pos.from_numpy(solver.pn_ / res)
        scene.set_camera(camera)
        scene.ambient_light((0.6, 0.6, 0.6))
        scene.point_light(pos=(-1.5, 1.5, 1.5), color=(1, 1, 1))
        scene.particles(particle_pos, color=(
            22/255, 136/255, 210/255), radius=0.0021)
        canvas.scene(scene)
        canvas.set_background_color((0.1, 0.1, 0.1))
        window.save_image('images/mpm3d/mpm3d_{:04d}.png'.format(_))
        window.show()
