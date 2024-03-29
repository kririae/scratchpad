#!/usr/bin/env python3

import copy

import igl
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from matplotlib import cm

import open3d as o3d  # For visualization

dataset = o3d.data.BunnyMesh()
mesh = o3d.io.read_triangle_mesh(dataset.path)

v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)

# Build the Laplacian matrix based on the mesh
L = -igl.cotmatrix(v, f)

# Solve the eigenvalue problem, in a brute-force way
print('Solving the eigenvalue problem...')
_, ev = eigsh(L, 32, which='SM')
print(_)

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(v)
mesh.triangles = o3d.utility.Vector3iVector(f)
mesh.compute_vertex_normals()

for i in range(0, 32, 4):
    color = cm.plasma(ev[:, i] / np.max(ev[:, i]))[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([mesh])
