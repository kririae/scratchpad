#!/usr/bin/env python

import open3d as o3d
import numpy as np
import copy
import scipy.sparse as sp
from matplotlib import cm
from scipy.sparse.linalg import spsolve

print(o3d.__version__)

# Deformation force weight
w = 1


def build_laplacian_matrix(mesh):
    # Init the Laplacian matrix
    num_vertices = len(mesh.vertices)
    L = sp.lil_array((num_vertices, num_vertices))

    # Fill the Laplacian matrix with mesh topology
    mesh.compute_adjacency_list()
    for i in range(num_vertices):
        L[i, i] = len(mesh.adjacency_list[i])
        for j in mesh.adjacency_list[i]:
            L[i, j] = -1

    return L, copy.deepcopy(np.asarray(mesh.vertices))


def create_masked_identity(num_vertices, indices):
    mask = np.zeros(num_vertices)
    mask[indices] = 1
    mask = sp.diags(mask, 0)
    return mask


def create_masked_delta(delta, indices):
    tmp = copy.deepcopy(delta)
    tmp[indices] = 0
    tmp = delta - tmp
    return tmp


# Create the raw mesh
mesh = o3d.geometry.TriangleMesh.create_cylinder(split=16)

# Deform the mesh vertices
indices_top = [idx for idx, i in enumerate(mesh.vertices) if i[2] == 1]
indices_bottom = [idx for idx, i in enumerate(mesh.vertices) if i[2] == -1]
edited_vertices = copy.deepcopy(np.asarray(mesh.vertices))
edited_vertices[indices_top] += [3.5, 0, 0]  # move top vertices in x direction
indices = indices_top + indices_bottom

# Build the Laplacian matrix
L, x = build_laplacian_matrix(mesh)

num_vertices = len(mesh.vertices)
delta = L @ x
delta.resize((2 * num_vertices, 3))

Le = sp.lil_array((2 * num_vertices, num_vertices))
Le[:num_vertices, :num_vertices] = L
Le[num_vertices:, :num_vertices] = w * \
    create_masked_identity(num_vertices, indices)

delta[num_vertices:] = w * create_masked_delta(edited_vertices, indices)

# Prepare the system of equations
Lt = Le.transpose()
LtL = Lt @ Le
LtD = Lt @ delta

# Solve the system of equations
x_ = spsolve(LtL, LtD)  # Solve for L^T L x = L^T delta

delta_p = mesh.vertices - x_
delta_p = np.linalg.norm(delta_p, axis=1)

# Map the delta_p to the color map
delta_p = cm.viridis(delta_p / np.max(delta_p))[:, :3]

mesh_reconstructed = copy.deepcopy(mesh)
mesh_reconstructed.vertices = o3d.utility.Vector3dVector(x_)
mesh_reconstructed.compute_vertex_normals()
mesh_reconstructed.vertex_colors = o3d.utility.Vector3dVector(delta_p)

mesh_edited = copy.deepcopy(mesh)
mesh_edited.vertices = o3d.utility.Vector3dVector(edited_vertices)
o3d.visualization.draw_geometries([mesh_reconstructed])
