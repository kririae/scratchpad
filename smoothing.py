#!/usr/bin/env python3
import os
import igl
from tqdm import tqdm
import scipy as sp
import numpy as np
import open3d as o3d


def plot(V: np.ndarray, F: np.ndarray, C: np.ndarray = None):
    """Plot a mesh given its vertices and faces."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(F)
    mesh.compute_vertex_normals()
    if C is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(C)
    o3d.visualization.draw_geometries([mesh])


def main():
    dataset = o3d.data.ArmadilloMesh()
    mesh = o3d.io.read_triangle_mesh(dataset.path)

    v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)

    # Build the Laplacian matrix based on the mesh
    L = igl.cotmatrix(v, f)
    h = 100

    # Emplot explicit smoothing
    A = sp.sparse.eye(L.shape[0]) - h * L
    for i in range(3):
        x, _ = sp.sparse.linalg.cg(A, v[:, i], maxiter=512)
        v[:, i] = x
    plot(v, f)


if __name__ == '__main__':
    main()
