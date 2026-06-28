"""Utilities extracted from TD2_ZepengCHU.ipynb.

This module follows the notebook-first layout used in the reference coursework repo: the notebook keeps the narrative, and this file collects reusable definitions.
"""

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.linalg as sla
from scipy.sparse import bmat, csc_matrix, csr_matrix, issparse, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree


def get_border_edges(triangles):
    """
    Get the border edges of a mesh. In no particular order

    Parameters
    ----------
    triangles : ndarray of shape (n_triangles, 3)

    Returns
    -------
    border_edges : list of list of length n_border_edges.
                   The border edges of the mesh. Element i contains the two
                     vertices of the i-th border edge.
    """

    edges = np.vstack([
        triangles[:, [0, 1]],
        triangles[:, [1, 2]],
        triangles[:, [2, 0]]
    ])

    edges = np.sort(edges, axis=1)

    edges_tuple = [tuple(e) for e in edges]

    unique_edges, counts = np.unique(edges_tuple, axis=0, return_counts=True)
    border_edges = unique_edges[counts == 1]
    border_edges = border_edges.tolist()

    return border_edges


def get_n_points_on_circle(n_points):
    """
    Builds n_points evenly spaced points on the unit circle.

    Parameters
    ----------
    n_points : int
        Number of points to generate

    Returns
    -------
    points : ndarray of shape (n_points, 2)
        The points on the unit circle
    """
    # YOUR CODE HERE
    # even angles between 0 and 2π
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    x = np.cos(angles)
    y = np.sin(angles)

    points = np.stack([x, y], axis=1)

    return points


def find_next_edge(current_edge, remaining_edges):
        for i, edge in enumerate(remaining_edges):
            if current_edge[1] in edge:
                return i, edge #if edge[0]==current_edge[1] else edge[::-1]
        return None, None


def build_ordered_edges(triangles):
    """
    Compute an ordered list of edges that form a path around the border of the mesh.

    Parameters
    ----------
    triangles : ndarray of shape (n_triangles, 3)

    Returns
    -------
    ordered_edge_list : list of list of length n_border_edges.
                        The border edges of the mesh. Element i contains the two
                        vertices of the i-th border edge.
    """

    border_edges = get_border_edges(triangles)  # (p,2)
    ordered_edge_list = []

    remaining = [list(e) for e in border_edges]
    ordered_edge_list = []

    while remaining:

        current_edge = remaining.pop(0)
        u, v = current_edge
        loop = [[u, v]]

        while True:
            idx, edge = find_next_edge(loop[-1], remaining)
            if edge is None:
                break
            if edge[0] == loop[-1][1]:
                oriented = [edge[0], edge[1]]
            else:
                oriented = [edge[1], edge[0]]
            loop.append(oriented)
            remaining.pop(idx)
            if loop[-1][1] == loop[0][0]:
                break

        ordered_edge_list.extend(loop)

    return ordered_edge_list


def build_M(faces):
    """
    Build the M matrix above with values only at border edges.
    M can be build by adding the formula for M_ij for each edge of each face on the mesh.
    Coefficients at interior edges will vanish because they appear twice with opposite signs.

    Parameters
    ----------
    faces : ndarray of shape (n_faces, 3)

    Returns
    -------
    M : scipy.sparse.csr_matrix of shape (n_vertices, n_vertices)
    """
    faces = np.asarray(faces, dtype=np.int64)
    n_vertices = int(faces.max()) + 1 if faces.size else 0
    M = lil_matrix((n_vertices, n_vertices), dtype=np.float64)

    for i, j, k in faces:
        M[i, j] += 0.5
        M[j, k] += 0.5
        M[k, i] += 0.5
        M[j, i] -= 0.5
        M[k, j] -= 0.5
        M[i, k] -= 0.5

    M = M.tocsr()
    return M


class KNNSearch:
    """
    Simple wrapper for KD-tree nearest neighbor search.
    """
    def __init__(self, X):
        self.tree = cKDTree(X)

    def query(self, Y, k=1):
        # Return only the indices of nearest neighbors
        _, idx = self.tree.query(Y, k)
        return idx


def compute_nearest_neighbor(X, Y):
    """
    Compute the nearest neighbor in Y for each point in X

    Parameters:
    -----------
    X : (n, d) array of points
    Y : (m, d) array of points

    Returns:
    --------
    nearst_neighbor : (n,) array of indices of the nearest neighbor in Y for X
    """
    # TODO DO NOT USE LOOPS

    querier = KNNSearch(Y)
    nn_indices = querier.query(X, 1)
    return nn_indices


def compute_rigid_transform(X_source, X_target):
    """
    Compute the optimal rotation matrix and translation that aligns two point clouds of the same size X_source and X_target.
    This rotation should be applied to X_source.

    Parameters:
    -----------
    X_source : (n, d) array of points
    Y_target : (n, d) array of points

    Returns:
    --------
    R : (d, d) rotation matrix
    t : (d,) translation vector
    """
    # TODO
    Xs = np.asarray(X_source, dtype=np.float64)
    Xt = np.asarray(X_target, dtype=np.float64)
    assert Xs.shape == Xt.shape and Xs.ndim == 2, "Shapes must match (n, d)."

    mu_s = Xs.mean(axis=0)
    mu_t = Xt.mean(axis=0)
    Xs_c = Xs - mu_s
    Xt_c = Xt - mu_t

    H = Xs_c.T @ Xt_c  # (d, d)

    U, S, Vt = np.linalg.svd(H, full_matrices=False)

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = mu_t - R @ mu_s
    return R, t


def transform_pointcloud(X,R,t):
    """
    Transform a point cloud X by a rotation matrix R and a translation vector t.

    Parameters:
    -----------
    X : (n, d) array of points
    R : (d, d) rotation matrix
    t : (d,) translation vector

    Returns:
    --------
    X_transformed : (n, d) array of transformed points
    """
    return X @ R.T + t


def icp_align(X_source, Y_target, n_iter=10):
    """
    Align two point clouds X_source and Y_target using the ICP algorithm.

    Parameters:
    -----------
    X_source : (n, d) array of points
    Y_target : (m, d) array of points
    n_iter   : int - number of iterations of the ICP algorithm

    Returns:
    --------
    X_aligned : (n, d) array of aligned points
    """
    X_aligned = X_source.copy()
    for i in range(n_iter):
        nn_indices = compute_nearest_neighbor(X_aligned, Y_target)
        Y_matched = Y_target[nn_indices]

        R, t = compute_rigid_transform(X_aligned, Y_matched)

        X_aligned = transform_pointcloud(X_aligned, R, t)

        # Print Error
        mean_err = np.mean(np.linalg.norm(X_aligned - Y_matched, axis=1))
        print(f"ICP iter {i+1}/{n_iter} - mean error = {mean_err:.6f}")

    return X_aligned


def plot_superimposed(mesh1, mesh2, color_1=[139, 0, 139], color_2=[210, 105, 30], *args, **kwargs):
    """
    Plot the superposition of the two meshes
    """
    #2B7FFF#2B7FFF
    if meshplot:
        cmap_1 = np.ones(mesh1.vertices.shape)*np.array(color_1)/255.
        cmap_2 = np.ones(mesh2.vertices.shape)*np.array(color_2)/255.
    else:
        cmap_1 = np.ones(mesh1.vertices.shape)*np.array(color_1)
        cmap_2 = np.ones(mesh2.vertices.shape)*np.array(color_2)
    cmap = np.concatenate([cmap_1, cmap_2], axis=0)
    mesh = TriMesh(np.concatenate([mesh1.vertices, mesh2.vertices], axis=0),
                   np.concatenate([mesh1.faces, mesh2.faces+mesh1.n_vertices], axis=0)).process(k=0)
    return renderer.plot(mesh, cmap, *args, **kwargs)


def get_per_vertex_neighbors(faces):
    """
    Compute per-vertex neighbors from a list of triangles

    Parameters:
    -----------
    faces : (n, 3) array of vertex indices for each triangle

    Returns:
    --------
    neighbors : list of lists of vertex indices
    """

    neighbors = [set() for _ in range(faces.max()+1)]
    for face in faces:
        neighbors[face[0]].add(face[1])
        neighbors[face[0]].add(face[2])

        neighbors[face[1]].add(face[0])
        neighbors[face[1]].add(face[2])

        neighbors[face[2]].add(face[0])
        neighbors[face[2]].add(face[1])

    return [list(n) for n in neighbors]


def get_arap_edge_covariance(x,y, cotan_matrix, per_vertex_neighbors):
    """
    Compute the covariance matrix of the edge between x and y. (Formula B_i)

    Parameters:
    -----------
    x : (n,3) array of coordinates of x
    y : (n, 3) array of coordinates of y
    cotan_matrix : (n,n) cotan matrix of the mesh
    per_vertex_neighbors : (n,) list with list of neighbors of each vertex

    Returns:
    --------
    covariances : (n, 3, 3) covariance matrices of the edge between x and y


    """

    x = np.asarray(x)
    y = np.asarray(y)
    n, d = x.shape
    covariances = np.zeros((n, d, d), dtype=x.dtype)

    # if issparse(cotan_matrix):
    #     W = cotan_matrix.tocsr()
    #     for i, neigh in enumerate(per_vertex_neighbors):
    #         xi, yi = x[i], y[i]
    #         Bi = np.zeros((d, d), dtype=x.dtype)
    #         for j in neigh:
    #             wij = W[i, j]
    #             wij = float(wij) if np.ndim(wij) else wij
    #             if wij == 0.0:
    #                 continue
    #             Bi += wij * np.outer(yi - y[j], xi - x[j])
    #         covariances[i] = Bi
    #     return covariances

    # Dense / array-like path
    W = cotan_matrix.toarray() if hasattr(cotan_matrix, "toarray") else np.asarray(cotan_matrix)
    x = np.asarray(x); y = np.asarray(y)
    n, d = x.shape
    cov = np.zeros((n, d, d), dtype=x.dtype)

    for i, neigh in enumerate(per_vertex_neighbors):
        if not neigh:
            continue
        nbr = np.asarray(neigh, dtype=int)
        xi, yi = x[i], y[i]
        wij   = W[i, nbr]          # (k,)
        Ydiff = yi - y[nbr]        # (k,d)
        Xdiff = xi - x[nbr]        # (k,d)
        cov[i] = (Ydiff.T * wij) @ Xdiff
    return cov


def get_rot_from_covariances(covariances):
    """
    Compute optimal rotation matrix from edge covariance matrices, using SVD.

    Parameters:
    -----------
    covariances : (n, 3, 3) covariance matrices for each vertex

    Returns:
    --------
    rots : (n, 3, 3) rotation matrices for each vertex

    """
    covariances = np.asarray(covariances)
    n, d, _ = covariances.shape
    rots = np.empty_like(covariances)

    for i in range(n):
        U, _, Vt = np.linalg.svd(covariances[i], full_matrices=False)
        R = U @ Vt
        # Enforce det(R)=+1 (avoid reflections)
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        rots[i] = R

    return rots


def compute_ARAP_rotated_vert(vertices, rotations, cotan_matrix, per_vertex_neighbors):
    """
    Compute the right hand term of the ARAP linear system (formula b_i)

    Parameters:
    -----------
    vertices : (n, 3) array of vertices
    rotations : (n, 3, 3) array of rotation matrices
    cotan_matrix : (n,n) cotan matrix of the mesh
    per_vertex_neighbors : (n,) list with list of neighbors of each vertex

    Returns:
    --------
    b : (n, 3) right hand term of the ARAP linear system
    """
    X = np.asarray(vertices)
    R = np.asarray(rotations)
    W = np.asarray(cotan_matrix)
    n, d = X.shape
    b = np.zeros((n, d), dtype=X.dtype)

    for i, neigh in enumerate(per_vertex_neighbors):
        xi, Ri = X[i], R[i]
        acc = np.zeros(d, dtype=X.dtype)
        for j in neigh:
            acc += 0.5 * W[i, j] * (Ri + R[j]) @ (xi - X[j])
        b[i] = acc
    return b


def arap_solve_once(X, Y0, W_wij, neighbors, id_lock, y_lock, n_iters=5):
    n, d = X.shape
    Y = Y0.copy()

    diag = np.sum(W_wij, axis=1)
    L = -W_wij.copy()
    np.fill_diagonal(L, diag)

    Lc = L.copy()
    for k in id_lock:
        Lc[k, :] = 0.0
        Lc[k, k] = 1.0

    for _ in range(n_iters):
        cov = get_arap_edge_covariance(X, Y, W_wij, neighbors)
        R = get_rot_from_covariances(cov)
        b = compute_ARAP_rotated_vert(X, R, W_wij, neighbors)
        b[id_lock] = y_lock
        for c in range(d):
            Y[:, c] = np.linalg.solve(Lc, b[:, c])
    return Y


def choose_anchor_indices(Y, T, k_keep=0.2, mutual=True):
    n = Y.shape[0]
    tree_T = cKDTree(T)
    d_st, nn_st = tree_T.query(Y, k=1)
    if mutual:
        tree_S = cKDTree(Y)
        _, nn_ts = tree_S.query(T, k=1)
        mutual_mask = (nn_ts[nn_st] == np.arange(n))
    else:
        mutual_mask = np.ones(n, dtype=bool)

    k = max(1, int(k_keep * n))
    idx_sorted = np.argsort(d_st)
    cand = idx_sorted[:k]
    id_lock = cand[mutual_mask[cand]]
    if id_lock.size == 0:
        id_lock = idx_sorted[:1]
    y_lock = T[nn_st[id_lock]]
    return id_lock, y_lock


def nonrigid_icp_arap(source_vertices, source_faces, target_vertices, W_wij,
                      n_icp=5, n_arap=5, keep_ratio=0.2, mutual_nn=True):
    neighbors = get_per_vertex_neighbors(source_faces)
    Y = source_vertices.copy()
    for _ in range(n_icp):
        id_lock, y_lock = choose_anchor_indices(Y, target_vertices,
                                                k_keep=keep_ratio, mutual=mutual_nn)
        Y = arap_solve_once(X=source_vertices, Y0=Y, W_wij=W_wij, neighbors=neighbors,
                            id_lock=id_lock, y_lock=y_lock, n_iters=n_arap)
    return Y


__all__ = ['get_border_edges', 'get_n_points_on_circle', 'find_next_edge', 'build_ordered_edges', 'build_M', 'KNNSearch', 'compute_nearest_neighbor', 'compute_rigid_transform', 'transform_pointcloud', 'icp_align', 'plot_superimposed', 'get_per_vertex_neighbors', 'get_arap_edge_covariance', 'get_rot_from_covariances', 'compute_ARAP_rotated_vert', 'arap_solve_once', 'choose_anchor_indices', 'nonrigid_icp_arap']
