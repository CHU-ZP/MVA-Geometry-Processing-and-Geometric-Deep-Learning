"""Utilities extracted from mesh_operators_diffusion.ipynb.

This module follows the notebook-first layout used in the reference coursework repo: the notebook keeps the narrative, and this file collects reusable definitions.
"""

import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from scipy.spatial import cKDTree


def read_off(filepath):
    """
    Reads a simple .off file
    
    Input
    --------------
    filepath : str - path to the .off file
    
    Output
    --------------
    vertices : (n,3) array of vertex coordinates (float)
    faces    : (m,3) array of faces defined by vertices index (integers)
    """
    ## TODO
    # READ THE OFF FILE with path "filepath" and return vertex and faces information
    if len(filepath.split(".")) == 1:
        filepath += ".off"
    with open(filepath, 'r') as f:
        all_lines = f.readlines()
    
    
    n_vertices = int(all_lines[1].split(" ")[0])
    n_faces = int(all_lines[1].split(" ")[1])

    vertices_list = []
    for i in range(n_vertices):
        vertices_list.append([float(x) for x in all_lines[2+i].split(" ")[:3]])

    faces_list = []
    for i in range(n_faces):
        # Be careful to convert to int. Otherwise, you can use np.array(faces_list).astype(np.int32)
        faces_list.append([int(x) for x in all_lines[2+i+n_vertices].split(" ")[1:4]])
    faces = np.array(faces_list)
    vertices = np.array(vertices_list)
    return vertices, faces


class MyMesh:
    def __init__(self, path):
        """
        Initialize the mesh from a path
        """
        self.vertices, self.faces = read_off(path)


def compute_faces_areas(vertices, faces):
    """
    Compute the area of each face
    
    Input
    --------------
    vertices : (n,3) - vertex coordinates
    faces    : (m,3) - faces defined by vertex indices
    
    Output
    --------------
    faces_areas : (m,) - area of each face
    """
    ## TODO
    # Compute the area of each triangle of the mesh.
    verts_faces = vertices[faces]
    v_1 = verts_faces[:, 0] - verts_faces[:, 1]
    v_2 = verts_faces[:, 0] - verts_faces[:, 2]
    n = np.cross(v_1, v_2)
    faces_areas = 0.5*np.linalg.norm(n, axis=-1)
    return faces_areas


def area_matrix(vertices, faces):
    """
    Compute the diagonal area matrix
    
    Input
    --------------
    vertices : (n,3) - vertex coordinates
    faces    : (m,3) - faces defined by vertex indices
    
    Output
    --------------
    A : (n,n) sparse matrix in DIAgonal format
    """
    #TODO
    # Compute the area of each vertex in the mesh.
    # Use the formula above.
    
    faces_areas = compute_faces_areas(vertices, faces)
    vertex_areas = np.zeros(vertices.shape[0])
    for idx, f in enumerate(faces):
        vertex_areas[f[0]] += faces_areas[idx]/3
        vertex_areas[f[1]] += faces_areas[idx]/3
        vertex_areas[f[2]] += faces_areas[idx]/3
    
    
    # Create a SPARSE diagonal matix from vertex areas
    N = vertices.shape[0]
    A = sparse.dia_matrix((vertex_areas, 0), shape=(N, N))
    return A


def cotan_weights(vertices, faces):
    """
    Compute cotangent weights for each face. Each vertex will carry the cotan of its angle.
    
    For a face [i, j, k], the output will be [alpha_{jk}, alpha_{ki}, alpha_{ij}].
    
    Input
    ----------
    vertices : (n,3) - The vertices of the mesh
    faces : (m,3) The faces of the mesh
    
    Output
    -------
    cotan_weights : (m,3) The cotangent weights for each face
    """
    ### TODO - Compute the cotangent weights
    verts_faces = vertices[faces]
    e_01 = verts_faces[:, 0] - verts_faces[:, 1]
    norm_01 = np.sqrt((e_01* e_01).sum(axis=-1))
    e_02 = verts_faces[:, 0] - verts_faces[:, 2]
    norm_02 = np.sqrt((e_02 * e_02).sum(axis=-1))
    e_12 = verts_faces[:, 1] - verts_faces[:, 2]
    norm_12 = np.sqrt((e_12 * e_12).sum(axis=-1))
    alpha_01 = np.arccos(np.abs((e_02 * e_12).sum(axis=-1))/(norm_02*norm_12))
    alpha_02 = np.arccos(np.abs((e_01 * e_12).sum(axis=-1))/(norm_01*norm_12))
    alpha_12 = np.arccos(np.abs((e_01 * e_02).sum(axis=-1))/(norm_01*norm_02))
    # This can be done in parallel using numpy, or by looping over the faces
    return np.concatenate((alpha_12[:, None], alpha_02[:, None], alpha_01[:, None]), axis=-1)


def cotan(x):
    ## Most of the time it's okay to use 1/np.tan, but remember that it doesn't work for x = pi/2
    return np.cos(x)/np.sin(x)


def cotan_matrix2(vertices, faces):
    ## Very naive version : might crash on computer with less than 16GB of memory (for the bunny), and super slow
    alphas = cotan_weights(vertices, faces)
    n_v = vertices.shape[0]
    W = np.zeros((n_v, n_v))
    for idx_face, (i, j, k) in enumerate(faces):
        for idx_tup, [idx_1, idx_2] in enumerate([(j, k), (k, i), (i, j)]):
            W[idx_1, idx_2] += -0.5*cotan(alphas[idx_face][idx_tup])
            W[idx_2, idx_1] += -0.5*cotan(alphas[idx_face][idx_tup])
    for i in range(n_v):
        W[i, i] -= W[i, :].sum()
    return W


def cotan_matrix_naive(vertices, faces):
    alphas = cotan_weights(vertices, faces)
    n_v = vertices.shape[0]
    # In comment: naive version of building I,J,V, loop over faces, slow because 
    # calculation inside a python loop
    I, J, V = [], [], []
    for idx_face, (i, j, k) in enumerate(faces):
        for idx_tup, [idx_1, idx_2] in enumerate([(j, k), (k, i), (i, j)]):
            value = -0.5*cotan(alphas[idx_face][idx_tup])
            I.append(idx_1)
            J.append(idx_2)
            V.append(value)

            I.append(idx_2)
            J.append(idx_1)
            V.append(value)

            I.append(idx_1)
            J.append(idx_1)
            V.append(-value)

            I.append(idx_2)
            J.append(idx_2)
            V.append(-value)
    
    I_np = np.array(I)
    J_np = np.array(J)
    V_np = np.array(V)
    W = sparse.csc_matrix((V_np, (I_np, J_np)), shape=(n_v, n_v))
    return W


def cotan_matrix(vertices, faces):
    """
    Compute the stiffness matrix
    
    Input
    --------------
    vertices : (n,3) - vertex coordinates
    faces    : (m,3) - faces defined by vertex indices
    
    Output
    --------------
    W : (n,n) sparse matrix in CSC format
    """
    # TODO
    # Compute the entries I,J,V of the stiffness matrix
    # Note that the same pair (i,j) of indices can appear multiple times in I,J
    # The corresponding values in V are then summed by scipy.
    alphas = cotan_weights(vertices, faces)
    n_v = vertices.shape[0]
    # Here, use advantage of numpy arrays
    list_I = []
    list_J = []
    list_V = []
    for idx_tup, (idx_1, idx_2) in enumerate([(1, 2), (2, 0), (0, 1)]):
        value = -0.5*cotan(alphas[:, idx_tup])
        list_I += [faces[:, idx_1], faces[:, idx_2], faces[:, idx_1], faces[:, idx_2]]
        list_J += [faces[:, idx_2], faces[:, idx_1], faces[:, idx_1], faces[:, idx_2]]
        list_V += [value, value, -value, -value]
    I = np.concatenate(list_I, axis=0)
    J = np.concatenate(list_J, axis=0)
    V = np.concatenate(list_V, axis=0)
    W = sparse.csc_matrix((V, (I, J)), shape=(n_v, n_v))
    return W


class MyMesh:
    def __init__(self, path):
        """
        Initialize the mesh from a path
        """
        self.vertices, self.faces = read_off(path)
        
    def compute_laplacian(self):
        self.A = area_matrix(self.vertices, self.faces)
        self.W = cotan_matrix(self.vertices, self.faces)


def diffuse_full(f, mesh, t):
    """
    Diffuse a function f on a mesh for time t
    
    Input
    --------------
    f       : (n,) - function values
    mesh    : MyMesh - mesh on which to diffuse
    t       : float - time for which to diffuse
    
    Output
    --------------
    f_diffuse : (n,) values of f after diffusion
    """
    # TODO
    # Solve the Diffusion process using the formula above
    left_matrix = mesh.A + t*mesh.W
    right_term = f
    f_diffuse = sparse.linalg.spsolve(left_matrix, right_term)
    return f_diffuse


class MyMesh:
    def __init__(self, path):
        """
        Initialize the mesh from a path
        """
        self.vertices, self.faces = read_off(path)
        
    def compute_laplacian(self):
        self.A = area_matrix(self.vertices, self.faces)
        self.W = cotan_matrix(self.vertices, self.faces)
        
    def compute_eigendecomposition(self, K):
        self.eigenvalues, self.eigenvectors = sparse.linalg.eigsh(self.W, M=self.A,
                                                                  k=K, sigma=-0.01)


def diffuse_spectral(f, mesh, t, k):
    """
    Diffuse a function f on a mesh for time t
    
    Input
    --------------
    f       : (n,) - function values
    mesh    : MyMesh - mesh on which to diffuse
    t       : float - time for which to diffuse
    k       : int - size of the basis to use for diffusion
    
    Output
    --------------
    f_diffuse : (n,) values of f after diffusion
    """
    # TODO
    # Solve the Diffusion process using spectral analysis.
    # Use the formula above.
    # Note that you should return the function u_t, NOT alpha_t !
    beta = (f[:, None] * (mesh.A @ mesh.eigenvectors[:, :k])).sum(axis=0)
    alpha = np.exp(-mesh.eigenvalues[:k]*t)*beta
    f_diffuse = ((mesh.eigenvectors[:, :k])*alpha[None, :]).sum(axis=-1)
    return f_diffuse


def compute_HKS(mesh, n_times, k):
    # Defines a list of time parameters at which to compute HKS
    abs_ev = sorted(np.abs(mesh.eigenvalues[:k]))
    t_list = np.geomspace(4*np.log(10)/abs_ev[-1], 4*np.log(10)/abs_ev[1], n_times)  # (n_times,)
    X_t = np.exp(-mesh.eigenvalues[None, :k]*t_list[:, None])
    C_t = 1./X_t.sum(axis=-1)
    ## TODO COMPUTE HKS
    HKS = C_t * (X_t[None, :] * (mesh.eigenvectors**2)[:, None, :k]).sum(axis=-1)
    return HKS


class KNNSearch(object):
    DTYPE = np.float32
    NJOBS = 4

    def __init__(self, data):
        self.data = np.asarray(data, dtype=self.DTYPE)
        self.kdtree = cKDTree(self.data)

    def query(self, kpts, k, return_dists=False):
        kpts = np.asarray(kpts, dtype=self.DTYPE)
        nndists, nnindices = self.kdtree.query(kpts, k=k, workers=self.NJOBS)
        if return_dists:
            return nnindices, nndists
        else:
            return nnindices

    def query_ball(self, kpt, radius):
        kpt = np.asarray(kpt, dtype=self.DTYPE)
        assert kpt.ndim == 1
        nnindices = self.kdtree.query_ball_point(kpt, radius, n_jobs=self.NJOBS)
        return nnindices


def grad_f(f, vertices, faces, normals):
    """
    Compute the gradient of a function on a mesh

    Parameters
    --------------------------
    f          : (n,) function value on each vertex
    vertices   : (n,3) coordinates of vertices
    faces      : (m,3) indices of vertices for each face
    normals    : (m,3) normals coordinate for each face
    face_area : (m,) - Optional, array of per-face area, for faster computation
    use_sym    : bool - If true, uses the (slower but) symmetric expression
                 of the gradient

    Output
    --------------------------
    gradient : (m,3) gradient of f on the mesh
    """
    v1 = vertices[faces[:,0]]  # (m,3)
    v2 = vertices[faces[:,1]]  # (m,3)
    v3 = vertices[faces[:,2]]  # (m,3)

    f1 = f[faces[:,0]]  # (m,)
    f2 = f[faces[:,1]]  # (m,)
    f3 = f[faces[:,2]]  # (m,)

    # Compute area for each face
    face_areas = 0.5 * np.linalg.norm(np.cross(v2-v1,v3-v1),axis=1)  # (m)

   

    grad1 = np.cross(normals, v3-v2)/(2*face_areas[:,None])  # (m,3)
    grad2 = np.cross(normals, v1-v3)/(2*face_areas[:,None])  # (m,3)
    grad3 = np.cross(normals, v2-v1)/(2*face_areas[:,None])  # (m,3)

    gradient = f1[:,None] * grad1 + f2[:,None] * grad2 + f3[:,None] * grad3

    return gradient


def div_f(f, vertices, faces, normals, vert_areas):
    """
    Compute the divergence of a vector field on a mesh

    Parameters
    --------------------------
    f          : (m,3) vector field on each face
    vertices   : (n,3) coordinates of vertices
    faces      : (m,3) indices of vertices for each face
    normals    : (m,3) normals coordinate for each face
    vert_area : (m,) - array of per-vertex area

    Output
    --------------------------
    divergence : (n,) divergence of f on the mesh
    """
    n_vertices = vertices.shape[0]

    v1 = vertices[faces[:,0]]  # (m,3)
    v2 = vertices[faces[:,1]]  # (m,3)
    v3 = vertices[faces[:,2]]  # (m,3)

    grad1 = np.einsum('ij,ij->i', np.cross(normals, v3 - v2) / 2, f)  # (m,)
    grad2 = np.einsum('ij,ij->i', np.cross(normals, v1 - v3) / 2, f)  # (m,)
    grad3 = np.einsum('ij,ij->i', np.cross(normals, v2 - v1) / 2, f)  # (m,)

    I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])  # (3*m)
    J = np.zeros_like(I)
    V = np.concatenate([grad1, grad2, grad3])

    div_val = sparse.coo_matrix((V, (I, J)), shape=(n_vertices, 1)).todense()

    return np.asarray(div_val).flatten() / vert_areas


def get_mean_edge_length(mesh):
    e1 = mesh.vertices[mesh.faces[:, 1]] - mesh.vertices[mesh.faces[:, 0]]
    e2 = mesh.vertices[mesh.faces[:, 2]] - mesh.vertices[mesh.faces[:, 0]]
    e3 =  mesh.vertices[mesh.faces[:, 2]] - mesh.vertices[mesh.faces[:, 1]]
    l1 = np.linalg.norm(e1, axis=-1)
    l2 = np.linalg.norm(e2, axis=-1)
    l3 = np.linalg.norm(e3, axis=-1)
    return (np.mean(l1) + np.mean(l2) + np.mean(l3))/3


def face_normals(mesh):
    e1 = mesh.vertices[mesh.faces[:, 1]] - mesh.vertices[mesh.faces[:, 0]]
    e2 = mesh.vertices[mesh.faces[:, 2]] - mesh.vertices[mesh.faces[:, 0]]
    cross = np.cross(e1, e2)
    mesh_normals = cross/np.linalg.norm(cross, axis=-1, keepdims=True)
    return mesh_normals


def heat_distance(mesh, idx):
    dirac = np.zeros(mesh.vertices.shape[0])
    dirac[idx] = 1
    # Computing diffusion of dirac 
    h = get_mean_edge_length(mesh)
    diff_h = diffuse_full(dirac, mesh, h**2)
    # Computing face normals
    mesh_normals = face_normals(mesh)

    # Heat method
    grad_diff = grad_f(diff_h, mesh.vertices, mesh.faces, mesh_normals)
    X = -grad_diff/np.linalg.norm(grad_diff, axis=-1, keepdims=True)
    div_X = div_f(X, mesh.vertices, mesh.faces, mesh_normals, mesh.A.diagonal())
    # Linear system (don't forget the mass matrix when discretizing!) 
    left = mesh.W 
    right = mesh.A @ div_X
    sol = sparse.linalg.spsolve(left, right)
    dists = sol - np.min(sol)
    return dists, h


def line_plot(dists, h):
    # Simple function to plot the level set of geod distances
    plot_func = copy.deepcopy(dists)
    eps = dists.max()/15
    color_plot = plt.cm.coolwarm((plot_func-plot_func.min())/(plot_func.max()-plot_func.min()))
    for i in range(15):
        color_plot[np.logical_and(dists>i*eps-h/2, dists<i*eps+h/2), :] = 0
    return color_plot


__all__ = ['read_off', 'MyMesh', 'compute_faces_areas', 'area_matrix', 'cotan_weights', 'cotan', 'cotan_matrix2', 'cotan_matrix_naive', 'cotan_matrix', 'diffuse_full', 'diffuse_spectral', 'compute_HKS', 'KNNSearch', 'grad_f', 'div_f', 'get_mean_edge_length', 'face_normals', 'heat_distance', 'line_plot']
