import itertools
from functools import reduce 
import math
import numpy as np


def missing_vertex(clique, face):
    """Get the only element in clique that is not present in face."""
    missing = set(clique).difference(set(face))
    return list(missing)[0]

def count_k_simplex(n, k):
    """Count the maximum number of k-simplex for a simplicial complex of n vertices."""
    return math.comb(n, k+1)

def generate_k_simplex(n, k):
    """Generate all the k-simplex, following the global ordering of vertices, for a simplicial complex of n vertices."""
    if k == -1: 
        return [[]]
    else:
        return [list(t) for t in itertools.combinations(range(n), k+1)]
    
def hilbert_boundary_op(k, abs, header=False):
    """
    Generate the function $\partial_k: \mathcal{C}_k \to \mathcal{C}_{k-1}$.
    """

    # the vertices of the graph are the number of 0-simplices
    n = len(abs[0])
    
    if k == 0:
        # zero-th homology (see math.stackexchange.com/questions/4333707/definition-of-zeroth-homology)
        # every vertex gets mapped to the constant element 'zero' 
        operator = np.ones(shape=(1, n))
        if header:
            return pd.DataFrame(operator, index=['zero'], columns=[str(s) for s in abs[0]]) 
        else:
            return operator
    
    # map k-simplices to (k-1)-simplices
    candidate_cliques = generate_k_simplex(n, k)
    candidate_faces = generate_k_simplex(n, k-1)
    c_in = count_k_simplex(n, k)
    c_out = count_k_simplex(n, k-1)

    # create an empty operator
    operator = np.zeros(shape=(c_out, c_in))

    for i_clique, clique in enumerate(candidate_cliques):
        if clique in abs[k]:
            for i_face, face in enumerate(candidate_faces):
                if face in abs[k-1]:
                    if set(face) <= set(clique):
                        missing_vtx = missing_vertex(clique, face)
                        j = clique.index(missing_vtx)
                        operator[i_face][i_clique] = (-1)**j
    
    operator = operator.astype(int)
    if not header:
        return operator
    else:
        row_names = [str(item) for item in candidate_faces]
        col_names = [str(item) for item in candidate_cliques]
        df = pd.DataFrame(operator, index=row_names, columns=col_names)
        return df
    
def hilbert_laplacian_op(k, abs):
    """
    Generate the function $\Delta_k: \mathcal{C}_k \to \mathcal{C}_{k}$.
    """
    assert k >= 0, "Cannot have negative Laplacians"

    # Equation (8) in https://arxiv.org/pdf/2108.06547.pdf
    if k == 0:
        B1 = hilbert_boundary_op(1, abs)
        B1_dag = B1.conj().T
        return B1 @ B1_dag
        
    # Equation (6) in https://arxiv.org/pdf/2108.06547.pdf
    assert k < len(abs[0]) - 1, "The order 'k' must be lower than the number of 0-simplices minus one"
    Bk = hilbert_boundary_op(k, abs)
    BK_dag = Bk.conj().T
    Bk1 = hilbert_boundary_op(k+1, abs)
    Bk1_dag = Bk1.conj().T
    return BK_dag @ Bk + Bk1 @ Bk1_dag

def test():
    triangle = {
        0: [[0], [1], [2]],
        1: [[0, 1], [0, 2], [1, 2]],
        2: [[0, 1, 2]]
    }
    B0 = hilbert_boundary_op(0, triangle, header=True)
    B1 = hilbert_boundary_op(1, triangle, header=True)
    Lneg1 = hilbert_laplacian_op(-1, triangle)
    L0 = hilbert_laplacian_op(0, triangle)
    L1 = hilbert_laplacian_op(1, triangle)
    eigvals, eigvecs = np.linalg.eigh(L0)

def generate_clique_complex(G):
    list_cliques = list(nx.enumerate_all_cliques(G))
    abs = {size-1: list(filter(lambda k: len(k) == size, list_cliques)) for size in range(1, len(G.nodes)+1)}
    return abs


square = {
    0: [[0], [1], [2], [3]],
    1: [[0, 1], [1, 2], [2, 3], [0, 3]],
    2: []
}

np.set_printoptions(precision=3, suppress=True)

L0 = hilbert_laplacian_op(0, square)
L1 = hilbert_laplacian_op(1, square)
print("Eigenvalues L0:", np.linalg.eigh(L0)[0])
print("Eigenvalues L1:", np.linalg.eigh(L1)[0])
B1 = hilbert_boundary_op(1, square)
print(f"Boundary_1:\n{B1}")
print(f"Boundary_2:\n{hilbert_boundary_op(2, square)}")
print(f"L_1:\n{hilbert_laplacian_op(1, square)}")
print(f"B1B1:\n{B1.T @ B1}")

import sympy
import functools

def cycles(k, abs):
    partial_k = hilbert_boundary_op(k, abs)
    m = sympy.Matrix(partial_k).nullspace()
    if len(m) > 0:
        m = functools.reduce(lambda a, b: a.row_join(b), m)
        return np.array(m, dtype=int)
    else:
        return np.array([])

def boundaries(k, abs):
    partial_k_plus_1 = hilbert_boundary_op(k+1, abs)
    if partial_k_plus_1.size > 0 and not np.all(np.isclose(partial_k_plus_1, 0)):
        m = sympy.Matrix(partial_k_plus_1).columnspace()
        m = functools.reduce(lambda a, b: a.row_join(b), m)
        return np.array(m, dtype=int)
    else:
        return np.array([])
    
print(f"Cycles L1:\n{cycles(1, square)}")
print(f"Boundaries L1:\n{boundaries(1, square)}")

