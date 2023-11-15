import itertools
from functools import reduce 
import math
import numpy as np
import pandas as pd
import sympy as sp
import networkx as nx


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
    if k == -1:
        B0 = hilbert_boundary_op(0, abs)
        B0_dag = B0.conj().T
        return B0 @ B0_dag
        
    # L_k = B_k^dag B_k + B_(k+1) B_(k+1)^dag
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