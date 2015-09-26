"""
Methods for belief propagation and exact maximum marginal inference (SSL-H)

First version:  Dec 6, 2014
This version: Sept 22, 2015
Author: Wolfgang Gatterbauer <gatt@cmu.edu>
"""

from __future__ import division
from SSLH_utils import degree_matrix
from scipy.sparse import issparse
import numpy as np
PREC = 1e-4


def linBP_undirected(X, W, Hc, echo=True, numIt=10, debug=1):
    """Linearized belief propagation for undirected graphs

    Parameters
    ----------
    X : [n x k] np array
        seed belief matrix
    W : [n x n] sparse.csr_matrix
        sparse weighted adjacency matrix
    Hc : [k x k] np array
        centered coupling matrix
    echo:  Boolean (Default=True)
        whether or not echo cancellation term is used
    numIt : int
        number of iterations to perform
    debug : int (Default = 1)
        0 : no debugging and just returns F
        1 : tests for correct input, and just returns F
        2 : tests for correct input, and returns list of F

    Returns (if debug==0 or ==1)
    -------------------------------
    F : [n x k] np array
        final belief matrix, each row normalized to form a label distribution

    Returns (if debug==2 )
    ------------------------
    List of F : [(actualNumIt+1) x n x k] np array
        list of final belief matrices for each iteration, represented as 3-dimensional numpy array
        Also includes the original beliefs as first entry (0th iteration). Thus has (actualNumIt + 1) entries

    Notes
    -----
    Uses: degree_Matrix(W)

    References
    ----------
    .. [1] W. Gatterbauer, S. Guennemann, D. Koutra, and C. Faloutsos, and H. van der Vorst,
        "Linearized and Single-Pass Belief Propagation", PVLDB 8(5): 581-592 (2015).
    """
    # TODO: include convergence condition

    if debug >= 1:
        n1, n2 = W.shape
        n3, k1 = X.shape
        k2, k3 = Hc.shape
        assert(n1 == n2 & n2 == n3)
        assert(k1 == k2 & k2 == k3)
        assert(issparse(W))
    if debug == 2:
        listF = [X]     # store the beliefs for each iteration (including 0th iteration = explicit beliefs)

    if echo is False:
        F = X
        for _ in range(numIt):
            F = X + W.dot(F).dot(Hc)
            if debug == 2:
                listF.append(F)
    else:
        F = X
        H2 = Hc.dot(Hc)
        D = degree_matrix(W)
        for _ in range(numIt):
            F = X + W.dot(F).dot(Hc) - D.dot(F).dot(H2)     # W.dot(F) is short form for: sparse.csr_matrix.dot(W, F)
            if debug == 2:
                listF.append(F)

    if debug <= 1:
        return F
    else:
        return np.array(listF)
