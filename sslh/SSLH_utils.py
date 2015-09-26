"""
Useful general functions that are repeatedly used by the main methods of SSL-H

Nomenclature:
    X, Xc, X0: [n x k] np.array of normalized (centered, explicit) beliefs
    Xd: node id -> class id dictionary
    H, Hc: [k x k] np.array of (centered) coupling matrix
    W: sparse edge matrix

First version: Dec 6, 2014
This version: Sept 21, 2015
Author: Wolfgang Gatterbauer <gatt@cmu.edu>
"""


from __future__ import division
from scipy.sparse import csr_matrix, kron, diags
from scipy.optimize import newton
from pyamg.util.linalg import approximate_spectral_radius    #pyamg seems to take a long time to import, perhaps fork
from sklearn.preprocessing import normalize, scale
from itertools import product
import numpy as np
PREC = 1e-4                 # defined precision for checking sum = 1+-PREC
SPECTRAL_TOLERANCE = 1e-8   # if std of a centered matrix is too small, then those are basically rounding errors
# np.random.seed(0)         # seed for random generator


def check_normalized_beliefs(X):
    """Verifies that rows in a [n x k] np.array of beliefs are valid probability distributions
    Allows n=1, thus one dimensional vector [ , ] in addition to 2-dimensional [[ , ]]
    """
    assert type(X).__module__ == "numpy"    # make sure it is not a matrix
    correct = X.all() >= 0
    if len(X.shape) == 1:                   # special case in case n=1 and no nesting: [[...]] -> [...]
        return correct and np.abs(np.sum(X)-1) <= PREC
    else:
        return correct and (np.abs(np.sum(X,1)-1) <= PREC).all()


def check_centered_beliefs(Xc):
    """Verifies  that rows in a [n x k] np.array of centerd beliefs sum up to 1
    Allows [ , ] in addition to [[ , ]] for n=1
    """
    assert type(Xc).__module__ == "numpy"   # make sure it is not a matrix
    if len(Xc.shape) == 1:                  # special case in case n=1 and no nesting: [[...]] -> [...]
        return np.abs(np.sum(Xc)) <= PREC
    else:
        return (np.abs(np.sum(Xc,1)) <= PREC).all()


def check_dictionary_beliefs(Xd, n=None, k=None, zeroindexing=True, fullBeliefs=False):
    """Verify that the explicit belief dictionary Xd (node id -> class id) has certain properties
    fullBeliefs = True: every node has at least one belief
    zeroindexing  = True if first node is indexed by 0 [instead of by 1] (only for fullBeliefs)
    """
    keys = Xd.keys()
    values = set(Xd.values())
    v0 = min(keys)
    v1 = max(keys)
    n2 = len(Xd.keys())                     # number of node assignments
    j0 = min(values)
    j1 = max(values)
    k2 = len(values)                        # number of different classes
    if zeroindexing:
        correct = v0 >=0
        delta = 0
    else:
        correct = v0 >=1
        delta = -1
    if fullBeliefs:
        correct = correct and n2 == len(set(Xd.keys()))
        correct = correct and v0 == - delta                # minimum node id = 0 (or 1)
        correct = correct and j0 == - delta
        correct = correct and v1 == n2 - 1 - delta         # maximum node id = length - 1 (or 0)
        correct = correct and j1 == k2 - 1 - delta         # maximum class id = size - 1 (or 0)
        if n is not None:
            correct = correct and n == n2
        if k is not None:
            correct = correct and k == k2
    if n is not None:
        correct = correct and n >= v1 + 1 + delta
    if k is not None:
        correct = correct and k >= j1 + 1 + delta
    return correct


def check_explicit_beliefs(X):
    """Verifies that a given [n x k] np.array of beliefs assigns each node to maximal one class with 100%.
    Thus, each row has maximum one 1-entry per row, and the rest are all 0s.
    Allows [ , ] in addition to [[ , ]] for n=1
    """
    assert type(X).__module__ == "numpy"        # make sure it is not a matrix
    correct = set(np.hstack(X)) == set([0, 1]), "Input X can contain only 0 or 1 (as int or float)"
    if len(X.shape) == 1:                       # special case in case n=1 and no nesting: [[...]] -> [...]
        return correct and np.abs(np.sum(X)) <= 1
    else:
        return correct and (np.abs(np.sum(X,1)) <= 1).all()


def to_centering_beliefs(X):
    """Centers an [n x k] np.array of valid normalized beliefs around 1/k
    Allows [ , ] in addition to [[ , ]] for n=1
    """
    # assert check_normalized_beliefs(X)    # statement would prevents using explicit belief vector
    assert type(X).__module__ == "numpy"    # make sure it is not a matrix
    if len(X.shape) == 1:                   # special case in case n=1 and no nesting: [[...]] -> [...]
        k = len(X)
    else:
        (_, k) = X.shape
    return X - 1./k


def from_centering_beliefs(Xc):
    """Uncenteres a centered [n x k] np.array
    Allows [ , ] in addition to [[ , ]] for n=1
    """
    assert check_centered_beliefs(Xc)
    if len(Xc.shape) == 1:           # special case in case n=1 and no nesting: [[...]] -> [...]
        k = len(Xc)
    else:
        (_, k) = Xc.shape
    return Xc + 1./k


def to_dictionary_beliefs(X):
    """Transforms explicit belief assignment as [n x k] matrix to dictionary (node id -> belief id)
    Assumes zeroindexing
    requires nested [[,]]
    http://stackoverflow.com/questions/29301899/iterate-over-nested-arrays-storing-indexes
    !!! what if several beliefs. Check for it
    """
    check_explicit_beliefs(X)
    assert len(X.shape) == 2        # requires nested [[,]]
    Xd = {i[0] : i[1] for i in product(*[range(dim) for dim in X.shape]) if X[i] > 0}
    return Xd


def from_dictionary_beliefs(Xd, n=None, k=None, zeroindexing=True):
    """Takes an explicit belief assignment from dictionary format (node id -> belief id) and transforms it into a [n x k] belief matrix.
    If n=None, assumes node (n-1) [or n for zeroindexing=False] appears.
    If k=None, assumes class (k-1) [or k for zeroindexing=False] appears.
    """
    check_dictionary_beliefs(Xd, n=n, k=k, zeroindexing=zeroindexing)

    keys = Xd.keys()
    values = Xd.values()
    if zeroindexing:
        delta = 0
    else:
        delta = -1

    if n is None:
        n = max(keys) + 1 + delta
    if k is None:
        k = max(values) + 1 + delta

    X = np.zeros((n, k), dtype=np.int)
    for k,v in Xd.items():
        X[k+delta,v+delta] = 1
    # for i in range(n):
    #     X[i,Xd[i]] = 1
    return X


def to_explicit_bool_vector(Xc):
    """Returns a n-dimensional Boolean np.array that indicates which nodes have some beliefs <> 0.
    Every row with sum(abs(entries)) > 0 is assumed to be explicit.
    Input: [n x k] np array
    Returns: Boolean n-dimensional np array
    """
    Xb = np.array(np.sum(np.abs(Xc), 1) > PREC).flatten()     # matrix -> ndarray, then flatten
    return Xb


def to_explicit_list(Xc):
    """Returns a list of indices of nodes with explicit beliefs
    Assumes a [n x k] np.array of normalized or explicit centered beliefs X with n > 1.
    every row with sum(abs(entries)) > 0 is assumed to be explicit
    """
    s = np.array(np.sum(np.abs(Xc), 1)).flatten()             # simple flat array (not matrix) of absolute row sums
    Xl = [i for i, j in enumerate(s) if j > PREC]
    return Xl


def max_binary_matrix(X, threshold=0):
    """Returns a [n x k] np.array binary integer matrix for top class for each node
    Assumes a [n x k] np.array matrix X; also allows a threshold value for declaring winner
    Makes sure that matrix has 2 dimensions
    """
    assert type(X).__module__ == "numpy"            # make sure it is not a matrix
    X = np.atleast_2d(X)                            # require 2d arrays
    X2 = X.max(1, keepdims=True) - threshold        # creates vertical [[],] vector of max per row
    X3 = 1*(X >= X2)                                # creates integer
    return X3


def row_normalize_matrix(M, norm='l1'):
    """Normalizes or standardizes each row of a two-dimensional array (not necessarily numpy array)
    Allows L2 or L1 norms ('l1' or 'l2'), or 'zscores' ("scaling").
    Allows rows with zero vectors.
    Serves as wrapper around sklearn.preprocessing.normalize and sklearn.preprocessing.scale functions.
    Fixes following issues:
        (1) allows int in addition to float
        (2) allows single rows as input
    """
    M = np.atleast_2d(M)    # both preprocessing.normalize and zscore require 2d arrays
    if M.dtype.kind != 'f':
        M = 1. * M          # M *= 1. does not work to replace dtype i with f

    if norm in ['l2', 'l1']:
        return normalize(M, axis=1, norm=norm)
    elif norm == 'zscores':
        return scale(M, axis=1, with_mean=True, with_std=True, copy=True)


def degree_matrix(W, indegree=True):
    """Calculates diagonal in- or outdegree degree matrices
    D_in: indegree=True (sum of squared col entries)
    D_out: indegree=False (sum of squared row entries)
    """
    n, _ = W.shape
    row, col = W.nonzero()         # transform the sparse W back to row col format
    weight = W.data
    weight2 = np.square(weight)
    W2 = csr_matrix((weight2, (row, col)), shape=(n, n))
    if indegree:
        # degree = [sum([weight[i]**2 for i in range(len(col)) if col[i] == index]) for index in range(n)]. super slow
        # degree = collections.Counter(col)   # count multiplicies of nodes classes. Slower even for degrees = 1
        degree = W2.transpose().dot(np.ones([n]))
    else:
        degree = W2.dot(np.ones([n]))
    return diags(degree, 0)


def matrix_difference(X, Y, ignore_rows=list([]), similarity='accuracy', vector=False):
    """Calculate difference (or similarity) between two [n x k] matrices X (ground truth) and Y (predicted).
    Difference is calculated row-wise (for each node separately).
    Allows to return a n-dimensional vector with row-wise differences or a single average over all rows.
    Optional argument specifies which rows should be ignored
    (e.g., because they were labeled with explicit beliefs, and we calulate accuracy for unlabeled nodes).
    Allows 'accuracy', 'precision', 'recall', 'l2' differences, 'cosine', 'cosine_ratio' similarities.
    For 'accuracy', 'precision', 'recall': compares for each row in GT and Method, the classes with top beliefs.

    Uses: max_binary_matrix()

    Parameters
    ----------
    X : [n x k] np array
        true belief matrix (GT)
    Y : [n x k] np array
        predicted belief matrix
    ignore_rows : int list, optional (Default=empty)
        list of rows to ignore [list of explicit beliefs that are not evaluated]
    similarity : what type of similarity function used, optional (Default='accuracy')
        similarity='accuracy' : "In multilabel classification, this function computes subset accuracy:
            the set of labels predicted for a sample must *exactly* match the
            corresponding set of labels in X." Copied from [sklearn.metrics.accuracy_score]
        similarity='precision' :
        similarity='recall' :
        similarity='cosine' : Compute cosine similarity between rows in [n x k] arrays X and Y.
            Returns a 1-d array with n entries, one for each row.
            Cosine similarity, or the cosine kernel, computes similarity as the
            normalized dot product of X and Y:
                K(X, Y) = <X, Y> / (||X||*||Y||)
            [Inspired by sklearn.metrics.pairwise.cosine_similarity, but returns simple vector instead of matrix]
        similarity='cosine_ratio' : Use cosine similarity as before, but then also multiply with ratio of lengths.
            In other words, this is the ratio of the projection of the smaller vector onto the larger vector.
            Solves the problem if both vectors point in same direction but are of different lengths (not yet converged)
        similarity='l2' : Compute L2 difference between rows in [n x k] arrays X and Y.
    vector : bool, optional (Default=False)
        if True, then returns the vector of individal accuracies per row instead of the average over all

    Returns
    -------
    accuracy :  float, or
                numpy.ndarray(floats)
    """
    # verify input
    if similarity not in ('accuracy', 'precision', 'recall', 'cosine','cosine_ratio', 'l2'):
        raise ValueError("'%s' is not a supported similarity function" % similarity)
    X = np.atleast_2d(X)    # needed for using the shape function below
    Y = np.atleast_2d(Y)
    n, k = X.shape
    n2, k2 = Y.shape
    assert(n == n2), "Matrices need to have the same dimensions"
    assert(k == k2), "Matrices need to have the same dimensions"

    # use list ignore_rows to ignore certain rows
    indTest = list(set(range(n)).difference(set(ignore_rows)))   # evaluate accuracy only on implicit beliefs
    X = np.asarray(X[indTest])
    Y = np.asarray(Y[indTest])

    if similarity in ['cosine', 'cosine_ratio']:
        Xn = row_normalize_matrix(X, norm='l2')
        Yn = row_normalize_matrix(Y, norm='l2')
        Z = np.array([np.dot(Xn[i, :], Yn[i, :]) for i in range(Xn.shape[0])])
            # new n = first entry of shape (after ignoring explicit belief rows)
        if similarity == 'cosine_ratio':
            Xl = np.linalg.norm(X, axis=1)  # calculate length of each row-vectors
            Yl = np.linalg.norm(Y, axis=1)  # calculate length of each row-vectors
            Zl = np.array([i/j if i<j else j/i if j<i else 1 for (i,j) in zip(Xl,Yl)])
            Z = Z*Zl
    elif similarity == 'l2':
        Z = np.linalg.norm(X-Y, axis=1)
    else:
        THRESHOLD = 1e-8  # threshold for binarization
        Xb = max_binary_matrix(X, threshold=THRESHOLD)   # binary matrices indicating one or more top beliefs per row
        Yb = max_binary_matrix(Y, threshold=THRESHOLD)
        Zb = Xb * Yb  # overlap

        x = Xb.sum(axis=1)  # sum up the rows of binary matrices
        y = Yb.sum(axis=1)
        z = Zb.sum(axis=1)

        if similarity == 'precision':
            Z = 1. * z / y
        elif similarity == 'recall':
            Z = 1. * z / x
        elif similarity == 'accuracy':
            Z = 1. * ( (Xb != Yb).sum(axis=1) == 0 )

    if vector:
        return Z
    else:
        return np.average(Z)


def matrix_convergence_percentage(X0, X1, threshold=0.9962, ignore_rows=list([])):
    """Takes two 2d numpy array and returns the percentage of rows that have cosine similarity > threshold.
    (In other words, the percentage of beliefs that have converged).
    Optional argument specifies which rows should be ignored
    (because they were labeled with explicit beliefs, and we calulate accuracy for unlabeled nodes).
    Two rows with only zeros are assumed to be perpendicular
    (thus nodes without any beliefs in early iterations are assumed not to have yet converged;
    an important assumption is that every connected component has at least one node with explicit beliefs).
    It is recommended to first standardize the input matrices before using this function.

    Uses: matrix_difference()

    Parameters
    ----------
    X0 : [n x k] np array
        first matrix
    Y1 : [n x k] np array
        second matrix
    ignore_rows : int list, optional (Default=empty)
        list of rows to ignore [list of explicit beliefs that are not evaluated]
    threshold : float (Default = 0.9962)
        threshold cosine similarity between rows in [n x k] arrays X and Y.

    Returns
    -------
    result_fractionce :  float
        Fraction of rows with cosine similarity (between two input matrices) > threshold
    """
    result_vector = matrix_difference(X0, X1, similarity='cosine_ratio', vector=True, ignore_rows=ignore_rows)
    result_fraction = sum(1*(result_vector > threshold)) / len(result_vector)
    # print result_vector                             # used for debugging
    # print np.degrees(np.arccos(result_vector))      # used for debugging
    return result_fraction


def replace_fraction_of_rows(X0, f, avoidNeighbors=False):
    """Given [n x k] matrix. Replace a random fraction f of row with 0-vector (i.e. replace exactly f*n rows).
    Returns new [n x k] array, plus indices of remaining explicit beliefs (i.e. (1-f)*n of the remaining rows).
    """
    # TODO: create variant that leaves no neighbors connected
    n, _ = X0.shape
    r = int(round(f*n))
    if not avoidNeighbors:
        ind = np.random.choice(n, r, replace=False)         # index of replaced rows
        X = np.array(X0)
        X[ind, :] = 0
        indGT = list(set(range(n)).difference(set(ind)))    # index of unchanged rows
        indGT.sort()
        return X, indGT
    else:
        None
        # TODO


def eps_convergence_linbp(Hc, W, echo=False, rho_W=None):
    """Calculates eps_convergence with which to multiply Hc, for LinBP (with or w/o echo) to be at convergence boundary.
    Returns 0 if the values HC are too small (std < SPECTRAL_TOLERANCE)
    Assumes symmetric W and symmetric and doubly stochastic Hc

    Uses: degree_matrix()

    Parameters
    ----------
    Hc : np.array
        Centered coupling matrix (symmetric, doubly stochastic)
    W : sparse matrix
        Sparse edge matrix (symmetric)
    echo : boolean
        True to include the echo cancellation term
    rho_W : float
        the spectral radius of W as optional input to speed up the calculations
    """
    if np.std(Hc) < SPECTRAL_TOLERANCE:
        return 0
    else:
        if rho_W is None:
            rho_W = approximate_spectral_radius(csr_matrix(W, dtype='f'))    # needs to transform from int
        rho_H = approximate_spectral_radius(np.array(Hc, dtype='f'))                # same here
        eps = 1. / rho_W / rho_H

        # if echo is used, then the above eps value is used as starting point
        if echo:
            Hc2 = Hc.dot(Hc)
            D = degree_matrix(W)

            # function for which we need to determine the root: spectral radius minus 1
            def radius(eps):
                return approximate_spectral_radius( kron(Hc, W).dot(eps) - kron(Hc2, D).dot(eps**2) ) - 1

            # http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html#scipy.optimize.newton
            eps2 = newton(radius, eps, tol=1e-04, maxiter=100)
            eps = eps2

        return eps
