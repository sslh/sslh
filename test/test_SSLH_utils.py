"""
Test class for utility functions SSL-H

First version: Dec 6, 2014
This version: June 1, 2015
Author: Wolfgang Gatterbauer <gatt@cmu.edu>
"""


from SSLH_utils import (check_normalized_beliefs, check_centered_beliefs,
                        to_centering_beliefs, from_centering_beliefs,
                        check_dictionary_beliefs, from_dictionary_beliefs,
                        check_explicit_beliefs, to_dictionary_beliefs,
                        to_explicit_bool_vector, to_explicit_list,
                        matrix_difference, max_binary_matrix,
                        row_normalize_matrix,
                        replace_fraction_of_rows,
                        matrix_convergence_percentage,
                        eps_convergence_linbp, degree_matrix)
from SSLH_files import load_X, load_W, load_H
import numpy as np
import scipy.sparse as sps
import time


# -- Determine path to data irrespective (!) of where the file is run from
from os.path import abspath, dirname, join
from inspect import getfile, currentframe
current_path = dirname(abspath(getfile(currentframe())))
data_directory = join(current_path, 'data/')
fig_directory = join(current_path, 'figs/')


def test_transform_beliefs():
    print "\n-- 'check_normalized_beliefs', 'to_centering_beliefs' --"
    X = np.array([[1.0001, 0, 0]])
    print "X: ", X
    assert check_normalized_beliefs(X)
    print "X centered: ", to_centering_beliefs(X)

    Y = np.array([0.9999, 0, 0])
    print "Y: ", Y
    assert check_normalized_beliefs(Y)
    print "Y centered: ", to_centering_beliefs(Y)

    Z = np.array([[1.001, 0, 0]])
    print "Z: ", Z
    assert not check_normalized_beliefs(Z)

    W = np.array([0.999, 0, 0])
    print "W: ", W
    assert not check_normalized_beliefs(W)

    print "\n-- 'check_centered_beliefs', 'from_centering_beliefs'"
    Xc = np.array([[1.0001, -1, 0]])
    print "Xc: ", Xc
    assert check_centered_beliefs(Xc)
    print "Xc uncentered: ", from_centering_beliefs(Xc)

    Yc = np.array([0.9999, -1, 0])
    print "Yc: ", Yc
    assert check_centered_beliefs(Yc)
    print "Yc uncentered: ", from_centering_beliefs(Yc)

    Zc = np.array([[1.001, -1, 0]])
    print "Zc: ", Zc
    assert not check_centered_beliefs(Zc)

    Wc = np.array([0.999, -1, 0])
    print "Wc: ", Wc
    assert not check_centered_beliefs(Wc)


    print "\n--'to_centering_beliefs', 'from_centering_beliefs' for matrices --"
    X = np.array(   [[1,0,0],
                     [0.8,0.2,0],
                     [1./3,1./3,1./3],
                     [0,0,1],
                     [0,0,1],
                     [0.5,0,0.5]])
    print "X original:\n", X
    print "np.sum(X,1):\n", np.sum(X,1)
    print "X.sum(axis=1, keepdims=True):\n", X.sum(axis=1, keepdims=True)
    print "X.shape:", X.shape
    print "len(X.shape): ", len(X.shape)

    Xc = to_centering_beliefs(X)
    print "X centered:\n", Xc
    Y = from_centering_beliefs(Xc)
    print "X again un-centered:\n", Y

    fileNameX = join(data_directory, 'Torus_X.csv')
    X, _, _ = load_X(fileNameX, n=8, zeroindexing=False)
    X = X.dot(0.1)
    print "\nCentered X for Torus example as input\n", X
    Xc = from_centering_beliefs(X)
    print "X un-centered:\n", Xc


    X = np.array(   [[1,0,0]])
    print "\nX original:\n", X
    Xc = to_centering_beliefs(X)
    print "X centered:\n", Xc
    Y = from_centering_beliefs(Xc)
    print "X back non-centered:\n", Y

    X = np.array(   [1,0,0])
    print "\nX original:\n", X
    print "np.sum(X,0):", np.sum(X,0)
    print "X.sum(axis=0, keepdims=True):", X.sum(axis=0, keepdims=True)
    print "X.shape: ", X.shape
    print "len(X.shape): ", len(X.shape)


def test_dictionary_transform():
    print "\n-- 'check_dictionary_beliefs', 'from_dictionary_beliefs' --"
    Xd = {1: 1, 2: 2, 3: 3, 5: 1}
    print "Xd:", Xd

    print "zeroindexing=True:"
    print "X:\n", from_dictionary_beliefs(Xd, n=None, k=None, zeroindexing=True)
    print "zeroindexing=False:"
    print "X:\n", from_dictionary_beliefs(Xd, n=None, k=None, zeroindexing=False)
    print "zeroindexing=True, n=7, k=5:"
    print "X:\n", from_dictionary_beliefs(Xd, n=7, k=5, zeroindexing=True)

    print "\nzeroindexing=False, fullBeliefs=True:"
    X1 = {1: 1, 2: 2, 3: 3, 4: 1}
    assert check_dictionary_beliefs(X1, n=None, k=None, zeroindexing=False, fullBeliefs=True)
    print "X1:", X1

    print "zeroindexing=True, fullBeliefs=True:"
    X2 = {0: 0, 1: 1, 2: 2, 3: 0}
    assert check_dictionary_beliefs(X2, n=None, k=None, zeroindexing=True, fullBeliefs=True)
    print "X2:", X2

    print "zeroindexing=True, fullBeliefs=False:"
    X3 = {0: 0, 1: 1, 2: 2, 4: 0}
    assert check_dictionary_beliefs(X3, n=None, k=None, zeroindexing=True, fullBeliefs=False)
    print "X3:", X3

    print "zeroindexing=True, fullBeliefs=False:"
    X4 = {0: 1, 2: 2, 4: 0}
    assert check_dictionary_beliefs(X4, n=None, k=None, zeroindexing=True, fullBeliefs=False)
    print "X4:", X4

    print "\nerrors:"
    X5 = {0: 0, 1: 1, 2: 3, 3: 0}
    print "X5:", X5
    assert not check_dictionary_beliefs(X5, n=None, k=None, zeroindexing=False, fullBeliefs=True)
    X6 = {0: 1, 1: 1, 2: 2, 4: 1}
    print "X6:", X6
    assert not check_dictionary_beliefs(X6, n=None, k=None, zeroindexing=True, fullBeliefs=True)


    print "\n-- 'check_explicit_beliefs', 'to_dictionary_beliefs' --"
    X = np.array(  [[1., 0, 0],
                    [0, 0, 0],
                    [0, 0, 1],
                    ])
    print "original X:\n", X
    print "Stacked X:\n", np.hstack(X)
    print "List of X entires:\n", set(np.hstack(X))
    print "Verify: ", set(np.hstack(X)) == set([0, 1])
    print "Verify: ", set([0., 1.]) == set([0, 1])
    assert check_explicit_beliefs(X)
    Y = np.array([1., 0, 0])
    assert check_explicit_beliefs(Y)
    Xd = to_dictionary_beliefs(X)
    print "Xd: ", Xd


def test_to_explicit_vectors():
    print "\n-- 'to_explicit_bool_vector', 'to_explicit_list' --"
    fileNameX = join(data_directory, 'Torus_X.csv')
    X, _, _ = load_X(fileNameX, n=8, zeroindexing=False)
    print "Torus X:\n", X
    Xb = to_explicit_bool_vector(X)
    print 'Xb:\n', Xb
    Xl = to_explicit_list(X)
    print 'Xl:\n', Xl

    Y = np.array(   [[0,0,0],
                     [0,0,1],
                     [0,1,1],
                     [0,0,-1],
                     [0,0,0.0001],
                     [0,0,0.001]])
    print "\nY:\n", Y
    Yb = to_explicit_bool_vector(Y)
    print 'Yb:\n', Yb
    Yl = to_explicit_list(Y)
    print 'Yl:\n', Yl


def test_max_binary_matrix():
    print "\n-- 'max_binary_matrix' --"
    X = np.array(   [[1,0,0],
                     [10,8,5],
                     [1./3,1./3,1./3],
                     [0,0,1],
                     [0,0.9,1],
                     [0.5,0,0.5]])
    print "X original:\n", X
    Xb = max_binary_matrix(X)
    print "X with winning classes (no tolerance):\n", Xb
    Xb = max_binary_matrix(X, 0.2)
    print "X with winning classes (with 0.2 tolerance):\n", Xb

    X = np.array(   [[10,9,0]])
    print "\nX original:\n", X
    Xb = max_binary_matrix(X,2)
    print "X with winning classes (with 2 tolerance):\n", Xb


def test_row_normalize_matrix():
    print "\n-- 'row_normalize_matrix' (l1, l2, zscores) --"
    v = np.array([1, 1, 0, 0, 0])
    print "original:\n ", v
    print "l2:\n ", row_normalize_matrix(v, norm='l2')
    print "l1:\n ", row_normalize_matrix(v, norm='l1')
    print "zscores:\n ", row_normalize_matrix(v, norm='zscores')

    v = np.array([1, 1, 1, 0, 0])
    print "\noriginal:\n ", v
    print "l2:\n ", row_normalize_matrix(v, norm='l2')
    print "l1 :\n ", row_normalize_matrix(v, norm='l1')
    print "zscores:\n ", row_normalize_matrix(v, norm='zscores')

    X = np.array(  [[1, 0, 0],
                    [0, 0, 0],
                    [1, -1, -1],
                    [1, -1, -1.1],
                    [1, -2, -3],])
    print "\noriginal:\n", X
    print "l2:\n", row_normalize_matrix(X, norm='l2')
    print "!!! Notice that l1 norm with negative values is counterintuitive: !!!"
    print "l1:\n", row_normalize_matrix(X, norm='l1')
    print "zscores:\n", row_normalize_matrix(X, norm='zscores')

    X = np.array([[0, 20, 0],
                  [21, 0, 0],
                  [0, 0, 14]])
    print "\noriginal:\n", X
    print "l2:\n", row_normalize_matrix(X, norm='l2')
    print "l1:\n", row_normalize_matrix(X, norm='l1')
    print "zscores:\n", row_normalize_matrix(X, norm='zscores')


    print "\n -- zscore and normalizing together --"
    v = np.array([1, 1, 0, 0, 0])
    print "original:\n  ", v
    print "zscore:\n  ", row_normalize_matrix(v, norm='zscores')
    print "normalized zscore:\n  ", \
        row_normalize_matrix(
            row_normalize_matrix(v, norm='zscores'), norm='l2')
    print "normalized zscore normalized:\n  ", \
        row_normalize_matrix(
            row_normalize_matrix(
                row_normalize_matrix(v,norm='l2'), norm='zscores'), norm='l2')

    X = np.array(  [[1, 0, 0],
                    [1, -1, -1],
                    [1, -1, -1.1],
                    [1, -2, -3],
                    [0, 0, 0],
                    [1,1,-1],
                    [1,1.1,-1],
                    [1,1,1]])
    print "\noriginal:\n", X
    print "zscore:\n", row_normalize_matrix(X, norm='zscores')
    print "normalized:\n", row_normalize_matrix(X, norm='l2')
    print "normalized zscore:\n", \
        row_normalize_matrix(
            row_normalize_matrix(X,norm='zscores'), norm='l2')
    print "normalized zscore normalized:\n", \
        row_normalize_matrix(
            row_normalize_matrix(
                row_normalize_matrix(X,norm='l2'),norm='zscores'),norm='l2')
    print "zscore normalized zscore normalized:\n", \
        row_normalize_matrix(
            row_normalize_matrix(
                row_normalize_matrix(
                    row_normalize_matrix(X,norm='l2'),norm='zscores'),norm='l2'),norm='zscores')


def test_degree_matrix():
    print "\n-- 'degree_matrix' --"
    row = [0, 0, 0, 1, 2, 3]
    col = [1, 2, 3, 4, 4, 4]
    weight = [2, 3, 4, 1, 2, 3]
    W = sps.csr_matrix((weight, (row, col)), shape=(5, 5))
    print "Dense:\n", W.todense()
    D_in = degree_matrix(W, indegree=True)
    D_out = degree_matrix(W, indegree=False)
    print "D_in (col sum):\n", D_in.todense()
    print "D_out (row sum):\n", D_out.todense()

    print "\nTest with big random matrix"
    n = 100000
    d = 10
    row = np.random.randint(n, size=n*d)
    col = np.random.randint(n, size=n*d)
    weight = np.random.randint(1, 10, size=n*d)
    W = sps.csr_matrix((weight, (row, col)), shape=(n, n))

    # -- optionally replace all degrees by 1
    row, col = W.nonzero()
    weight = [1]*len(row)
    W = sps.csr_matrix((weight, (row, col)), shape=(n, n))

    start = time.time()
    D_in = degree_matrix(W, indegree=True)
    end = time.time()-start
    print "Time:", end


def test_matrix_difference_with_cosine_simililarity():
    print "\n-- 'matrix_difference' (cosine), 'row_normalize_matrix' --"
    print "k=3"
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([1, 1, 0])
    print "Cosine with original:\n  ", \
        matrix_difference(v1,
                          v1, similarity='cosine')
    print "Cosine with original zscore:\n  ", \
        matrix_difference(row_normalize_matrix(v1, norm='zscores'),
                          row_normalize_matrix(v1, norm='zscores'), similarity='cosine')
    print "Cosine with zscore :\n  ", \
        matrix_difference(v1,
                          row_normalize_matrix(v1, norm='zscores'), similarity='cosine')
    print "Cosine with normal:\n  ", \
        matrix_difference(v1,
                          v2, similarity='cosine')
    print "Cosine with normal after both zscore:\n  ", \
        matrix_difference(row_normalize_matrix(v1, norm='zscores'),
                          row_normalize_matrix(v2, norm='zscores'), similarity='cosine')
    print "! Notice that average guessing leads to expectation of 0!"
    print "Cosine v1, v3:\n  ", \
        matrix_difference(v1,
                          v3, similarity='cosine')
    print "Cosine v1, v3 after zscore:\n  ", \
        matrix_difference(row_normalize_matrix(v1, norm='zscores'),
                          row_normalize_matrix(v3, norm='zscores'), similarity='cosine')

    print "\nk=5"
    v1 = np.array([1, 0, 0, 0, 0])
    v2 = np.array([0, 1, 0, 0, 0])
    v3 = np.array([1, 1, 0, 0, 0])
    v4 = np.array([0, 0, 0, 0, 0])
    print "Cosine with normal:\n  ", \
        matrix_difference(v1,
                          v2, similarity='cosine')
    print "Cosine with normal after both zscore:\n  ", \
        matrix_difference(row_normalize_matrix(v1, norm='zscores'),
                          row_normalize_matrix(v2, norm='zscores'), similarity='cosine')
    print "! Notice that average guessing leads to expectation of 0!"
    print "Cosine v1, v3:\n  ", \
        matrix_difference(v1,
                          v3, similarity='cosine')
    print "Cosine v1, v3 after zscore:\n  ", \
        matrix_difference(row_normalize_matrix(v1, norm='zscores'),
                          row_normalize_matrix(v3, norm='zscores'), similarity='cosine')
    print "Average Cos similarity partly zscore:\n  ", \
        matrix_difference(row_normalize_matrix(v1, norm='zscores'),
                          row_normalize_matrix(v3, norm='zscores'), similarity='cosine')
    print "Cosine with 0-vector:\n  ", \
        matrix_difference(row_normalize_matrix(v1, norm='zscores'),
                          row_normalize_matrix(v4, norm='zscores'), similarity='cosine')
    print

    X = np.array([[1, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0]])
    Y = np.array([[1, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [1, 1.1, 0, 0, 0]])
    print "X\n", X
    print "Y\n", Y
    Xs = row_normalize_matrix(X, norm='zscores')
    Ys = row_normalize_matrix(Y, norm='zscores')
    print "Xs\n", Xs
    print "Ys\n", Ys

    print "\nCosine original:\n  ", \
        matrix_difference(X,
                          Y, vector=True, similarity='cosine')
    print "Cosine zscore:\n  ", \
        matrix_difference(Xs,
                          Ys, vector=True, similarity='cosine')
    print "Average cosine zscore:\n  ", \
        matrix_difference(X,
                          Y, similarity='cosine')


def test_matrix_difference_with_accuracy_etc():
    print "\n-- 'matrix_difference' (precision/recall/accuracy/cosine), 'max_binary_matrix' --"
    X_true = np.array([[2, 0, 0],
                       [2, 0, 2],
                       [0, 1, 0],
                       [0, 0, 3],
                       [0, 0, 3],
                       [1, 0, 2],
                       [0, 3, 3]])
    X_pred = np.array([[1, 1, 2],
                       [2, 1, 2],
                       [3, 4, 0],
                       [1, 1, 2],
                       [2, 1, 1],
                       [1, 2, 2],
                       [1, 2.99, 3]])
    X_true_b = max_binary_matrix(X_true)
    X_pred_b = max_binary_matrix(X_pred)
    X_pred_b1 = max_binary_matrix(X_pred, threshold=0.1)
    print "X_true:\n", X_true
    print "X_pred:\n", X_pred
    print "X_true binary:\n", X_true_b
    print "X_pred binary:\n", X_pred_b
    print "X_pred binary with threshold 0.1:\n", X_pred_b1

    ind = list([])
    # ind = list([0, 1])
    # ind = list([1, 2, 3, 4, 5])
    # ind = list([0, 2, 3, 4, 5, 6])
    print "\nPrecision:\n", matrix_difference(X_true, X_pred, ind, vector=True, similarity='precision')

    print "*** type:", type (matrix_difference(X_true, X_pred, ind, vector=True, similarity='precision'))

    print "Recall:\n", matrix_difference(X_true, X_pred, ind, vector=True, similarity='recall')
    print "Accuracy:\n", matrix_difference(X_true, X_pred, ind, vector=True, similarity='accuracy')
    cosine_list = matrix_difference(X_true, X_pred, ind, vector=True, similarity='cosine')
    print "Cosine:\n", cosine_list
    print "Cosine sorted:\n", sorted(cosine_list, reverse=True)

    print "\nPrecision:\n", matrix_difference(X_true, X_pred, ind, similarity='precision')
    print "Recall:\n", matrix_difference(X_true, X_pred, ind, similarity='recall')
    print "Accuracy:\n", matrix_difference(X_true, X_pred, ind)
    print "Cosine:\n", matrix_difference(X_true, X_pred, ind, similarity='cosine')


def test_matrix_difference():
    print "\n-- 'matrix_difference' (cosine/cosine_ratio/l2), 'to_centering_beliefs' --"
    X0 = np.array([[2, 0, 0],
                   [2, 0, 2],
                   [0, 1, 0],
                   [0, 0, 3],
                   [0, 0, 3],
                   [1, 0, 2],
                   [0, 3, 3],
                   [0, 0, 0],
                   [0, 1, 0],
                   [0, 1, 0],
                   [9, 9, 9],
                   [9, 9, 9],
                   [100, 100, 100],])
    X1 = np.array([[1, 1, 2],
                   [2, 1, 2],
                   [3, 4, 0],
                   [1, 1, 2],
                   [2, 1, 1],
                   [1, 2, 2],
                   [1, 2, 3],
                   [0, 0, 0],
                   [1, 0, 0],
                   [0, 2, 0],
                   [9, 9, 9],
                   [8, 9, 9],
                   [100, 100, 101],])
    print "X0:\n", X0
    print "X1:\n", X1

    result = matrix_difference(X0, X1, similarity='cosine', vector=True)
    print "cosine:\n", result
    result = matrix_difference(X0, X1, similarity='cosine_ratio', vector=True)
    print "cosine_ratio:\n", result
    result = matrix_difference(X0, X1, similarity='l2', vector=True)
    print "l2:\n", result

    X0 = np.array([[ 1.       ,   0.       ,   0.        ],
                  [ 0.30804075,  0.56206462,  0.12989463],
                  [ 0.32434628,  0.33782686,  0.33782686],
                  [ 0.30804075,  0.12989463,  0.56206462],
                  [ 0.14009173,  0.71981654,  0.14009173],
                  [ 0.32273419,  0.21860539,  0.45866042],
                  [ 0.33804084,  0.32391832,  0.33804084],
                  [ 0.45866042,  0.21860539,  0.32273419]])
    X1 = np.array([[ 1.      ,    0.      ,    0.        ],
                  [ 0.22382029,  0.45296374,  0.32321597],
                  [ 0.32434628,  0.33782686,  0.33782686],
                  [ 0.22382029,  0.32321597,  0.45296374],
                  [ 0.2466463 ,  0.5067074 ,  0.2466463 ],
                  [ 0.32273419,  0.21860539,  0.45866042],
                  [ 0.33804084,  0.32391832,  0.33804084],
                  [ 0.45866042,  0.21860539,  0.32273419]])
    print "\nX0:\n", X0
    print "X1:\n", X1

    result = matrix_difference(X0, X1, similarity='cosine_ratio', vector=True)
    print "cosine:\n", result

    # X0z = row_normalize_matrix(X0, norm='zscores')
    # X1z = row_normalize_matrix(X1, norm='zscores')
    X0z = to_centering_beliefs(X0)
    X1z = to_centering_beliefs(X1)

    print "\nX0z:\n", X0z
    print "X1z:\n", X1z

    result = matrix_difference(X0z, X1z, similarity='cosine_ratio', vector=True)
    print "cosine zscores:\n", result

    # actualPercentageConverged = matrix_convergence_percentage(X0z, X1z, threshold=convergenceCosineSimilarity)

    X0 = np.array([1, 0, 0])
    X1 = np.array([1, 1, 0])
    print "\nX0:\n", X0
    print "X1:\n", X1
    result = matrix_difference(X0, X1, similarity='cosine_ratio', vector=True)
    print "cosine zscores:\n", result


def test_matrix_convergence_percentage():
    print "\n-- 'matrix_convergence_percentage' --"
    X0 = np.array([[2, 0, 0],
                   [2, 0, 2],
                   [0, 1, 0],
                   [0, 0, 3],
                   [0, 0, 3],
                   [1, 0, 2],
                   [0, 3, 3],
                   [0, 0, 0],
                   [9, 9, 9],
                   [100, 100, 100],])
    X1 = np.array([[1, 1, 2],
                   [2, 1, 2],
                   [3, 4, 0],
                   [1, 1, 2],
                   [2, 1, 1],
                   [1, 2, 2],
                   [1, 2, 3],
                   [0, 0, 0],
                   [8, 9, 9],
                   [100, 100, 101],])
    print "X0:\n", X0
    print "X1:\n", X1

    threshold = 0.5
    percentage = matrix_convergence_percentage(X0, X1, threshold)
    print "percentage converged (original):\n", percentage

    X0z = row_normalize_matrix(X0, norm='zscores')
    X1z = row_normalize_matrix(X1, norm='zscores')
    percentage = matrix_convergence_percentage(X0z, X1z, threshold)
    print "percentage converged (after zscore):\n", percentage


def test_replace_fraction_of_rows():
    print "\n-- 'replace_fraction_of_rows' --"
    In = np.array([[2, 0, 0],
                   [2, 0, 2],
                   [0, 1, 0],
                   [0, 0, 3],
                   [0, 0, 3],
                   [1, 0, 2]])
    f = 0.7
    Out, ind = replace_fraction_of_rows(In, f)
    print 'In:\n', In
    print 'f =', f
    print 'Out:\n', Out
    print 'ind of remaining 1-f=30% rows:\n', ind


def test_eps_convergence_linbp_Torus():
    print "\n-- 'eps_convergence_linbp' for Torus ---"
    W, n = load_W(join(data_directory, 'Torus_W.csv'), zeroindexing=False)
    print 'W dense:\n', W.todense()
    print 'W:\n', W
    Hc, k, _ = load_H(join(data_directory, 'Torus_H.csv'), zeroindexing=False)
    print "H\n", Hc
    print

    # Simple spectral = 0.658
    start = time.time()
    eps = eps_convergence_linbp(Hc, W)
    end = time.time()-start
    print "Eps:", eps
    print "Time needed:", end

    # Echo spectral = 0.488
    start = time.time()
    eps = eps_convergence_linbp(Hc, W, echo=True)
    end = time.time()-start
    print "Eps:", eps
    print "Time needed:", end


if __name__ == '__main__':
    test_transform_beliefs()
    test_dictionary_transform()
    test_to_explicit_vectors()
    test_max_binary_matrix()
    test_row_normalize_matrix()
    test_degree_matrix()
    test_matrix_difference_with_cosine_simililarity()
    test_matrix_difference_with_accuracy_etc()
    test_matrix_difference()
    test_matrix_convergence_percentage()
    test_replace_fraction_of_rows()
    test_eps_convergence_linbp_Torus()
