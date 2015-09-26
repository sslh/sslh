"""Test class for 'SSLH_files' (file manipulation for SSL-H)

First version: Dec 6, 2014
This version: Sept 25, 2015
Author: Wolfgang Gatterbauer <gatt@cmu.edu>
"""


import numpy as np
import pytest
from SSLH_files import (load_W, save_W,
                        load_X, save_X,
                        load_Xd, save_Xd,
                        load_H, save_H,
                        load_csv_records, save_csv_records)


# -- Determine path to data irrespective (!) of where the file is run from
# TODO: how to deal with case when the data reside in a parent folder
from os.path import abspath, dirname, join
from inspect import getfile, currentframe
current_path = dirname(abspath(getfile(currentframe())))
data_directory = join(current_path, 'data/')


def test_load_and_save_W():

    print "\n-- 'load_W', 'save_W'"
    print "Example where file does not exist:"
    filename1 = 'notExistingFile.csv'
    with pytest.raises(IOError):
        W, n = load_W(join(data_directory, filename1))     # files still without zero indexing

    print "\nTorus, zeroindexing=False"
    filename2 = 'Torus_W.csv'
    print "File directory:\n  ", join(data_directory, filename2)
    W, n = load_W(join(data_directory, filename2), zeroindexing=False)
    print "W:\n", W
    print "Dense:\n", W.todense()
    print "Shape: ", W.shape

    print "\nTorus, n=9"
    W, n = load_W(join(data_directory, filename2), zeroindexing=False, n=9)
    print "W:\n", W
    print "Dense:\n", W.todense()
    print "Shape: ", W.shape

    print "\nSave same data with n=9 and load without specifying n=9 during load"
    filename3 = 'Torus_W3.gz'
    save_W(join(data_directory, filename3), W, saveWeights=True)      # gzip is possible
    W, n = load_W(join(data_directory, filename3))                    # newly written files use zeroindexing=True
    print "Dense:\n", W.todense()
    print "Shape: ", W.shape

    print "\nLoad data with float"
    W, n = load_W(join(data_directory, 'Torus_W3.gz'), usefloat=True)     # use float for weights other than integers
    print "W:\n", W
    print "Shape: ", W.shape

    print "\nSave data with float works, load without specifying float"
    filename4 = 'Torus_W4.csv'
    save_W(join(data_directory, filename4), W, saveWeights=True)
    W, n = load_W(join(data_directory, filename4), usefloat=True)
    print "W:\n", W

    print "\n2 columns, directed, space as delimiter"
    filename5 = 'test_load_and_save_W.txt'
    W, n = load_W(join(data_directory, filename5), delimiter=None, zeroindexing=False)
    print "W\n", W

    print "\n2 columns, directed, space as delimiter, doubleUndirected"
    W, n = load_W(join(data_directory, filename5), delimiter=None, zeroindexing=False, doubleUndirected=True)
    print "W\n", W

    print "\n3 columns, directed, space as delimiter, doubleUndirected"
    W, n = load_W(join(data_directory, 'Torus_W6.csv'), usefloat=True, doubleUndirected=True)
    print "W\n", W

    print "\nSave the same without weights, and load again"
    filename7 = 'Torus_W7.csv'
    save_W(join(data_directory, filename7), W, delimiter=' ')
    W, n = load_W(join(data_directory, filename7), delimiter=None)
    print "W\n", W


def test_load_and_save_X():
    print "\n-- 'load_X', 'save_X'"
    print "Load matrix and specify n=8"
    # filename1 = 'data/Torus_X.csv'
    X, _, _ = load_X(join(data_directory, 'Torus_X.csv'), n=8, zeroindexing=False)
    print "X:\n", X
    print "Shape:", X.shape

    print "\nSave and load matrix with n=8 k=3"
    filename2 = 'Torus_X2.csv'
    save_X(join(data_directory, filename2), X)
    X, _, _ = load_X(join(data_directory, filename2), n=8, k=3)
    print "X:\n", X

    print "\nSave and load matrix with n=8 k=3, delimiter= ' ', format gz"
    filename3 = 'Torus_X3.gz'
    save_X(join(data_directory, filename3), X, delimiter=' ')
    X, _, _ = load_X(join(data_directory, filename3), n=8, k=3, delimiter=None)
    print "X:\n", X

    print "\nInput contains float"
    X, _, _ = load_X(join(data_directory, 'Torus_X4.csv'), n=8, k=3)
    print "X:\n", X

    print "\nInput contains only 1 (explicit beliefs)"
    X, _, _ = load_X(join(data_directory, 'Torus_X5.csv'), n=8, k=3)
    print "X:\n", X

    print "\nInput contains only two columns. Correct"
    X, _, _ = load_X(join(data_directory, 'Torus_X_twoColumns.csv'), n=8, k=3, delimiter=None)
    print "X:\n", X

    print "\nInput contains only two columns. With assert error (node with several classes)"
    try:
        X, _, _ = load_X(join(data_directory, 'Torus_X_twoColumns_assertError.csv'), n=8, k=3, delimiter=None)
    except AssertionError, e:
        print "! Assertion error:\n", e


def test_load_and_save_Xd():
    print "\n-- 'save_Xd', 'load_Xd"
    filename = 'test_load_and_save_Xd.csv'
    Xd = {0: 4, 1: 5, 8: -1}
    print "Original Xd:\n", Xd
    save_Xd(join(data_directory, filename), Xd)

    Xd = load_Xd(join(data_directory, filename))
    print "Loaded Xd:\n", Xd


def test_load_and_save_H():
    print "\n-- 'load_H', 'save_H'"
    H, _, _ = load_H(join(data_directory, 'Torus_H.csv'), zeroindexing=False)
    print "Loaded H: \n", H
    print "Shape: ", H.shape

    filename2 = 'Torus_H2.csv'
    print "Saving and loading can lead to different precision"
    save_H(join(data_directory, filename2), H)
    H, _, _ = load_H(join(data_directory, filename2))
    print H
    print "Shape: ", H.shape
    save_H(join(data_directory, 'Torus_H3.csv'), H)

    print "\nLoading always leads to float"
    H = np.array([[1, 2, 3],
                  [4, 5, 6]])        # change 6 to 6.5
    print "Original H:\n", H
    print "Shape: ", H.shape
    filename3 = 'test_load_and_save_H.csv'
    save_H(join(data_directory, filename3), H, delimiter=' ')
    H, _, _ = load_H(join(data_directory, filename3), delimiter=None)
    print "Loaded H:\n", H
    print "Shape: ", H.shape


def test_load_and_save_csv_records():
    print "\n-- 'load_csv_records', 'save_csv_records'"
    print "Can recover int and float from strings"
    records = [('a', 1, 1.1), ('b', 2, 2.2)]
    filename = 'test_load_and_save_csv_records.csv'
    save_csv_records(join(data_directory, filename), records)
    records2 = load_csv_records(join(data_directory, filename))
    print "Loaded record from csv:\n", records2


if __name__ == '__main__':
    test_load_and_save_W()
    test_load_and_save_X()
    test_load_and_save_Xd()
    test_load_and_save_H()
    test_load_and_save_csv_records()
