"""
File manipulation for SSL-H: Loads from and saves to csv formats.
Nomenclature: W, X, H[j,i], n, l, k, Xd, Xr (relational)

First version: Dec 6, 2014
This version: Sept 16, 2015
Author: Wolfgang Gatterbauer <gatt@cmu.edu>
"""


import numpy as np
from scipy.sparse import csr_matrix
import csv


def load_W(fileName, skiprows=1, zeroindexing=True, n=None, delimiter=',', usefloat=False, doubleUndirected=False):
    """Reads a weighted adjacency matrix W from CSV and returns it as sparse CSR matrix W together with n.
    CSV input format: 2 or 3 columns: row, col (, weight): weight is optional.
    Returns None, None if the file does not exist (instead of throwing an error).

    Parameters
    ----------
    fileName : string
        fileName including path
    skiprows : int, optional (Default=1)
        number of top rows to skip
    zeroindexing : bool, optional (default=True)
        whether first node is indexed by 0 (instead of by 1)
    n : int, optional (Default=0)
        number of different nodes
    delimiter : string, optional  (Default = ',')
        CSV delimiter: ',' or or '\t' or None (uses whitespace)
    usefloat : bool, optional (default=False)
        whether to use float instead of int for saving weights
        If not set, then weight is assumed to integer.
    doubleUndirected : bool, optional (default=False)
        whether to add for each edge, also a back edge
        If edge is already present, then doubles adds the weights

    Returns
    -------
    W : [n x n] sparse matrix: scipy.sparse.csr.csr_matrix
        weighted adjacency matrix in sparse format
        None in case the file does not exist
    n : int
        number of nodes
        None in case the file does not exist

    Assumes
    -------
    Assumes node labels 0-(n-1) [or 1-n for zeroindexing=False]
    If n is not specified, then the code taks n-1 = maximum integer in rows or columns (or n for "zeroindexing=False")
    File name does not matter. Even GZ is possible.
    Node ids need to be integers. Integers can start at 0 or 1 ("zeroindexing=True" or False).
    Weight is assumed integer. Otherwise float if "usefloat=True".
    """
    try:
        if usefloat:
            data = np.loadtxt(fileName, dtype=float, delimiter=delimiter, skiprows=skiprows)
        else:    # if dtype not specified, then float by default
            data = np.loadtxt(fileName, dtype=int, delimiter=delimiter, skiprows=skiprows)
        k, m = np.shape(data.T)
        assert k == 2 or k == 3                     # only accept 2 or 3 columns
        if k == 3:
            row, col, weight = data.T
        else:                                       # in case it only has 2 columns, then the weights are all 1s
            row, col = data.T
            weight = [1] * m                        # create the weight array so code later is the same

        if not zeroindexing:                        # transform from 1-indexing to 0-indexing
            row -= 1
            col -= 1
        if n is None:
            n = max(np.max(row),np.max(col))        # assumes the last node appears (= has at least one edge)
            n += 1                                  # assumes zeroindexing
        if doubleUndirected:                        # repeat the same edge also in other direction (np.array)
            row0 = row                               # sums weights for repeated directededges
            row = np.append(row, col)
            col = np.append(col, row0)
            weight = np.append(weight, weight)

        W = csr_matrix((weight, (row, col)), shape=(n, n))
        return W, n
    except IOError as error:
        # print("Error: {0:s}: could not be opened: {1:s}".format(fileName, error.strerror))
        raise error
        # return None, None


def save_W(fileName, W, delimiter=',', saveWeights=False):
    """Saves a sparse weighted adjacency matrix in CSV sparse format
    Includes a header and uses zero indexing.
    Deals with both integers and floats as weights.

    Parameters
    ----------
    fileName : string
        fileName including path
    W : [n x n] sparse matrix: scipy.sparse.csr.csr_matrix
        weighted adjacency matrix in sparse format
    delimiter : string, optional  (Default = ',')
        CSV delimiter: ',' or or '\t' or ' ' (not None!)
    """
    row, col = W.nonzero()
    weight = W.data
    if not saveWeights:
        data = np.array(zip(row, col))
        np.savetxt(fileName, data, fmt='%g', delimiter=delimiter, header='source,destination')
    else:
        data = np.array(zip(row, col, weight))
        np.savetxt(fileName, data, fmt='%g', delimiter=delimiter, header='source,destination,weight')


def load_X(fileName, n=None, k=None, skiprows=1, zeroindexing=True, delimiter=','):
    """Reads a belief matrix Xr from CSV in relational format (node, label, belief)
    Returns a dense [n x k] matrix X, together with n and k
    Returns None, None if the file does not exist
    CSV input format: 2 or 3 columns: node, label (, belief): belief is optional and otherwise assume = 1
    node indexes are integers starting at 0 or 1

    Parameters
    ----------
    fileName : string
        fileName including path
    n : int, optional (Default=0)
        number of different nodes
    k : int, optional (Default=0)
        number of different classes
    skiprows : int, optional (default=1)
        number of top rows to skip
    zeroindexing : bool, optional (default=True)
        whether first node is indexed by 0 [instead of by 1]
    delimiter : string, optional  (Default = ',')
        CSV delimiter: ',' or or '\t' or None (uses whitespace)

    Returns
    -------
    X : dense [n x k] np.array
        None in case the file does not exist
    n : int
        number of nodes
        None in case the file does not exist
    k : int
        number of classes
        None in case the file does not exist

    Assumes
    -------
    assumes node labels 0-(n-1), class labels 0-(k-1) [or 1-n and 1-k for zeroindexing=False]
    assumes node (n-1) [or n for zeroindexing=False] has at least one edge, if n is not explicitly specified
    assumes label (k-1) [or k for zeroindexing=False] is assigned to at least one node, if k is not explicitly specified
    """
    try:
        data = np.loadtxt(fileName, delimiter=delimiter, skiprows=skiprows)     # does float by default
        num_col, m = np.shape(data.T)

        assert num_col == 2 or num_col == 3, "The data file can contain only 2 or 3 columns"
        if num_col == 3:
            node, label, belief = data.T
        else:                       # in case it only has 2 columns, then the weights are all 1s
            node, label = data.T
            assert len(set(node)) == len(node), "With 2 input columns, every node can be assigned to maximal one class"
            belief = [1] * m        # create the weight array so code later is the same

        if not zeroindexing:        # transform from 1 to 0 indexing
            node -= 1
            label -= 1
        if n is None:
            n = np.max(node)        # assumes the last node appears (= has explicit beliefs)
            n += 1                  # assumes zeroindexing
        if k is None:
            k = np.max(label)       # assumes that at least one node has the last class as explicit belief
            k += 1                  # assumes zeroindexing

        Xs = csr_matrix((belief, (node, label)), shape=(n, k))
        X = np.array(Xs.todense())  # np.array cast required to avoid output format: (numpy.matrixlib.defmatrix.matrix)
        return X, n, k

    except IOError as error:
        return None, None, None


def save_X(fileName, X, delimiter=','):
    """Saves a dense explicit belief matrix X to CSV
    Includes a header and uses zero indexing

    Parameters
    ----------
    fileName : string
        fileName including path
    X : dense matrix
        explicit belief matrix
    delimiter : string, optional  (Default = ',')
        CSV delimiter: ',' or or '\t' or ' ' (not None!)
    """
    n, k = X.shape
    z = [(i, j, X[i,j]) for i in range(n) for j in range(k) if X[i, j] != 0]     # only save entries that are not 0
    data = np.array(z)
    np.savetxt(fileName, data, fmt='%g', delimiter=delimiter, header='node,class,belief')


def load_Xd(fileName, skiprows=1, delimiter=','):
    """Loads a dictionary node id -> class id
    CSV input format: node id, class id.

    Parameters
    ----------
    fileName : string
        fileName including path
    skiprows : int, optional (Default=1)
        number of top rows to skip
    delimiter : string, optional  (Default = ',')
        CSV delimiter: ',' or or '\t' or None (uses whitespace)
    Returns
    -------
    Xd : dictionary
        maps node ids to class ids
    """
    try:
        data = np.loadtxt(fileName, dtype=int, delimiter=delimiter, skiprows=skiprows)
        '''if dtype not specified, then float by default'''
        k, _ = np.shape(data.T)
        assert k == 2       # only accept dictionary
        keys, values = data.T
        Xd = {k: v for k,v in zip(keys,values)}
        return Xd
    except IOError as error:
        return None, None, None


def save_Xd(fileName, Xd, delimiter=','):
    """Saves a dictionary node id -> class id

    Parameters
    ----------
    fileName : string
        fileName including path
    Xd : dictionary
        maps node ids to class ids
    delimiter : string, optional  (Default = ',')
        CSV delimiter: ',' or or '\t' or ' ' (not None!)
    """
    z = [(k,v) for k,v in Xd.items()]
    data = np.array(z)
    np.savetxt(fileName, data, fmt='%g', delimiter=delimiter, header='node,class')


def load_H(fileName, l=None, k=None, skiprows=1, zeroindexing=True, delimiter=','):
    """Reads a [l x k] directed potential or coupling matrix H and returns a dense [l x k] numpy array H, together with l and k
    Returns None, None, None if the file does not exist
    CSV input format for H(j,i): j, i, weight (from, to, weight)

    Parameters
    ----------
    fileName : string
        fileName including path
    l : int, optional (Default=None)
        number of different classes (vertical = rows)
    k : int, optional (Default=None)
        number of different classes (horizontal = columns)
    skiprows : int, optional (Default=1)
        number of top rows to skip
    zeroindexing : bool, optional (default=True)
        whether first node is indexed by 0 [instead of by 1]
    delimiter : string, optional  (Default = ',')
        CSV delimiter: ',' or or '\t' or None (uses whitespace)

    Returns
    -------
    H : dense [l x k] numpy array
        None in case the file does not exist
    l : int
        number of classes (vertical = rows)
        None in case the file does not exist
    k : int
        number of classes (horizontal = columns)
        None in case the file does not exist

    Assumes
    -------
    Assumes class labels 0-(k-1) [or 1-k for zeroindexing=False]
    Assumes class (k-1) [or k for zeroindexing=False] must be mentioned, if k is not explicitly specified
    """
    try:
        data = np.loadtxt(fileName, delimiter=delimiter, skiprows=skiprows)
        row, col, weight = data.T
        if not zeroindexing:        # transform from 1 to 0 indexing
            row -= 1
            col -= 1
        if l is None:
            l = np.max(row)         # assumes that last class is mentioned
            l += 1                  # assumes zeroindexing
        if k is None:
            k = np.max(col)         # assumes that last class is mentioned
            k += 1                  # assumes zeroindexing
        Hs = csr_matrix((weight, (row, col)), shape=(l, k))
        H = np.array(Hs.todense())  # todense would create a matrix object, but we use arrays
        return H, l, k
    except IOError as error:
        return None, None, None


def save_H(fileName, H, delimiter=','):
    """Saves a dense [l x k] coupling matrix H to CSV
    CSV format for H(j,i): j, i, weight (from, to, weight)
    Includes a header and uses zero indexing

    Parameters
    ----------
    fileName : string
        fileName including path
    H : dense matrix
        coupling matrix
    delimiter : string, optional  (Default = ',')
        CSV delimiter: ',' or or '\t' or ' ' (not None!)
    """
    l, k = H.shape
    z = [(j, i, H[j, i]) for j in range(l) for i in range(k)]
    data = np.array(z)
    np.savetxt(fileName, data, fmt='%g', delimiter=delimiter, header='from,to,weight')


def save_csv_records(fileName, records, header_row=None):
    """Given a filename and records (= list of tuples, e.g. from a database query), writes the records to a CSV file
    Overwrites the content of an existing file.
    Optionally takes a header as argument for first row"""
    with open(fileName, 'w') as f:           # 'w' overwrite, 'a' append
        writer = csv.writer(f, delimiter=',')
        if header_row is not None:
            writer.writerow(header_row)
        for row in records:
            writer.writerow(row)


def load_csv_records(fileName):
    """Given a CSV filename, returns records (= list of tuples)
    Automatically transforms '2' to integer 2 and '1.9' to float 1.9
    Output format is list tuples to be consistent with the result of a database query"""
    def cast_if_possible(x):
        """Function that takes a string and returns int or float if possible, otherwise just string
        Useful when reading from database where output is string by default before casting"""
        try:
            return int(x)
        except ValueError:
            try:
                return float(x)
            except ValueError:
                return x
    records = []
    with open(fileName, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            row2 = [cast_if_possible(x) for x in row]
            records.append(tuple(row2))
    return records
