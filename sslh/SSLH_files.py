"""
File manipulation for SSL-H: Loads from and saves to csv formats with comma ',' delimiter.
Nomenclature: W, X, H[j,i], n, l, k, Xd, Xr (relational)

(C) Wolfgang Gatterbauer, 2016
"""


import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import csv
import warnings


def load_W(fileName, skiprows=1, zeroindexing=True, n=None, doubleUndirected=False):
    """Reads a weighted adjacency matrix W from CSV and returns it as sparse CSR matrix W together with n.
    CSV input format: 2 or 3 columns: row, col (, weight): weight is optional.
    [Deprecated: Returns None, None if the file does not exist (instead of throwing an error).]

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
    doubleUndirected : bool, optional (default=False)
        whether to add for each edge, also a back edge
        If edge is already present, then doubles the weights

    Returns
    -------
    W : [n x n] sparse matrix: scipy.sparse.csr.csr_matrix
        weighted adjacency matrix in sparse format
    n : int
        number of nodes

    Assumes
    -------
    Assumes node labels 0-(n-1) [or 1-n for zeroindexing=False]
    If n is not specified, then the code taks n-1 = maximum integer in rows or columns (or n for "zeroindexing=False")
    Node ids need to be integers. Integers can start at 0 or 1 ("zeroindexing=True" or False).
    """
    try:
        data = pd.read_csv(fileName, delimiter=',', skiprows=skiprows, header=None)

        m, k = data.shape
        assert k == 2 or k == 3                     # only accept 2 or 3 columns
        row = data.iloc[:, 0].values
        col = data.iloc[:, 1].values
        if k== 3:
            weight = data.iloc[:, 2].values
        else:  # in case it only has 2 columns, then the weights are all 1s
            weight = [1] * m  # create the weight array so code later is the same

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
        raise error


def save_W(fileName, W, saveWeights=False):
    """Saves a sparse weighted adjacency matrix in CSV sparse format
    Includes a header and uses zero indexing.
    Deals with both integers and floats as weights.
    [Deprecated: delimiter=',']

    Parameters
    ----------
    fileName : string
        fileName including path
    W : [n x n] sparse matrix: scipy.sparse.csr.csr_matrix
        weighted adjacency matrix in sparse format
    saveWeights: Boolean  (Default = False)
        saves weights as 3rd column, otherwise just 2 columns
    """
    row, col = W.nonzero()
    weight = W.data

    if not saveWeights:
        d = np.array([row, col]).transpose()
        df = pd.DataFrame(d, columns=['source', 'destination'])             # order 'source' -> 'destination' needs to be specified, otherwise alphabetically
    else:
        d = np.array([row, col, weight]).transpose()
        df = pd.DataFrame(d, columns=['source', 'destination', 'weight'])

    df.to_csv(fileName, sep=',', index=False)


def load_X(fileName, n=None, k=None, skiprows=1, zeroindexing=True):
    """Reads a belief matrix Xr from CSV in relational format (node, label, belief)
    Returns a tuple: a dense [n x k] matrix X, together with n and k
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
        data = pd.read_csv(fileName, delimiter=',', skiprows=skiprows, header=None)

        m, num_col = data.shape
        assert num_col == 2 or num_col == 3                     # only accept 2 or 3 columns
        node = data.iloc[:, 0].values
        label = data.iloc[:, 1].values
        if num_col == 3:
            belief = data.iloc[:, 2].values
        else:  # in case it only has 2 columns, then each class can only have one belief
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


def save_X(fileName, X):
    """Saves a dense explicit belief matrix X to CSV
    Includes a header and uses zero indexing

    Parameters
    ----------
    fileName : string
        fileName including path
    X : dense matrix
        explicit belief matrix
    """
    n, k = X.shape

    X2 = csr_matrix(X)              # fast way to get X matrix into row / col / weight format
    row, col = X2.nonzero()
    weight = X2.data
    data = np.array([row, col, weight]).transpose()

    df = pd.DataFrame(data, columns=['node', 'class', 'belief'])
    df['node'] = df['node'].astype(int)     # Force node and class datatype workaround (https://github.com/pandas-dev/pandas/issues/9287)
    df['class'] = df['class'].astype(int)
    df.to_csv(fileName, sep=',', index=False)



def save_csv_record(fileName, record, append=True):
    """Given a filename and record (= one tuple or list), writes the record to a CSV file
    By default appends to existing file"""
    if append:
        with open(fileName, 'a') as f:      # 'a' append
            writer = csv.writer(f, delimiter=',')
            writer.writerow(record)
    else:
        with open(fileName, 'w') as f:      # 'w' overwrite
            writer = csv.writer(f, delimiter=',')
            writer.writerow(record)


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
