"""
Generators for labeled random graphs for SSL-H
Inspiration: http://networkx.github.io/documentation/latest/_modules/networkx/generators/random_graphs.html

(C) Wolfgang Gatterbauer, 2016
"""


from __future__ import print_function
from __future__ import division
import random
import warnings
from random import randint
from numpy.random import random_sample, shuffle
import numpy as np
from scipy.sparse import csr_matrix
from scipy import optimize
from math import ceil, pi
import collections          # collections.Counter
from SSLH_utils import (row_normalize_matrix,
                        check_normalized_beliefs,
                        from_dictionary_beliefs,
                        calculate_potential_from_row_normalized)
import networkx as nx
from scipy.sparse.csgraph import connected_components
RANDOMSEED = None


def planted_distribution_model_H(n, alpha, H, d_out,
                                 distribution='powerlaw', exponent=-0.5,
                                 directed=True,
                                 backEdgesAllowed=False,
                                 sameInAsOutDegreeRanking=False,
                                 debug=0):
    """Variation on planted_distribution_model that uses (H, d_out) instead of (P, m)
    """
    k = len(alpha)
    if not isinstance(d_out, (collections.Sequence, np.ndarray)):  # allow single number as input
        d_out = [d_out] * k

    P = calculate_potential_from_row_normalized(H, alpha, d_out)
    m = np.rint(n * np.array(alpha).dot(np.array(d_out)))

    return planted_distribution_model(n=n, alpha=alpha, P=P, m=m,
                                            distribution=distribution, exponent=exponent,
                                            directed=directed,
                                            backEdgesAllowed=backEdgesAllowed,
                                            sameInAsOutDegreeRanking=sameInAsOutDegreeRanking,
                                            debug=debug)



def planted_distribution_model(n, alpha, P, m,
                               distribution='powerlaw', exponent=-0.5,
                               directed=True,
                               backEdgesAllowed=False,
                               sameInAsOutDegreeRanking=False,
                               debug=0):
    """Creates a directed random graph with planted compatibility matrix 'P'.
    Accepts (n, alpha_vec, P, m). The alternative version accepts: (n, alpha_vec, H, d_out_vec).
    If directed==False: creates an undirected graph. Requires:
        1) P to be symmetric, with identical row and column sum
        2) ignores: backEdgesAllowed, sameInAsOutDegreeRanking
    Notice: m = |W| for directed, but 2m = |W| for undirected.
    Notice: Average outdegree d_out = m/n for directed, but average total degree d = 2m/n for directed and undirected

    Parameters
    ----------
    n : int
        number of nodes
    alpha : k-dimensional ndarray or list
        node label distribution
    P : [k,k] ndarray
        Compatibility matrix (no need for column-normalized or symmetric)
    m : int
        total number of edges
    distribution : string, optional (Default = 'powerlaw')
        'uniform', 'triangle', 'powerlaw': used with "create_distribution_vector(n, m, distribution, exponent)"
    exponent : float, optional (Default = None)
        only for 'powerlaw', by default = -1
    directed : Boolean, optional (Default = True)
        False: then constructs an undirected graph. Requires symmetric doubly stochastic potential
    backEdgesAllowed : Boolean, optional (Default = False)
        False: then two nodes cannot be connected by two edges back and forth
        Overwritten for undirected to be False
    sameInAsOutDegreeRanking : Boolean, optional (Default = False)
        True: then node with highest indegree also has highest outdegree among its peers
        Overwritten for undirected to be False
    debug : int (Default = 0)
        0: print nothing
        1: prints some statistics
        2: prints even node degree distributions

    Returns
    -------
    W : sparse.csr_matrix
        sparse weighted adjacency matrix
    Xd : dictionary
        Explicit beliefs
    """

    # -- Jump out from inner loop if graph cannot be found
    # Define an exception that allows to jump out from inner loop to a certain outer loop
    class GraphNotFound(Exception):
        pass


    # -- Initialization
    alpha = np.asarray(alpha)
    k = len(alpha)
    k1, k2 = P.shape
    assert k == k1 and k == k2

    if not directed:
        for i in range(k):                  # symmetric matrix
            for j in range(k):
                assert P[i,j] == P[j,i]

        if backEdgesAllowed:
            warnings.warn("'backEdgesAllowed' set to False")
            backEdgesAllowed = False            # Otherwise, same edge could be created twice, redundant
        if sameInAsOutDegreeRanking:
            warnings.warn("'sameInAsOutDegreeRanking' set to False")
            sameInAsOutDegreeRanking = False    # Otherwise in uniform distribution not correct


    # --- Big loop that attempts to create a graph for 20 times (sometimes the parameters don't allow a graph)
    attempt = 0
    finished = False
    while attempt < 20 and not finished:

        # -- (1) n_vec: np.array: number of nodes for each class
        n_vec = np.array(alpha*n, int)  # number of nodes in each class
        delta = np.sum(n_vec) - n
        n_vec[k-1] = n_vec[k-1] - delta     # make sure sum(N)=n, in case there are rounding errors, correct the last entry

        # -- Xd: dictionary: class of each node
        Xl = [ [i]*n_vec[i] for i in range(k) ]
        Xl = np.hstack(Xl)                  # flatten nested array
        shuffle(Xl)                         # random order of those classes. Array that maps i -> k
        Xd = {i : Xl[i] for i in range(n)}  # Xd: dictionary that maps i -> k


        P_tot = m * P / P.sum()
        P_tot = np.rint(P_tot).astype(int)

        delta = m - P_tot.sum()
        P_tot[0][0] = P_tot[0][0] + delta

        assert np.all(P_tot >= 0), "Negative values in P_tot due to rounding errors. Change graph parameters"   # Can happen for H with 0 entries due to necessary rounding to closest integers
        m_out_vec = P_tot.sum(1, keepdims=False)
        m = np.sum(m_out_vec)

        # -- m_in_vec: number of incoming edges per class / d_in_vec: np.array: average in-degree per class
        m_in_vec = P_tot.sum(0, keepdims=False)     # sums along vertical axis
        d_in_vec = 1. * m_in_vec / n_vec

        # -- (3) list_OutDegree_vecs, list_InDegree_vec: list of np.array: distribution of in/outdegrees for nodes in each class
        list_OutDegree_vec = []
        list_InDegree_vec = []
        for i in range(k):
            if not directed:            # undirected case works differently: create double the degrees
                m_out_vec[i] *= 2       # but then deduce 2 outdegrees per edge (ignoring indegrees completely)
            out_distribution = create_distribution_vector(n_vec[i], m_out_vec[i], distribution=distribution, exponent=exponent)
            list_OutDegree_vec.append(out_distribution)
            if directed:
                in_distribution = create_distribution_vector(n_vec[i], m_in_vec[i], distribution=distribution, exponent=exponent)
                list_InDegree_vec.append(in_distribution)

        # -- listlistNodes: list of randomly shuffled node ids for each class
        listlistNodes = [[node for node in range(n) if Xd[node] == i] for i in range(k)]
        for innerList in listlistNodes:
            shuffle( innerList )

        # -- list_OutDegree_nodes: list of list of node ids:
        #   contains for each outgoing edge in each class the start node id, later used for sampling
        list_OutDegree_nodes = []
        for i in range(k):
            innerList = []
            for j, item in enumerate(listlistNodes[i]):
                innerList += [item] * list_OutDegree_vec[i][j]
            list_OutDegree_nodes.append(innerList)

        if directed:
            if not sameInAsOutDegreeRanking:        # shuffle the randomly ordered list again before assigning indegrees
                for innerList in listlistNodes:
                    np.random.shuffle( innerList )
            list_InDegree_nodes = []        # list of each node times the outdegree
            for i in range(k):
                innerList = []
                for j, item in enumerate(listlistNodes[i]):
                    innerList += [item] * list_InDegree_vec[i][j]
                list_InDegree_nodes.append(innerList)

        if debug >= 1:
            print("\n-- Print generated graph statistics (debug >= 1):")

            print("n_vec: ", n_vec)

            print("m_out_vec: ", m_out_vec)
            print("m: ", m)
            print("P:\n", P)
            print("P_tot:\n", P_tot)
            if not directed:
                print ("P_tot x 2:\n", P_tot*2)
            print ("m_in_vec: ", m_in_vec)
            print ("d_in_vec: ", d_in_vec)
            for i in range(k):

                print ("sum(list_OutDegree_vec[", i, "]): ", sum(list_OutDegree_vec[i]))
                print ("len(list_OutDegree_vec[", i, "]): ", len(list_OutDegree_vec[i]))
                if debug == 2:
                    print ("out_distribution class[", i, "]:\n", list_OutDegree_vec[i])
                if directed:
                    print ("sum(list_InDegree_vec[", i, "]): ", sum(list_InDegree_vec[i]))
                    print ("len(list_InDegree_vec[", i, "]): ", len(list_InDegree_vec[i]))
                    if debug == 2:
                        print ("in_distribution class[", i, "]:\n", list_InDegree_vec[i])

            print("list_OutDegree_nodes:\n ", list_OutDegree_nodes)


        # -- (4) create actual edges: try 10 times
        row = []
        col = []
        edges = set()       # set of edges, used to verify if a given edge already exists
        try:

            for i in range(k):
                for j in range(k):
                    counterCollision = 0
                    while P_tot[i][j] > 0:

                        # -- pick two compatible nodes for candidate edge (s, t)
                        i_index = local_randint(len(list_OutDegree_nodes[i]))
                        s = list_OutDegree_nodes[i][i_index]

                        if directed:
                            j_index = local_randint(len(list_InDegree_nodes[j]))
                            t = list_InDegree_nodes[j][j_index]
                        else:
                            j_index = local_randint(len(list_OutDegree_nodes[j]))   # Re-use OutDegree
                            t = list_OutDegree_nodes[j][j_index]

                        # -- check that this pair can be added as edge, then add it
                        if (not s == t and
                                not (s, t) in edges and
                                (backEdgesAllowed or not (t, s) in edges)):


                            def time_funct1():
                                row.append(s)
                                col.append(t)
                                edges.add((s, t))


                            def time_funct2():
                                del list_OutDegree_nodes[i][i_index]

                                if directed:
                                    del list_InDegree_nodes[j][j_index]
                                else:
                                    if i == j and i_index < j_index:            # careful with deletion if i == j
                                        del list_OutDegree_nodes[j][j_index-1]  # prior deletion may have changed the indices
                                    else:
                                        del list_OutDegree_nodes[j][j_index]

                                P_tot[i][j] -= 1
                                counterCollision = 0

                            time_funct1()
                            time_funct2()


                        # -- throw exception if too many collisions (otherwise infinite loop)
                        else:
                            counterCollision += 1
                            if counterCollision > 2000:
                                raise GraphNotFound("2000 collisions")

        except GraphNotFound as e:
            print ("Failed attempt #{}".format(attempt))
            attempt +=1
        else:
            finished = True

    if not finished:
        raise GraphNotFound("Graph generation failed")

    if not directed:
        row2 = list(row)    # need to make a temp copy
        row.extend(col)
        col.extend(row2)

    Ws = csr_matrix(([1]*len(row), (row, col)), shape=(n, n))
    return Ws, Xd



# === Graph generation helper functions ===========================================================================================================


def create_distribution_vector(n, m, distribution='uniform', exponent=-1):
    """Returns an integer distribution of length n, with total m items, and a chosen distribution

    Parameters
    ----------
    n : int
        The number of x values
    m : int
        The total sum of all entries to be created
    distribution : string, optional (Default = 'uniform')
        'uniform', 'triangle', 'powerlaw'
    exponent : float, optional (Default = None)
        only for 'powerlaw', by default = -1

    Returns
    -------
    distribution : np.array
        list of n values with sum = m, in decreasing value
    """
    if distribution == 'uniform':   # e.g., n=10, m=23: dist = [3, 3, 3, 2, 2, 2, 2, 2, 2, 2]
        d = m // n
        mod = m % n
        dist = [d+1] * mod, [d] * (n-mod)

    elif distribution == 'triangle':
        def triangle(x, slope):
            return int(ceil((x+pi/4)*slope-pi/pi/1e6))
            # Explanation 1: pi as irrational number for pivot of slope is chosen to make sure that there is
            #   always some slope from that point with a total number of points below (thus, only one point is added
            #   for an infinitesimal small increase of slope)
            # Explanation 2: pivot has vertical point to allow 0 edges for some nodes (small to only use if necessary)

        def sum_difference(k):  # given slope k, what is the difference to m currently
            s = m
            for i in range (n):
                s -= triangle(i,k)
            return s

        k0 = 2.*m/n**2
        slope = optimize.bisect(sum_difference, 0, 2*k0, xtol=1e-20, maxiter=500)
        # Explanation: find the correct slope so that exactly m points are below the separating line
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.bisect.html#scipy.optimize.bisect
        dist = []
        for i in range (n-1, -1, -1):        # counting backwards to start with biggest values first
            dist.append(triangle(i,slope))

    elif distribution == 'powerlaw':
        exponent -= pi/100000   # make sure that the exponent is irrational, allows to always find a solution


        def sum_difference(top):
            return m-np.sum(np.ceil(top * (np.arange(1, n+1) ** exponent) - pi/1e6))
            # Explanation: -pi/100000 allows 0 if m<n (as for triangle)

        def powerlaw_vec(top):
            return np.ceil(top * (np.arange(1, n + 1) ** exponent) - pi / 1e6).astype(int)

        integral = 1. / (exponent + 1) * (n**(exponent+1)-1)
        a0 = 1.*m/integral              # upper bound
        top = optimize.bisect(sum_difference, 0, a0, xtol=1e-20, maxiter=500)

        dist = powerlaw_vec(top)
    else:
        raise Exception("You specified a non-existing method")

    dist = np.hstack(dist).astype(int)  # creates numpy array (plus flattens the array, necessary for triangular)
    assert dist.sum() == m
    return dist


def local_randint(n):
    """Chooses random integer between 0 and n-1. Just a wrapper around ranint that also allows n=0, in which case returns 0"""
    if n == 0:
        return 0
    elif n > 0:
        return randint(0, n-1)
    else:
        raise Exception("Value >= 0 required")


# === Graph statistics ===============================================================================================================


def calculate_nVec_from_Xd(Xd):
    """Calculates 'n_vec': the number of times each node class occurs in graph.
    Given graph with explicit beliefs in dictionary format 'Xd'.
    Assumes zeroindexing.
    """
    X0 = from_dictionary_beliefs(Xd)
    return X0.sum(axis=0)



def calculate_Ptot_from_graph(W, Xd):
    """Calculates [k x k] array 'P_tot': the number of times each edge type occurs in graph.
    Uses a sparse directed (incl. undirected) [n x n] adjacency matrix 'W' and explicit beliefs in dictionary format 'Xd'.
    [Does not ignore weights of 'W'. Updated with simpler multiplication]
    Assumes zeroindexing.
    If normalizing is required later:
        m = sum(P_tot.flatten())       # sum of all entries = number of edges
        Pot = 1. * P_tot / m           # Potential: normalized sum = 1
        P_tot = Pot
    """
    X0 = from_dictionary_beliefs(Xd)
    return X0.transpose().dot(W.dot(X0))


def calculate_outdegree_distribution_from_graph(W, Xd=None):
    """Given a graph 'W', returns a dictionary {degree -> number of nodes with that degree}.
    If a dictionary 'Xd' of explicit beliefs is given, then returns a list of dictionaries, one for each node class.
    Takes weight into acount [OLD: Ignores weights of 'W'. Assumes zeroindexing.]
    Transpose W to get indegrees.
    """
    n, _ = W.shape
    countDegrees = W.dot(np.ones((n, 1))).flatten().astype(int)

    if Xd is None:
        countIndegrees = collections.Counter(countDegrees)  # count multiplicies of nodes classes
        return countIndegrees


    else:
        listCountIndegrees = []

        X0 = from_dictionary_beliefs(Xd)
        for col in X0.transpose():
            countDegreesInClass = countDegrees*col      # entry-wise multiplication
            countDegreesInClass = countDegreesInClass[np.nonzero(countDegreesInClass)]
            countIndegreesInClass = collections.Counter(countDegreesInClass)
            listCountIndegrees.append(countIndegreesInClass)

        return listCountIndegrees


def calculate_average_outdegree_from_graph(W, Xd=None):
    """Given a graph 'W', returns the average outdegree for nodes in graph.
    If a dictionary 'Xd' of explicit beliefs is given, returns a list of average degrees, one for each node class.
    Assumes zeroindexing. Ignores weights of 'W'.
    """
    if Xd is None:
        d_dic = calculate_outdegree_distribution_from_graph(W)
        return 1. * np.sum(np.multiply(d_dic.keys(), d_dic.values())) / sum(d_dic.values())
    else:
        d_dic_list = calculate_outdegree_distribution_from_graph(W, Xd)
        d_vec = []
        for d_dic in d_dic_list:
            d_vec.append(1. * np.sum(np.multiply(d_dic.keys(), d_dic.values())) / sum(d_dic.values()))
        return d_vec


def create_blocked_matrix_from_graph(W, Xd):
    """Given a graph 'W' and the classes of each node, permutes the labels of the nodes
        as to allow nicer block visualization with np.matshow.
        Thus nodes of same type will have adjacent ids.
    The new matrix starts with nodes of class 0, then 1, then 2, etc.
    Assumes zeroindexing. Xd needs to have appropriate size.
    Returns a new matrix W and new dictionary Xd
    """
    row, col = W.nonzero()                      # transform the sparse W back to row col format
    weight = W.data
    nodes = np.array(Xd.keys())
    classes = np.array(Xd.values())

    inds = np.lexsort((nodes, classes))         #  Sort by classes, then by nodes
    ranks = inds.argsort()                      #  This gives the new ranks of an original node

    W_new = csr_matrix((weight, (ranks[row], ranks[col])), shape=W.shape)  # edges only in one direction
    Xd_new = dict(zip(ranks[nodes], classes))
    return W_new, Xd_new





