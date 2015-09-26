"""
Test class for SSL-H inference methods (BP, exact Maximum marginals, linBP)

First version: Dec 6, 2014
This version: Sept 22, 2015
Author: Wolfgang Gatterbauer <gatt@cmu.edu>
"""


from __future__ import division                             # allow integer division
from SSLH_inference import linBP_undirected
from SSLH_utils import (to_centering_beliefs,
                        eps_convergence_linbp)
from SSLH_files import load_W
import numpy as np
import matplotlib.pyplot as plt
import os


# -- Determine path to data irrespective (!) of where the file is run from
from os.path import abspath, dirname, join
from inspect import getfile, currentframe
current_path = dirname(abspath(getfile(currentframe())))
data_directory = join(current_path, 'data/')
fig_directory = join(current_path, 'figs/')


def test_linBP_undirected_Torus():
    print "\n-- 'linBP_undirected', 'eps_convergence_linbp' with Torus --"

    # -- Load W, create X and P
    W, n = load_W(join(data_directory, 'Torus_W.csv'), zeroindexing=False, delimiter=',')
    X = np.array(
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1],
     [0, 0, 0],
     [0, 0, 0],
     [0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]])
    H = np.array(
    [[0.1, 0.8, 0.1],
     [0.8, 0.1, 0.1],
     [0.1, 0.1, 0.8]])
    print "W (dense):\n", W.todense()
    print "X:\n", X
    print "H:\n", H
    Hc = to_centering_beliefs(H)
    print "Hc:\n", Hc
    print

    # -- Other parameters
    eps = 0.4
    numMaxIt = 20

    eps_max = eps_convergence_linbp(Hc, W)
    print "eps_max: ", eps_max
    print "eps: ", eps
    Hc2 = Hc*eps
    print "P*eps:\n", Hc2
    print

    # --- linBP
    listF = linBP_undirected(X, W, Hc2, numIt=numMaxIt, debug=2)

    # --- Display BP results
    print "linBP results:"
    print "last two F:"
    print listF[-2]
    print listF[-1]
    print "\nValues for node 6 (zero indexing):"
    print listF[:, 6, :]

    # --- Visualize BP results
    filename = join(fig_directory, 'temp.pdf')
    print "\nVisualize values for node 3 (zero indexing):"
    node = 3
    plt.plot(listF[:, node, :], lw=2)
    plt.xlabel('# iterations')
    plt.ylabel('belief')
    plt.xlim(0, numMaxIt)

    print filename

    plt.savefig(filename, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype='letter', format='pdf',
                transparent=True, bbox_inches='tight', pad_inches=0.1,
                frameon=None)
    os.system("chmod 744 " + filename)  # first change permissions in order to open PDF
    os.system("open " + filename)       # open PDF


if __name__ == '__main__':
    test_linBP_undirected_Torus()
