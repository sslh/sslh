"""
Test class for SSL-H inference methods (BP, linBP_directed)

Sample code to demonstrate the use of functions
This code plots the time for BP vs directed LinBP

(C) Wolfgang Gatterbauer, 2016
"""

from __future__ import division             # allow integer division
from __future__ import print_function
import numpy as np
import datetime
import random
import os                                   # for displaying created PDF
import time
from SSLH_files import (save_csv_record,
                        save_W,
                        save_X,
                        load_W,
                        load_X)
from SSLH_utils import (from_dictionary_beliefs,
                        create_parameterized_H,
                        replace_fraction_of_rows,
                        to_centering_beliefs,
                        eps_convergence_directed_linbp)
from SSLH_graphGenerator import planted_distribution_model
from SSLH_inference import linBP_directed, beliefPropagation
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from pylab import (figure, xlabel, ylabel, savefig, show, xlim, ylim, xticks, grid, title)
import pandas as pd
pd.set_option('display.max_columns', None)      # show all columns
pd.options.mode.chained_assignment = None       # default='warn'
import seaborn.apionly as sns                   # importing without activating it. For color palette



# -- Determine path to data *irrespective* of where the file is run from
from os.path import abspath, dirname, join
from inspect import getfile, currentframe
current_path = dirname(abspath(getfile(currentframe())))
figure_directory = join(current_path, 'figs')
data_directory = join(current_path, 'data')



# -- Setup
# Since graph creation takes most time, especially for large graphs, saves graphs to a file format, then loads them later again.
# To save time, change to CREATE_DATA = ADD_DATA = CREATE_GRAPH = False after the first iteration
CHOICE = 1
CREATE_DATA = True
ADD_DATA = True
CREATE_GRAPH = True
SHOW_FIG = True
SHOW_GRAPH_TIME = False

filename = 'Fig_Timing_AAAI_{}'.format(CHOICE)
csv_filename = '{}.csv'.format(filename)
header = ['n',
          'type',
          'time',]


if CREATE_DATA:
    save_csv_record(join(data_directory, csv_filename), header, append=False)


# -- Default Graph parameters
distribution = 'powerlaw'
exponent = -0.3
k = 3
a = 1
err = 0
avoidNeighbors = False
convergencePercentage_W = None
f = 0.1
# propagation
pyamg = True
propagation_echo = True
scaling = 10
alpha = 0
beta = 0
gamma = 0
s = 0.1
numMaxIt = 10

# display
xtick_lab = [0.001, 0.01, 0.1, 1]
ytick_lab = np.arange(0, 1, 0.1)
xmin = 1000
xmax = 5e7
ymin = 1e-2
ymax = 5e3
eps_max_tol=1e-04

# -- Main Options
if CHOICE == 1:
    n_vec = [100, 200, 400, 800, 1600, 3200, 6400, 12800,
             25600, 51200, 102400,
             204800, 409600, 819200, 1638400, 3276800
             ]
    repeat_vec = [20, 20, 20, 20, 20, 10, 10, 5,
                  3, 3, 3,
                  3, 3, 3, 3, 3]
    eps_max_tol = 1e-02
    h = 8
    d = 10

else:
    raise Warning("Incorrect choice!")

alpha0 = np.array([a, 1., 1.])
alpha0 = alpha0 / np.sum(alpha0)
H0 = create_parameterized_H(k, h, symmetric=False)
H0c = to_centering_beliefs(H0)

RANDOMSEED = None  # For repeatability
random.seed(RANDOMSEED)  # seeds some other python random generator
np.random.seed(seed=RANDOMSEED)  # seeds the actually used numpy random generator; both are used and thus needed
print("CHOICE: {}".format(CHOICE))


def save_tuple(n, label, time):
    tuple = [str(datetime.datetime.now())]
    text = [n, label, time]
    tuple.extend(text)
    print("time {}: {}".format(label, time))
    save_csv_record(join(data_directory, csv_filename), tuple)


# -- Create data
if CREATE_DATA or ADD_DATA:

    for i, n in enumerate(n_vec):
        print("\nn: {}".format(n))
        repeat = repeat_vec[i]

        # -- Graph
        if CREATE_GRAPH:
            start = time.time()
            W, Xd = planted_distribution_model(n, alpha=alpha0, P=H0, m = n*d,
                                                      distribution=distribution,
                                                      exponent=exponent,
                                                      directed=True,
                                                      debug=False)
            X0 = from_dictionary_beliefs(Xd)
            time_graph = time.time() - start

            save_W(join(data_directory, '{}_{}_W.csv'.format(filename, n)), W, saveWeights=False)
            save_X(join(data_directory, '{}_{}_X.csv'.format(filename, n)), X0)
            save_tuple(n, 'graph', time_graph)

        else:
            W, _ = load_W(join(data_directory, '{}_{}_W.csv'.format(filename, n)), skiprows=1, zeroindexing=True, n=None, doubleUndirected=False)
            X0, _, _ = load_X(join(data_directory, '{}_{}_X.csv'.format(filename, n)), n=None, k=None, skiprows=1, zeroindexing=True)

        # -- Repeat loop
        for i in range(repeat):
            print("\n  repeat: {}".format(i))

            X2, ind = replace_fraction_of_rows(X0, 1-f, avoidNeighbors=avoidNeighbors, W=W)

            # -- Estimate Esp_max
            start = time.time()
            eps_max = eps_convergence_directed_linbp(P=H0, W=W, echo=propagation_echo, pyamg=pyamg, tol=eps_max_tol)
            time_eps_max = time.time() - start
            save_tuple(n, 'eps_max', time_eps_max)
            eps = s * eps_max

            # -- Propagate with LinBP
            X2c = to_centering_beliefs(X2, ignoreZeroRows=True)
            H2c = to_centering_beliefs(H0)
            try:
                start = time.time()
                F, actualIt, actualPercentageConverged = \
                    linBP_directed(X2c, W, H0,
                                   eps=eps,
                                   echo=propagation_echo,
                                   numMaxIt=numMaxIt,
                                   convergencePercentage=convergencePercentage_W,
                                   convergenceThreshold=0.9961947,
                                   debug=2)
                time_prop = time.time() - start
            except ValueError as e:
                print (
                "ERROR: {}: d={}, h={}".format(e, d, h))
            else:
                save_tuple(n, 'prop', time_prop)

            # -- Propagate with BP
            if n < 5e5:
                try:
                    start = time.time()
                    F, actualIt, actualPercentageConverged = \
                        beliefPropagation(X2, W, H0 ** eps,
                                       numMaxIt=numMaxIt,
                                       convergencePercentage=convergencePercentage_W,
                                       convergenceThreshold=0.9961947,
                                       debug=2)
                    time_bp = time.time() - start
                except ValueError as e:
                    print(
                        "ERROR: {}: d={}, h={}".format(e, d, h))
                else:
                    save_tuple(n, 'bp', time_bp)


# -- Read, aggregate, and pivot data for all options
df1 = pd.read_csv(join(data_directory, csv_filename))


# Aggregate repetitions
df2 = df1.groupby(['n', 'type']).agg \
    ({'time': [np.mean, np.median, np.std, np.size],  # Multiple Aggregates
      })
df2.columns = ['_'.join(col).strip() for col in df2.columns.values]  # flatten the column hierarchy
df2.reset_index(inplace=True)  # remove the index hierarchy
df2.rename(columns={'time_size': 'count'}, inplace=True)
print("\n-- df2 (length {}):\n{}".format(len(df2.index), df2.head(15)))

# Pivot table
df3 = pd.pivot_table(df2, index=['n'], columns=['type'], values=['time_mean', 'time_std'] )  # Pivot
df3.columns = ['_'.join(col).strip() for col in df3.columns.values]  # flatten the column hierarchy
df3.reset_index(inplace=True)  # remove the index hierarchy
print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(30)))

# Extract values
X = df3['n'].values     # plot x values
X = X*d                 # calculate edges (notice that for symmetric we would have to divide by 2 as one edge woudl appear twice in symmetric adjacency matrix)
if SHOW_GRAPH_TIME:
    Y_graph = df3['time_mean_graph'].values
Y_eps_max = df3['time_mean_eps_max'].values
Y_prop = df3['time_mean_prop'].values
Y_bp = df3['time_mean_bp'].values



# -- Figure
if SHOW_FIG:
    fig_filename = '{}.pdf'.format(filename)

    params = {'backend': 'pdf',
              'lines.linewidth': 4,
              'font.size': 10,
              'font.family': 'sans-serif',
              'font.sans-serif': [u'Arial', u'Liberation Sans'],
              'axes.labelsize': 20,  # fontsize for x and y labels (was 10, 24)
              'axes.titlesize': 16,     # 22
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'figure.figsize': [4, 4],
              'grid.color': '0.4',  # grid color
              'legend.fontsize': 12,    # 16
              'xtick.major.pad': 2,  # padding of tick labels: default = 4
              'ytick.major.pad': 1,  # padding of tick labels: default = 4
              'xtick.direction': 'out',  # default: 'in'
              'ytick.direction': 'out',  # default: 'in'
              }
    mpl.rcParams.update(params)

    fig = figure()
    ax = fig.add_axes([0.13, 0.17, 0.8, 0.8])

    C = (sns.color_palette("colorblind", 3))  # color palette from seaborn
    C[0] = 'blue'
    C[2] = C[1]
    C[1] = 'orange'

    # -- Draw the plots
    ax.plot([1, 1e7], [1e-5, 100], linewidth=1, color='gray', linestyle='dashed', label='1sec/100k edges', clip_on=True,
            zorder=4)
    if SHOW_GRAPH_TIME:
        ax.plot(X, Y_graph, linewidth=2, color='black', linestyle='solid', label='graph', clip_on=True)
    ax.plot(X, Y_bp, linewidth=4, color=C[1], linestyle='solid', label='BP (10 iterations)', clip_on=True,
            marker='o', markersize=6, markeredgewidth=0, zorder=4)
    ax.plot(X, Y_prop, linewidth=6, color=C[0], linestyle='solid', label='Lin (10 iterations)', clip_on=True,
            marker='o', markersize=8, markeredgewidth=0, zorder=4)
    ax.plot(X, Y_eps_max, linewidth=4, color=C[2], linestyle='dashed', label=r'Estimating $\epsilon_{{\mathrm{{max}}}}$', clip_on=True,
            marker='o', markersize=6, markeredgewidth=0, zorder=5)

    plt.xscale('log')
    plt.yscale('log')

    # -- Title and legend
    handles, labels = ax.get_legend_handles_labels()
    legend = plt.legend(handles, labels,
                        loc='upper left',     # 'upper right'
                        handlelength=2,
                        labelspacing=0,  # distance between label entries
                        handletextpad=0.3,  # distance between label and the line representation
                        borderaxespad=0.2,  # distance between legend and the outer axes
                        borderpad=0.3,  # padding inside legend box
                        numpoints=1,  # put the marker only once
                        )
    legend.set_zorder(3)
    frame_legend = legend.get_frame()
    frame_legend.set_linewidth(0.0)
    frame_legend.set_alpha(0.8)  # 0.8

    ax.yaxis.set_ticks_position('left')         # ticks only left and bottom
    ax.xaxis.set_ticks_position('bottom')

    frame = plt.gca()
    frame.tick_params(direction='out', width=1, )   # length=10

    # -- Figure settings and save
    grid(b=True, which='major', axis='both', alpha=0.2, linestyle='solid', linewidth=1, zorder=1)  # linestyle='dashed', which='minor', axis='y',
    grid(b=True, which='minor', axis='both', alpha=0.2, linestyle='solid', linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',
    xlabel(r'Number of edges ($m$)', labelpad=0)          # labelpad=0
    ylabel(r'Time [sec]', labelpad=0)
    xlim(xmin, xmax)
    ylim(ymin, ymax)

    savefig(join(figure_directory, fig_filename), format='pdf',
            dpi=None,
            edgecolor='w',
            orientation='portrait',
            transparent=False,
            bbox_inches='tight',
            pad_inches=0.05,
            frameon=None)

    os.system('open "' + join(figure_directory, fig_filename) + '"')  # shows actually created PDF
