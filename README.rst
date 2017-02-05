SSLH (Semi-Supervised Learning with Heterophily)
================================================


Home of ``SSLH`` on github:
`http://github.com/sslh/sslh/ <http://github.com/sslh/sslh/>`__


Documentation
-------------

This library implements efficient algorithms in linear algebra
for solving various inference and estimation problems
in networks with observed heteorphily between classes of nodes (Heterophily: "Opposites attract" vs. Homophily: "Birds of a feather flock together").
The technical framework is that of undirected graphical models (Markov Random Fields or Markov Networks).
The key idea is that after applying certain linearization assumptions (that change the semantics) the resulting formulations
allow several orders of magnitude speed-up in calculation.

The methods are described in detail in the following papers:

1. `Linearized and Single-Pass Belief Propagation <http://www.vldb.org/pvldb/vol8.html>`__. `Wolfgang Gatterbauer <http://gatterbauer.co>`__, `Stephan Günnemann <http://www.cs.cmu.edu/~sguennem/>`__, `Danai Koutra <http://web.eecs.umich.edu/~dkoutra/>`__, `Christos Faloutsos <http://www.cs.cmu.edu/~christos/>`__. PVLDB 8(5): 581-592 (2015). [`Paper (PDF) <http://www.vldb.org/pvldb/vol8/p581-gatterbauer.pdf>`__], [`Full version (PDF) <http://arxiv.org/pdf/1406.7288>`__]

2.  `Semi-Supervised Learning with Heterophily <http://arxiv.org/abs/1412.3100>`__. `Wolfgang Gatterbauer <http://gatterbauer.co>`__ [`Working paper (PDF) <http://arxiv.org/pdf/1412.3100>`__]


Usage & Documentation
---------------------

The package consists of:

1. A directory ``sslh`` that contains files with the main methods

2. A directory ``test`` that contains the test files, each of which makes use of methods from the corresonding file in the ``sslh`` directory.

Thus ideally take a look in the ``test`` directory, run some files and look through the annotations in the files.


Installation
------------

The latest version of SSLH can be installed from the master branch using pip:

.. code:: bash

    pip install sslh

or

.. code:: bash

    pip install git+https://github.com/wolfandthegang/sslh/

Another option is to clone the repository and install SSLH using ``python setup.py install`` or ``python setup.py develop``.



Dependencies
------------

SSLH is tested on Python 2.7 and depends on NumPy, SciPy, Sklearn, and PyAMG (see setup.py for version information).


Related initiatives
-------------------

Sklearn: includes methods for semi-supervised learning (assuming homophily): http://scikit-learn.org/stable/modules/label_propagation.html

PyPGMc: focusing on directed graphical models https://github.com/kadeng/pypgmc/

OpenGM: implementation in C++ http://hci.iwr.uni-heidelberg.de/opengm2/, https://github.com/opengm/opengm


--------------

License
-------
Copyright 2015 Wolfgang Gatterbauer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.

Distributed in the hope that it will be useful to other researchers,
however, unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0

Contact Me
----------

Questions or comments about ``SSLH``? Drop me an email at
gatt@cmu.com.

--------------

Changelog
=========

Version 0.1.0
-------------

-  **Initial Release**: Main method 'linBP_undirected' for linearized belief propagation with one single doubly stochastic and symmetric potential as described in "Linearized and Single-pass Belief Propagation"

Version 0.2.0
-------------

-  Linearized BP for networks with one single aribtrary potentials as described in "Linearization for Pairwise MRFs."

