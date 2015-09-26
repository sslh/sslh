"""
Unit test with 'Pytest' that can also be used in 'setup.py'

First version: Sept 25, 2015
This version: Sept 25, 2015
Author: Wolfgang Gatterbauer <gatt@cmu.edu>
"""


# # -- Locate the directory of this file and switch the path of execution
# Source: http://stackoverflow.com/questions/50499/how-do-i-get-the-path-and-name-of-the-file-that-is-currently-executing/
# from os.path import abspath, dirname
# from inspect import getfile, currentframe
# my_path = dirname(abspath(getfile(currentframe())))
# from os import chdir
# chdir(my_path)


# -- Actual pytest
import pytest
pytest.main("-v -x --pyargs test --doctest-glob='*.rst'")
    # --pyargs test: only tests below subdirectory 'test'
    # --doctest-glob='*.rst': ignore 'test*.txt' files
