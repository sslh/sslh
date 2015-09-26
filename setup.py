# file adapted from: https://github.com/jeffknupp/sandman/blob/develop/setup.py
# https://www.jeffknupp.com/blog/2013/08/16/open-sourcing-a-python-project-the-right-way/
# https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/


# from __future__ import print_function
from setuptools import setup, find_packages
import codecs
import os
import re


####################################################################

NAME = "sslh"
# PACKAGES = find_packages(exclude=["tests*"])
# PACKAGES = ["sslh", "test"]
PACKAGES = ["sslh"]
META_PATH = os.path.join("sslh", "__init__.py")
KEYWORDS = ["learning", "belief propagation", "classification"]
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Natural Language :: English",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 2.6",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: SQL",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering",
]
INSTALL_REQUIRES=['numpy>=1.9.1',
                  'scipy>=0.15.1',
                  # 'matplotlib>=1.4.2',
                  'psycopg2>=2.5.4',
                  'pyamg>=2.2.1',
                  # 'scikit-learn>=0.15.1',
                  # 'sklearn>=0.15.1',
                  'sklearn',
                  'pytest>=2.8.0',
                  ],

###################################################################


HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


META_FILE = read(META_PATH)


def find_meta(meta):
    """
    Extract __*meta*__ from META_FILE.
    """
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta),
        META_FILE, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


if __name__ == "__main__":
    setup(
        name=NAME,
        description='Techniques for Semi-Supervised Learning with Heterophily',
        long_description=read("README.rst"),
        license=find_meta("license"),
        url='http://github.com/wolfandthegang/sslh/',
        version=find_meta("version"),
        author=find_meta("author"),
        author_email=find_meta("email"),
        maintainer=find_meta("author"),
        maintainer_email=find_meta("email"),
        keywords=KEYWORDS,
        packages=PACKAGES,
        zip_safe=False,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        # include_package_data=True,          # ???? only if runtime relevant
    )
