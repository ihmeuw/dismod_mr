.. image:: https://travis-ci.org/ihmeuw/dismod_mr.svg?branch=master
    :target: https://travis-ci.org/ihmeuw/dismod_mr
    :alt: Latest Version

============
Introduction
============

This project is the descriptive epidemiological meta-regression tool,
DisMod-MR, which grew out of the Global Burden of Disease (GBD) Study
2010.  DisMod-MR has been developed for the Institute of Health
Metrics and Evaluation at the University of Washington from 2008-2013.

.. contents::

Examples
--------

`A motivating example: descriptive epidemiological meta-regression of Parkinson's Disease <http://nbviewer.ipython.org/github/ihmeuw/dismod_mr/blob/master/examples/pd_sim_data.ipynb>`_

`All examples <http://nbviewer.ipython.org/github/ihmeuw/dismod_mr/tree/master/examples/>`_

Installation
------------

Dismod MR requires PyMC2 which does not play nicely with normal Python
installation tools.  Fortunately, ``conda`` has solved this issue for us.
So first you'll need to setup a conda environment
(after `installing conda, if necessary <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_)
and install ``pymc``.  Then you can install ``dismod_mr`` using ``pip``.

.. code-block:: sh

   conda create --name=dismod_mr python=3.6 pymc
   conda activate dismod_mr
   pip install dismod_mr

Installing from source
++++++++++++++++++++++

If you want to install ``dismod_mr`` locally in an editable mode, the
instructions are very similar.  We'll clone the repository and install it
from a local directory instead of using ``pip`` to grab it from the Python
package index.

.. code-block:: sh

   conda create --name=dismod_mr python=3.6 pymc
   conda activate dismod_mr
   git clone git@github.com:ihmeuw/dismod_mr.git
   cd dismod_mr
   pip install -e .

Coding Practices
----------------

* Write tests before code
* Write equations before tests

* Test quantitatively with simulation data
* Test qualitatively with real data
* Automate tests

* Use a package instead of DIY
* Test the package

* Optimize code later
* Optimize code for readability before speed

* `.py` files should be short, less than 500 lines
* Functions should be short, less than 25 lines
