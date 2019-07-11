[![Build Status](https://travis-ci.org/ihmeuw/dismod_mr.svg?branch=master)](https://travis-ci.org/ihmeuw/dismod_mr)

Introduction
============

This project is the descriptive epidemiological meta-regression tool,
DisMod-MR, which grew out of the Global Burden of Disease (GBD) Study
2010.  DisMod-MR has been developed for the Institute of Health
Metrics and Evaluation at the University of Washington from 2008-2019.

Examples
========

[A motivating example: descriptive epidemiological meta-regression of Parkinson's Disease](http://nbviewer.ipython.org/github/ihmeuw/dismod_mr/blob/master/examples/pd_sim_data.ipynb)

All examples: http://nbviewer.ipython.org/github/ihmeuw/dismod_mr/tree/master/examples/

Installation
============

With conda and pip
------------------

Install [conda, if
necessary](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)).

Setup a conda environment if desired (recommended, but not required):
```
conda create --name=dismod_mr python=3.6
conda activate dismod_mr
```

Install PyMC 2 with conda (at the time of writing, there is something wrong with the pip installer for PyMC 2):
```
conda install pymc
```

Install DisMod-MR with pip:
```
pip install dismod_mr
```


From Source
-----------

Setup a conda environment (after [installing conda, if
necessary](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)):

```
conda create --name=dismod_mr python=3.6
conda activate dismod_mr
conda install pymc
git clone git@github.com:ihmeuw/dismod_mr.git
cd dismod_mr
pip install -U -e .
```




Coding Practices
================

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
