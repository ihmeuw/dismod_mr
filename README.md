Introduction
============

This project is the descriptive epidemiological meta-regression tool,
DisMod-MR, which grew out of the Global Burden of Disease (GBD) Study
2010.  DisMod-MR has been developed for the Institute of Health
Metrics and Evaluation at the University of Washington from 2008-2013.

Examples
========

[A motivating example: descriptive epidemiological meta-regression of Parkinson's Disease](http://nbviewer.ipython.org/github/ihmeuw/dismod_mr/blob/master/examples/pd_sim_data.ipynb)

All examples: http://nbviewer.ipython.org/github/ihmeuw/dismod_mr/tree/master/examples/

Installation
============

The file requirements.tex lists the Python packages necessary to run
DisMod-MR.  Unfortunately, PyCPPAD is not available with
pip/easy_install.

DisMod-MR has been installed successfully by at least one person outside of
my immediate peer group.  She says "I ended up using Anaconda and then
manually installing all the required packages individually. There were
also errors when I install a new package that required me to
re-install previous ones and restart the computer several times. Using
easyinstall was helpful. I also had to install GCC at some point. I
hope these would be helpful for making it easier for new users."

This installation section could be more detailed. Here is a little bit
more from the first successful install outside of IHME:

### For Macintosh OS X

I was able to set-up pycppad. One lesson learned was that using
pycppad with the Anaconda Python distribution makes the kernel die (at
least for OS X).

The method that I've found to be less prone to errors (at least for OS
X) is:

- Use the pre-installed python that comes with OS X (better to have
  other python distributions uninstalled)

- Install SciPy which contains most of the dependencies (and includes
  easy install if I remember correctly)

- Install the remaining dependencies using easy install (in my case,
  only simplejson I think)

For the consistent model:

- Install boost
  (http://www.boost.org/doc/libs/1_54_0/libs/python/doc/index.html)

- Install cppad (http://www.coin-or.org/CppAD/)

- Configure the setup.py file for pycppad to point to the directories
  and library files for boost and cppad

- Build and install pycppad

Other notes:

- I sometimes got errors about not finding a C compiler even though I
  had gcc installed; this was solved by making a symbolic link

PS Xcode and Command Line Tools should also be installed on OS X

### For Windows

I was able to set-up the age-standardising part of the model on a
Windows system. The installation had some differences from OS X. In a
nutshell:

- I was only able to make a 32-bit installation work for now.

- Some prerequisites: install *specifically* Visual Studio C++ 2008
  Express Edition (not the later versions!); also had to install gcc,
  gfortran, g++ via MinGW (then had to set path manually in the
  environment variables)

- Installed Python and set path manually in environment variables

- Had to use unofficial SciPy Stack installer from
  http://www.lfd.uci.edu/~gohlke/pythonlibs/

- Installed other remaining dependencies via command line (in my case,
  networkx and pymc)

- In the case of pymc, had to force to build using mingw32 but then
  had to install the built files using easy_install (that's just how I
  got it to work somehow)


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
