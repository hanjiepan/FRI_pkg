Examples
========

This folder contains sample notebooks using the components of the repository.

Overview
--------

- dirac_time_nonuniform.ipynb: Recover a signal in the form of a periodic stream of Diracs from non-uniformly spaced time samples (see Figure 6 in paper).

Run remotely
------------

To run the contents of this folder without going through any installation troubles, we have prepared the following mybinder instance for you. Note that changes made to this notebook will be lost when you close the instance.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hanjiepan/FRI_pkg/master)


Run locally
-----------

If you would like to have your own local copy of the notebooks (so that you can keep track of your changes for example), you can clone this repository and install the requirements on your local machine.   

We recommend to use a virtual environment for the dependencies. For example, on a linux machine using python's `virtualenv`:

```bash

# change to a directory of your choice.
cd <folder-of-your-choice>

# clone this repository.
git clone https://github.com/hanjiepan/FRI_pkg
cd FRI_pkg

# install virtualenv (might need option --user)
pip install virtualenv

# create and activate new environment at location of your choice.
virtualenv create <environment-folder>
source <environment-folder>/bin/activate

# install requirements of this repository.
pip install -r requirements.txt

```

To launch the notebook server it then suffices to run `jupyter notebook`.
