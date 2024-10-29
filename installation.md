# Installation
We have supplied the dependencies for Homework 1 in the requirements.txt file. You have a few options when installing the required packages.

# Installing with conda:

The installation instructions for conda and miniconda can be found [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).  After installing conda and calling `conda init`, you can create an environment and install the dependencies with the following three commands:

	1) conda create -n pa230 python=3.9
	2) conda activate pa230
	3) pip install -r requirements.txt

Note that you will need to activate this environment anytime you want to run the code.


# Installing via venv:

You can create a virtual python environment and install the dependencies there:
First install virtualenv if you don't have it yet, then create a virtual
environment called pa230 and activate it.

	1) pip install virtualenv
	2) python -m venv pa230
	3) source pa230/bin/activate

You can check that the environment was activated by calling

	which python

Then just install the requirements like in the previous step

	pip install -r requirements.txt


# System wide installation:
The last option is to just install all the dependencies system-wide without creating a virtual environment.
Note that this way you may get some dependency conflicts.

	1) pip install -r requirements.txt
			


