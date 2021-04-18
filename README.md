# 239AS_T99_RL_Agent
Repo for 239AS course project

Short instructions on installation:

1. Create a python 3.8 virtual environment for the project. If using conda, run
	conda create -p name_of_environment python=3.8

2. Add virtual environment to jupyter notebook search list. If using conda, run
	pip install --user ipykernel
	python -m ipykernel install --user --name=name_of_environment

3. Activate environment
	conda activate ./venv_name

4. Install libraries
	pip install -r requirements.txt
   important note: make sure that all libraries you are installing are listed in requirements.txt. This will help others to avoid issues while running your code.
