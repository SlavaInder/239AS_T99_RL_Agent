# 239AS_T99_RL_Agent

Repo for 239AS course project
## Short instructions on installation:
---  
### 1. Create a python 3.8 virtual environment for the project. If using conda, run:
````
conda create -p name_of_environment python=3.8
````

### 2. Add virtual environment to jupyter notebook search list. If using conda, run:
````
pip install --user ipykernel python -m ipykernel install --user --name=name_of_environment
````

### 3. Activate environment
````
conda activate ./venv_name
````
### 4. Install libraries
````
pip install -r requirements.txt
````
<ins>Important note</ins>: make sure that all libraries you are installing are listed in `requirements.txt`. This will help others to avoid issues while running your code.
## Task List
---
#### Current tasks:
1. Write a default code for creating SimpleDQN - a fully connected NN, for which we can set the number and the width of layers arbitry. Return Logits for actions
2. Create simple training cycle: playing until getting signal to stop, then train from randomly selected samples.
3. Create a code for classic tetris. Code needs to be expandable. Work in cooperation with 4.
4. Create a visualisation for classic tetris. Code needs to be expandable. Work in cooperation with 3.
