# PIPPO

bla bla

# Setup information

In order to execute the script you need to:
1) Install Python 3.5+ and Anaconda, as suggested in Qiskit installation guide (https://qiskit.org/documentation/install.html)
2) Create a folder and make it a Python Virtualenv
3) Clone this repo in the same folder
4) Position in the folder and run from command line: `pip install -r requirements.txt`

# Run information

5) Run the script as `python task.py <Lmax> [<yellow_gate> <green_gate>]`. 
args must follow this convention:
    - At least one argument is required: it is the circuit length Lmax
    - Two arguments are optional (in case, they have to be both specified). They can take their values in (x, y, z), and are the rotation axes for the parametric gates in the yellow and green blocks respectively