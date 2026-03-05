# Ai_proyect2
Second proyect for ai class

##### to verify python version run:
 python --version or py --version or python3 --version then input version on -m venv.venv and run all other code

# este es para Mac y Linux

python3 -m venv.venv
source .venv/bin/activate
pip install -r requirements.txt

# windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Ai_proyect2
Second project for AI class.

## Setup the Python environment

First verify Python is installed:

python --version
or
python3 --version

Python 3.9+ is recommended.

---

## macOS / Linux

Create the virtual environment:

python3 -m venv .venv

Activate the environment:

source .venv/bin/activate

Install project dependencies:

pip install -r requirements.txt

---

## Windows (PowerShell)

Create the virtual environment:

python -m venv .venv

Activate the environment:

.\.venv\Scripts\Activate.ps1

Install project dependencies:

pip install -r requirements.txt

## After activation

Verify the environment is active:

python -c "import sys; print(sys.executable)"

It should point to `.venv`.



#### Coworker install steps (put this in your README)

 clone the repo, then in PowerShell:

cd path\to\Ai_proyect2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt