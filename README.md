# mkpyp
make python projects idiomatically
```
Generates a new idiomatic Python project for linux/macos:
project structure - .gitignore, LICENSE
format - black, isort
lint - ruff, mypy, black
test - pytest, coverage
other - pre-commit, Makefile, version.py, requirements
```

## Installation
Install with `pip3 install mypyp`. Run with `mypyp`.

## Developer Installation
After cloning this repo or initializing a new git repository with `git init`, complete the following steps at root directory.
```shell
# create a new virtual environment (e. g. venv)
python3 -m venv .venv

# activate the virtual environment
source .venv/bin/activate

# install pip-compile for auto-generating requirement files
pip install pip-tools

# generate requirement files
make refresh-requirements

# install the requirements and install 'mkpyp' as editable package
make install
```

## Dependency Management
The location of where dependencies are declared depends on their scope. 

- Package dependencies must be put into `pyproject.toml [project] .dependencies`.
- Opt-in dependencies must be put into `pyproject.toml [project] .optional-dependencies`.
- Testing dependencies must be put into `requirements/testing.in`.
- Linting dependencies must be put into `requirements/linting.in`.

We generate the requirements files with `make refresh-requirements`. Reinstall with `make install`.

## Publish
1. ensure version is correct in mkpyp/version.py
2. run