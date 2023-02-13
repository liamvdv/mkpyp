sources = mkpyp tests

.PHONY: bump-minor
bump-minor:
	echo "Not Implemented Yet"

.PHONY: bump-major
bump-major:
	echo "Not Implemented Yet"

.PHONY: install
install:
	python -m pip install -U pip
	pip install -r requirements/all.txt
	pip install -e .

.PHONY: refresh-requirements
refresh-requirements:
	@echo "Updating requirements/*.txt files using pip-compile"
	find requirements/ -name '*.txt' ! -name 'all.txt' -type f -delete 
	pip-compile -q --resolver backtracking -o requirements/linting.txt requirements/linting.in
	pip-compile -q --resolver backtracking -o requirements/testing.txt requirements/testing.in
	pip-compile -q --resolver backtracking -o requirements/pyproject.txt pyproject.toml	
	# OPTDEP: make sure to also recompile requirements for optional dependencies, e. g. for slug
	# pip-compile -q --resolver backtracking -o requirements/pyproject+slug.txt pyproject.toml --extra=slug

	pip install --dry-run -r requirements/all.txt

.PHONY: format
format:
	isort $(sources)
	black $(sources)

.PHONY: lint
lint:
	ruff $(sources)
	isort $(sources) --check-only --df
	black $(sources) --check --diff

.PHONY: mypy
mypy:
	mypy $(sources) --disable-recursive-aliases

.PHONY: test
test:
	coverage run -m pytest --durations=10

.PHONY: testcov
testcov: test
	@echo "building coverage html"
	@coverage html 

.PHONY: all
all: lint mypy testcov

.PHONY: clean
clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]'`
	rm -f `find . -type f -name '*~'`
	rm -f `find . -type f -name '.*~'`
	rm -rf .cache
	rm -rf .ruff_cache
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf *.egg-info
	rm -rf build
	rm -rf dist
	rm -rf coverage.xml
	rm -f .coverage
	rm -f .coverage.*
	rm -rf htmlcov