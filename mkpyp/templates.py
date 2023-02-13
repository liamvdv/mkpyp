import io
import sys
from datetime import date
from pathlib import Path
from string import Template
from typing import Callable, Literal, Optional, TypedDict, get_args

from pydantic import BaseModel
from pydantic.dataclasses import dataclass


@dataclass
class Author:
    name: str
    email: str


@dataclass
class Dependency:
    name: str
    op: Literal[">=", "==", "<=", ""]  # two sided possible?
    version: Optional[str]

    def __str__(self) -> str:
        return f"'{self.name}{self.op or ''}{self.version or ''}'"

    @classmethod
    def from_string(cls, s: str) -> Optional["Dependency"]:
        s = s.strip()
        if not s:
            return None
        for operator in get_args(cls.__annotations__.get("op")):
            if operator == "":
                continue
            pkg_name, op, version = s.partition(operator)
            if op == operator:
                return cls(pkg_name, op, version)  # type: ignore[call-arg] # dataclasses
        return cls(s, "", "")  # type: ignore[call-arg] # dataclasses

    @classmethod
    def all_from_string(cls, line: str) -> list["Dependency"]:
        deps = [cls.from_string(slug) for slug in line.split(" ")]
        return [dep for dep in deps if dep]  # smile


class AnyProps(TypedDict):
    pass


template_pyproject_toml = Template(
    """
[build-system]
requires = ['hatchling']
build-backend = 'hatchling.build'

[tool.hatch.version]
path = '$NAME/version.py'

[tool.hatch.build.targets.sdist]
# limit which files are included in the sdist (.tar.gz) asset
include = [
    '/README.md',
    '/Makefile',
    '/$NAME',
    '/tests',
    '/requirements',
]

[project]
name = '$NAME'
description = '$DESCRIPTION'
authors = [$AUTHORS]
license = {file = 'LICENSE'}
readme = 'README.md'
classifiers = [
    'Development Status :: 3 - Alpha',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3 :: Only',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Internet',
]
requires-python = '$REQUIRES_PYTHON_VERSION'
# DEP: Define dependencies here. Also see setup.py to support GitHub metadata indexing.
dependencies = [$DEPENDENCIES]
# OPTDEP: Define opt-in dependencies here. Also grep over the code base to see other places that might need change.
#         Example: With 
#               optional-dependencies = { slug = ['python-dotenv>=0.10.4'] }
#         Users can install $NAME[slug] to also install python-dotenv
optional-dependencies = { }
dynamic = ['version']

[project.urls]
Homepage = '$HOMEPAGE_URL'
Documentation = '$DOCUMENTATION_URL'
Source = '$SOURCE_URL'
Changelog = '$CHANGELOG_URL'

[tool.pytest.ini_options]
testpaths = 'tests'
filterwarnings = [
    'error',
]

[tool.ruff]
line-length = 120
extend-select = ['Q', 'RUF100', 'C90']
flake8-quotes = {inline-quotes = 'double', multiline-quotes = 'double'}
mccabe = { max-complexity = 14 }

[tool.ruff.per-file-ignores]

[tool.black]
color = true
line-length = 120
target-version = ['py310'] # default
# skip-string-normalization = true


[tool.isort]
line_length = 120
known_first_party = '$NAME'
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
combine_as_imports = true

[tool.mypy]
python_version = '$PYTHON_VERSION'
show_error_codes = true
follow_imports = 'silent'
strict_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
disallow_any_generics = true
check_untyped_defs = true
no_implicit_reexport = true
warn_unused_configs = true
disallow_subclassing_any = true
disallow_incomplete_defs = true
disallow_untyped_decorators = true
disallow_untyped_calls = true

# for strict mypy
disallow_untyped_defs = true

[tool.coverage.run]
source = ['$NAME']
branch = true
# no context set

[tool.coverage.report]
precision = 2
exclude_lines = [
    'pragma: no cover',
    'raise NotImplementedError',
    'raise NotImplemented',
]
""".strip()
)


class PyprojectTomlProps(TypedDict):
    name: str
    description: str
    authors: list[Author]
    homepage_url: str
    source_url: str
    documentation_url: str
    changelog_url: str
    # dependencies refers only the the dependencies needed to install the library
    # and does not include the developer tooling / dependencies
    dependencies: list[Dependency]
    python_version: str


def pyproject_toml(out: io.TextIOBase, props: PyprojectTomlProps) -> None:
    NAME = props["name"]
    AUTHORS = ",\n\t".join(
        [
            "{{name = '{name}', email = '{email}'}}".format(name=author.name, email=author.email)
            for author in props["authors"]
        ]
    )
    HOMEPAGE_URL = props["homepage_url"]
    SOURCE_URL = props["source_url"]
    DOCUMENTATION_URL = props["documentation_url"]
    CHANGELOG_URL = props["changelog_url"]
    PYTHON_VERSION = props["python_version"]
    DESCRIPTION = props["description"]
    REQUIRES_PYTHON_VERSION = f">={props['python_version']}"
    DEPENDENCIES = ", ".join(str(dependency) for dependency in props["dependencies"])

    raw = template_pyproject_toml.substitute(
        NAME=NAME,
        AUTHORS=AUTHORS,
        HOMEPAGE_URL=HOMEPAGE_URL,
        SOURCE_URL=SOURCE_URL,
        PYTHON_VERSION=PYTHON_VERSION,
        DOCUMENTATION_URL=DOCUMENTATION_URL,
        CHANGELOG_URL=CHANGELOG_URL,
        DESCRIPTION=DESCRIPTION,
        REQUIRES_PYTHON_VERSION=REQUIRES_PYTHON_VERSION,
        DEPENDENCIES=DEPENDENCIES,
    )
    out.write(raw)


template_setup_py = Template(
    """
import sys

sys.stderr.write(\"""
===============================
Unsupported installation method
===============================
$NAME no longer supports installation with `python setup.py install`.
Please use `python -m pip install .` instead.
\"""
)
sys.exit(1)


# The below code will never execute, however GitHub is particularly
# picky about where it finds Python packaging metadata.
# See: https://github.com/github/feedback/discussions/6456
#
# To be removed once GitHub catches up.

setup(
    name='$NAME',
    # DEP: pyproject.toml is the authorative source for dependencies. 
    #      Update the line below to support GitHub metadata indexing. 
    install_requires=[$DEPENDENCIES],
)
""".strip()
)


class SetupPyProps(TypedDict):
    name: str
    dependencies: list[Dependency]


def setup_py(out: io.TextIOBase, props: SetupPyProps) -> None:
    DEPENDENCIES = ", ".join(str(dependency) for dependency in props["dependencies"])
    NAME = props["name"]

    raw = template_setup_py.substitute(DEPENDENCIES=DEPENDENCIES, NAME=NAME)
    out.write(raw)


class LicenseProps(TypedDict):
    license_type: Literal["custom", "MIT"]
    authors: list[Author]


template_mit_license = Template(
    """
MIT License

Copyright (c) $YEARSPAN $AUTHORS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""".strip()
)

template_custom_license = "TODO(all): https://choosealicense.com/"


def license(out: io.TextIOBase, props: LicenseProps) -> None:
    AUTHORS = ", ".join([author.name for author in props["authors"]]) + " and other contributors"
    # FEATURE(liamvdv): inform users that they may need to update their license:
    #                   Either YYYY - YYYY or just YYYY format.
    YEARSPAN = str(date.today().year)
    if props["license_type"] == "MIT":
        raw = template_mit_license.substitute(AUTHORS=AUTHORS, YEARSPAN=YEARSPAN)
    elif props["license_type"] == "custom":
        raw = template_custom_license
    else:
        raise ValueError(f"unknown license_type '{props['license_type']}'")
    out.write(raw)


template_version_py = Template(
    """
__all__ = "VERSION", "version_info"

VERSION = "$VERSION"

# OPTDEP: Add optional dependencies for your users here, e. g. "devtools", "typing-extensions".
opt_in_dependencies: list[str] = []

def version_info() -> str:
    import platform
    import sys
    from importlib import import_module
    from pathlib import Path

    optional_deps = []
    for p in opt_in_dependencies:
        try:
            import_module(p.replace("-", "_"))
        except ImportError:
            continue
        optional_deps.append(p)

    info = {
        "pydantic version": VERSION,
        "install path": Path(__file__).resolve().parent,
        "python version": sys.version,
        "platform": platform.platform(),
        "optional deps. installed": optional_deps,
    }
    return "\\n".join("{:>30} {}".format(k + ":", str(v).replace("\\n", " ")) for k, v in info.items())
""".strip()
)


class VersionPyProps(TypedDict):
    version: str
    # optional dependencies to be added by user.


def version_py(out: io.TextIOBase, props: VersionPyProps) -> None:
    VERSION = str(props["version"])

    raw = template_version_py.substitute(VERSION=VERSION)
    out.write(raw)


# NOTE(liamvdv): $$() is NOT Makefile syntax. Double dollar sign is the escape sequence for string.Template
template_makefile = Template(
    """
sources = $NAME tests

.PHONY: bump-minor
bump-minor:
\techo "Not Implemented Yet"

.PHONY: bump-major
bump-major:
\techo "Not Implemented Yet"

.PHONY: install
install:
\tpython -m pip install -U pip
\tpip install -r requirements/all.txt
\tpip install -e .

.PHONY: refresh-requirements
refresh-requirements:
\t@echo "Updating requirements/*.txt files using pip-compile"
\tfind requirements/ -name '*.txt' ! -name 'all.txt' -type f -delete 
\tpip-compile -q --resolver backtracking -o requirements/linting.txt requirements/linting.in
\tpip-compile -q --resolver backtracking -o requirements/testing.txt requirements/testing.in
\tpip-compile -q --resolver backtracking -o requirements/pyproject.txt pyproject.toml\t
\t# OPTDEP: make sure to also recompile requirements for optional dependencies, e. g. for slug
\t# pip-compile -q --resolver backtracking -o requirements/pyproject+slug.txt pyproject.toml --extra=slug

\tpip install --dry-run -r requirements/all.txt

.PHONY: format
format:
\tisort $$(sources)
\tblack $$(sources)

.PHONY: lint
lint:
\truff $$(sources)
\tisort $$(sources) --check-only --df
\tblack $$(sources) --check --diff

.PHONY: mypy
mypy:
\tmypy $$(sources) --disable-recursive-aliases

.PHONY: test
test:
\tcoverage run -m pytest --durations=10

.PHONY: testcov
testcov: test
\t@echo "building coverage html"
\t@coverage html 

.PHONY: all
all: lint mypy testcov

.PHONY: clean
clean:
\trm -rf `find . -name __pycache__`
\trm -f `find . -type f -name '*.py[co]'`
\trm -f `find . -type f -name '*~'`
\trm -f `find . -type f -name '.*~'`
\trm -rf .cache
\trm -rf .pytest_cache
\trm -rf .mypy_cache
\trm -rf .ruff_cache
\trm -rf *.egg-info
\trm -rf build
\trm -rf dist
\trm -rf coverage.xml
\trm -f .coverage
\trm -f .coverage.*
\trm -rf htmlcov
""".strip()
)


class MakefileProps(TypedDict):
    name: str


def makefile(out: io.TextIOBase, props: MakefileProps) -> None:
    NAME = props["name"]

    raw = template_makefile.substitute(NAME=NAME)
    out.write(raw)


template_pre_commit_config_yaml = Template(
    """
repos:
- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v4.3.0
  hooks:
  - id: check-yaml
    args: ['--unsafe']
  - id: check-toml
  - id: end-of-file-fixer
  - id: trailing-whitespace

- repo: local
  hooks:
  - id: lint
    name: Lint
    entry: make lint
    types: [python]
    language: system
    pass_filenames: false
  - id: mypy
    name: Mypy
    entry: make mypy
    types: [python]
    language: system
    pass_filenames: false
""".strip()
)
# TODO: coverage tool


def pre_commit_config_yaml(out: io.TextIOBase, props: AnyProps) -> None:
    del props  # Unused.
    raw = template_pre_commit_config_yaml.substitute()
    out.write(raw)


template_gitignore = Template(
    """
env/
venv/
.venv/
env3*/
Pipfile
*.lock
*.py[cod]
*.egg-info/
.python-version
/build/
dist/
.cache/
.mypy_cache/
.pytest_cache/
.coverage
/htmlcov/
.vscode/
_build/
$NAME/*.c
$NAME/*.so
.auto-format
/codecov.sh
/worktrees/
/.ruff_cache/
""".strip()
)


class GitignoreProps(TypedDict):
    name: str


def gitignore(out: io.TextIOBase, props: GitignoreProps) -> None:
    NAME = props["name"]
    raw = template_gitignore.substitute(NAME=NAME)
    out.write(raw)


# Note that ./pyproject is generated by pip-compile
template_requirements_all_txt = Template(
    """
-r ./pyproject.txt
-r ./linting.txt
-r ./testing.txt
""".strip()
)


def requirements_all_txt(out: io.TextIOBase, props: AnyProps) -> None:
    del props  # Unused.
    raw = template_requirements_all_txt.substitute()
    out.write(raw)


# pip-compile --output-file=requirements/linting.txt --resolver=backtracking requirements/linting.in
template_requirements_linting_in = Template(
    """
# Only specify development requirements here.
pre-commit
black
isort
ruff
mypy
""".strip()
)


def requirements_linting_in(out: io.TextIOBase, props: AnyProps) -> None:
    del props  # Unused.
    raw = template_requirements_linting_in.substitute()
    out.write(raw)


template_requirements_testing_in = Template(
    """
# Only specify testing requirements here.
coverage[toml]
pytest
""".strip()
)


def requirements_testing_in(out: io.TextIOBase, props: AnyProps) -> None:
    del props  # Unused.
    raw = template_requirements_testing_in.substitute()
    out.write(raw)


# TODO(liamvdv)
template_readme_md = Template(
    """
# $NAME
$DESCRIPTION

## Installation


## Developer Installation
Follow the below steps are cloning or creating a new git repository with `git init`.
Make sure you are in the root directory of your project.
```shell
# create a new virtual environment (e. g. venv)
python3 -m venv .venv

# activate the virtual environment
source .venv/bin/activate

# install pip-compile for auto-generating requirement files
pip install pip-tools

# generate requirement files
make refresh-requirements

# install the requirements and install '$NAME' as editable package
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
Build the project with `hatch build`.
Now run `hatch publish --repo test` to upload the package to `test.pypi.org`.
Use `hatch publish --repo main` to upload to the production PyPI.
Define custom targets as per defined [here](https://hatch.pypa.io/latest/publish/#repository).
""".strip()
)


class ReadmeMdProps(TypedDict):
    name: str
    description: str


def readme_md(out: io.TextIOBase, props: ReadmeMdProps) -> None:
    NAME = props["name"]
    DESCRIPTION = props["description"]

    raw = template_readme_md.substitute(NAME=NAME, DESCRIPTION=DESCRIPTION)
    out.write(raw)


class TemplateProps(  # type: ignore[misc]
    PyprojectTomlProps,
    SetupPyProps,
    LicenseProps,
    GitignoreProps,
    MakefileProps,
    VersionPyProps,
    ReadmeMdProps,
):
    pass


class TemplateFile(BaseModel):
    tempalte_spec = "0.1"
    props: TemplateProps

    class Meta:
        extra = "ignore"


def generate_file(path: Path, templator: Callable[[io.TextIOBase, object], None], props: TemplateProps) -> None:
    with path.open("x", encoding="utf-8") as file:
        templator(file, props)


def generate_output(path: Path, templator: Callable[[io.TextIOBase, object], None], props: TemplateProps) -> None:
    print("=" * 80)
    print(f"Writing {path}:")
    print("-" * 80)
    templator(sys.stdout, props)  # type: ignore[arg-type]
    print()
