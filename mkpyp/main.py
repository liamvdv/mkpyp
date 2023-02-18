from __future__ import print_function, unicode_literals

import re
import string
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional, get_args

import fire  # type: ignore[import]
import InquirerPy
from InquirerPy.validator import EmptyInputValidator

from mkpyp import templates

# TODO(liamvdv): start actually supporting automatic ChangeLog generation.


def get_git_config() -> dict[str, str]:
    data = {}
    res = subprocess.run(["git", "config", "--list"], stdout=subprocess.PIPE)
    git_user = res.stdout.strip().decode()
    for line in git_user.splitlines():
        k, _, v = line.partition("=")
        if v:
            data[k] = v
    return data


def is_available_pypi(name: str, cache: dict[str, bool] = {}) -> bool:
    if name in cache:
        return cache[name]
    invalid_name_pattern = "ERROR: Invalid requirement:"
    not_found_pattern = "ERROR: No matching distribution found for"

    proc = subprocess.run(
        ["pip3", "install", name, "--dry-run", "--no-color", "--disable-pip-version-check"],
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
    )
    text = proc.stderr.strip().decode()
    if invalid_name_pattern in text:
        raise Exception("invalid pip name")
    cache[name] = not_found_pattern in text
    return cache[name]


def get_python_version() -> str:
    """returns Major.Minor Python version, e. g. 3.10"""
    mmp = sys.version.split(" ")[0]
    mm = ".".join(mmp.split(".", 3)[:2])
    return mm


def normalized_pypi_name(name: str) -> str:
    """normalize package name to project name as per PEP 508
    See https://packaging.python.org/en/latest/specifications/name-normalization/"""
    return re.sub(r"[-_.]+", "-", name).lower()


name_alphabet_pypi = string.ascii_lowercase + string.digits + "-"
# disallow dots for top level name
name_alphabet_identifer = string.ascii_lowercase + string.digits + "_"


def is_normalized_pypi(name: str) -> bool:
    return len(name) > 0 and all([char in name_alphabet_pypi for char in name])


def is_valid_identifier(name: str) -> bool:
    """package name must be a valid python identifier"""
    return (
        len(name) > 0 and name[0] in string.ascii_lowercase and all([char in name_alphabet_identifer for char in name])
    )


def promp_user() -> templates.TemplateProps:
    git_config = get_git_config()

    questions: list[dict[str, Any]] = [
        {
            "type": "input",
            "name": "package_name",
            "message": "Package Name:",
            "validate": is_valid_identifier,
            "invalid_message": (
                f"must be a valid python identifier, only include characters [{name_alphabet_identifer}]"
            ),
        },
        {
            "type": "input",
            "name": "pypi_name",
            "message": "Project Name (PyPI):",
            "default": lambda answers: normalized_pypi_name(answers.get("package_name", "")),
            "validate": is_normalized_pypi,
            "invalid_message": f"includes letters other than [{name_alphabet_pypi}] or PyPI name isn't free",
        },
        {
            "type": "confirm",
            "name": "-",
            "message": "PyPI Name is already taken. Continue anyway?",
            "when": lambda answers: not is_available_pypi(answers.get("pypi_name")),
            "default": False,
            "filter": lambda proceed: True if proceed else exit(0),
        },
        {
            "type": "input",
            "name": "description",
            "message": "Project Description:",
        },
        {
            "type": "input",
            "name": "dependencies",
            "message": "Dependencies (space separated):",
            "default": "",
        },
        {
            "type": "input",
            "name": "python_version",
            "message": "Python Version:",
            "default": lambda _: get_python_version(),
            "validate": lambda v: semantic_version_validator(v, 1),
            "invalid_message": "must be Major.Minor all digits",
        },
        {
            "type": "list",
            "name": "license_type",
            "multiselect": False,
            "choices": get_args(templates.TemplateProps.__annotations__.get("license_type")),
            "message": "License:",
            "default": "MIT",
        },
        {
            "type": "input",
            "name": "author.name",
            "message": "Your Name:",
            "validate": EmptyInputValidator(),
            "default": lambda _: git_config.get("user.name", ""),
        },
        {
            "type": "input",
            "name": "author.email",
            "message": "Your Email:",
            "default": lambda _: git_config.get("user.email", ""),
        },
        {
            "type": "input",
            "name": "gh_owner",
            "message": "GitHub Repo Owner:",
            "validate": EmptyInputValidator(),
        },
        {
            "type": "input",
            "name": "version",
            "message": "Project Version:",
            "default": "0.1.0",
            "validate": lambda v: semantic_version_validator(v, 2),
            "invalid_message": "must be Major.Minor.Patch all digits",
        },
        {
            "type": "input",
            "name": "source_url",
            "message": "Source Url:",
            "default": lambda answers: f"https://github.com/{answers.get('gh_owner')}/{answers.get('pypi_name', '')}",
        },
        {
            "type": "input",
            "name": "documentation_url",
            "message": "Documentation Url:",
            "default": lambda answers: f"https://{answers.get('gh_owner')}.github.io/{answers.get('pypi_name', '')}",
        },
        {
            "type": "input",
            "name": "homepage_url",
            "message": "Homepage Url:",
            "default": lambda answers: answers.get("documentation_url", ""),
        },
        {
            "type": "input",
            "name": "changelog_url",
            "message": "Changelog Url:",
            "default": lambda result: result.get("documentation_url", ""),
        },
    ]
    result = InquirerPy.prompt(questions)  # type: ignore[attr-defined]

    name = result.pop("author.name", "")
    email = result.pop("author.email", "")
    result["authors"] = [templates.Author(name, email)]  # type: ignore[call-arg]

    result["dependencies"] = templates.Dependency.all_from_string(result.pop("dependencies", ""))

    return result


def prompt_do_proceed(default: bool = True) -> bool:
    questions = [
        {
            "type": "confirm",
            "message": "Proceed?",
            "name": "proceed",
            "default": default,
        }
    ]
    result = InquirerPy.prompt(questions)  # type: ignore[attr-defined]
    return result.get("proceed")


def semantic_version_validator(version: str, ndots: int) -> bool:
    assert 0 <= ndots <= 2, "version must be either Major.Minor.Patch or Major.Minor or Major"
    parts = version.split(".")
    return len(parts) == ndots + 1 and all(part.isnumeric() for part in parts)


class Action:
    func: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    children: list["Action"]

    def __init__(self, func: Callable[..., Any], *args: Any, children: Optional[list["Action"]] = None, **kwargs: Any):
        if children is None:
            children = []
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.children = children

    def then(self, *action: "Action") -> "Action":
        self.children.extend(action)
        return self

    def execute(self) -> None:
        # todo: add prefixed logger with msg=log.stack+{f.__name__}:{msg}
        self.func(*self.args, **self.kwargs)
        for action in self.children:
            action.execute()


def mkpyp(*args: Any, dry: bool = False, infile: str = None, outfile: str = None) -> None:  # type: ignore[assignment]
    """
    generate idiomatic python projects in subdirectory

    Parameters
    ----------
    infile
        load from filepath
    outfile
        write to filepath; do not generate
    dry
        no sideeffects, print actions to stdout
    """
    if args:
        print("mkpyp does not support arguments")
        exit(1)
    print("Abort with ctrl + c\n")
    if isinstance(infile, str):
        file = templates.TemplateFile.parse_file(infile)
    else:
        props = promp_user()
        file = templates.TemplateFile(props=props)

    raw = file.json(indent=4)
    if isinstance(outfile, str):
        if dry:
            print(f"Writing {outfile}:")
            print("-" * 80)
            print(raw)
        else:
            with open(outfile, "x", encoding="utf-8") as f:
                f.write(raw)
    else:
        print(raw)
        if prompt_do_proceed():
            generate(Path.cwd(), file.props, dry)


def generate(pwd: Path, props: dict[str, Any], testing: bool = True) -> None:
    if not pwd.exists():
        raise ValueError(f"parent directory does not exist: parent = {pwd}")
    pypi_name = props["pypi_name"]
    package_name = props["package_name"]

    def mkdir(path: Path) -> None:
        if not testing:
            path.mkdir()
        else:
            print("=" * 80, file=sys.stdout)
            print(f"Creating directory: {path}", file=sys.stdout)

    def mkfile(path: Path) -> None:
        if not testing:
            path.open("x").close()
        else:
            print(f"Creating file: {path}")

    filewriter = templates.generate_output
    if not testing:
        filewriter = templates.generate_file

    base = pwd / pypi_name
    pkg_base = base / package_name
    req_base = base / "requirements"
    docs_base = base / "docs"
    gh_base = base / ".github"
    workflow_base = gh_base / "workflows"
    Action(mkdir, base).then(
        Action(mkdir, pkg_base).then(
            Action(mkfile, pkg_base / "main.py"),
            Action(mkfile, pkg_base / "__init__.py"),
            Action(filewriter, pkg_base / "version.py", templates.version_py, props),
        ),
        Action(mkdir, req_base).then(
            Action(
                filewriter,
                req_base / "linting.in",
                templates.requirements_linting_in,
                props,
            ),
            Action(
                filewriter,
                req_base / "testing.in",
                templates.requirements_testing_in,
                props,
            ),
            Action(filewriter, req_base / "docs.in", templates.requirements_docs_in, props),
            # req_base / pyproject.txt is generated from pyproject.
            Action(filewriter, req_base / "all.txt", templates.requirements_all_txt, props),
        ),
        Action(mkdir, base / "tests"),
        Action(filewriter, base / "pyproject.toml", templates.pyproject_toml, props),
        Action(filewriter, base / "setup.py", templates.setup_py, props),
        Action(filewriter, base / "README.md", templates.readme_md, props),
        Action(filewriter, base / "Makefile", templates.makefile, props),
        Action(filewriter, base / "LICENSE", templates.license, props),
        Action(filewriter, base / ".gitignore", templates.gitignore, props),
        Action(
            filewriter,
            base / ".pre-commit-config.yaml",
            templates.pre_commit_config_yaml,
            props,
        ),
        Action(filewriter, base / "mkdocs.yml", templates.mkdocs_yml, props),
        Action(mkdir, docs_base).then(
            Action(filewriter, docs_base / "index.md", templates.docs_index_md, props),
        ),
        Action(mkdir, gh_base).then(
            Action(mkdir, workflow_base).then(
                Action(filewriter, workflow_base / "ci.yml", templates.gh_workflows_ci_yml, props),
            )
        ),
    ).execute()


def run() -> None:
    fire.Fire(mkpyp, name="mkpyp")


if __name__ == "__main__":
    run()
