from __future__ import print_function, unicode_literals
from typing import Callable, get_args, Optional
import templates  # todo: make full path
import os
import sys
from pathlib import Path
import string
import fire
from InquirerPy import prompt
from InquirerPy.validator import EmptyInputValidator
import subprocess
from pydantic import BaseModel

# TODO(liamvdv): start actually supporting automatic ChangeLog generation.


def get_git_config() -> dict:
    data = {}
    res = subprocess.run(["git", "config", "--list"], stdout=subprocess.PIPE)
    git_user = res.stdout.strip().decode()
    for line in git_user.splitlines():
        k, _, v = line.partition("=")
        if v:
            data[k] = v
    return data

def get_python_version() -> str:
    """returns Major.Minor Python version, e. g. 3.10"""
    mmp = sys.version.split(" ")[0]
    mm = ".".join(mmp.split(".", 3)[:2])
    return mm

name_alphabet = string.ascii_lowercase + string.digits + "_-"


def name_validator(name: str) -> bool:
    return (
        len(name) > 0
        and name[0] in string.ascii_lowercase
        and all([char in name_alphabet for char in name])
    )


def promp_user() -> templates.TemplateProps:
    git_config = get_git_config()

    questions = [
        {
            "type": "input",
            "name": "name",
            "message": "Project Name:",
            "validate": name_validator,
            "invalid_message": f"invalid python name: must start with a lowercase letter and only contain [{name_alphabet}]",
        },
        {
            "type": "input",
            "name": "description",
            "message": "Project Description:",
        },
        {
            "type": "input",
            "name": "version",
            "message": "Project Version:",
            "default": "0.1.0",
            "validate": lambda v: semantic_version_validator(v, 2),
            "invalid_message": "must be Major.Minor.Patch all digits"
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
            "name": "python_version",
            "message": "Python Version:",
            "default": lambda _: get_python_version(),
            "validate": lambda v: semantic_version_validator(v, 1),
            "invalid_message": "must be Major.Minor all digits"
        },
        {
            "type": "input",
            "name": "dependencies",
            "message": "Dependencies (space separated):",
            "default": "",
        },
        {
            "type": "list",
            "name": "license_type",
            "multiselect": False,
            "choices": get_args(
                templates.TemplateProps.__annotations__.get("license_type")
            ),
            "message": "License:",
            "default": "MIT",
        },
        {
            "type": "input",
            "name": "source_url",
            "message": "Source Url:",
            "default": f"https://github.com/",
        },
        {
            "type": "input",
            "name": "homepage_url",
            "message": "Homepage Url:",
            "default": lambda result: result.get("source_url", ""),
        },
        {
            "type": "input",
            "name": "documentation_url",
            "message": "Documentation Url:",
            "default": lambda result: result.get("source_url", ""),
        },
        {
            "type": "input",
            "name": "changelog_url",
            "message": "Changelog Url:",
            "default": lambda result: result.get("source_url", ""),
        },
    ]
    result = prompt(questions)

    name = result.pop("author.name", "")
    email = result.pop("author.email", "")
    result["authors"] = [templates.Author(name, email)]

    result["dependencies"] = templates.Dependency.all_from_string(
        result.pop("dependencies", "")
    )

    return result

def prompt_do_proceed(default=True) -> bool:
    questions = [{
        "type": "confirm",
        "message": "Proceed?",
        "name": "proceed",
        "default": default,
    }]
    result = prompt(questions)
    return result.get("proceed")

def semantic_version_validator(version: str, ndots: int) -> bool:
    assert 0 <= ndots <= 2, "version must be either Major.Minor.Patch or Major.Minor or Major" 
    parts = version.split(".")
    return len(parts) == ndots + 1 and all(part.isnumeric() for part in parts)


class Action:
    func: Callable
    args: list
    kwargs: dict
    children: list["Action"]

    def __init__(
        self, func: Callable, *args: list, children: list["Action"] = None, **kwargs
    ):
        if children is None:
            children = []
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.children = children

    def then(self, *action: "Action"):
        self.children.extend(action)
        return self

    def execute(self):
        # todo: add prefixed logger with msg=log.stack+{f.__name__}:{msg}
        self.func(*self.args, **self.kwargs)
        for action in self.children:
            action.execute()


class Mkpyp(object):
    """
        Generates a new idiomatic Python project for linux/macos:
        project structure - .gitignore, LICENSE
        format - black, isort
        lint - ruff, mypy, black
        test - pytest, coverage
        other - pre-commit, Makefile, version.py, requirements
    """
    def __init__(self):
        pass

    def init(self, dry=False, infile: Optional[str]=None, outfile: Optional[str]=None):
        """
        prompt user input and generates project in subdirectory 
        infile - load from template.json
        outfile - write a template.json; do not generate
        dry - no sideeffects, print output to stdout
        """
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
                with open(outfile, "x", encoding="utf-8") as file:
                    file.write(raw)
        else:
            print(raw)
            if prompt_do_proceed():
                generate(Path.cwd(), file.props, dry)

def generate(pwd: Path, props: dict, testing: bool = True):
    if not pwd.exists():
        raise ValueError(f"parent directory does not exist: parent = {pwd}")
    name = str(props["name"])
    base = pwd / name
    req_base = base / "requirements"

    def mock_mkdir(path: Path, *args, **kwargs):
        del args, kwargs  # Unused.
        print("=" * 80, file=sys.stdout)
        print(f"Creating directory: {path}", file=sys.stdout)

    mkdir = os.mkdir if not testing else mock_mkdir
    filewriter = templates.generate_file if not testing else templates.generate_output

    mkfile = (
        lambda p: p.open("x").close()
        if not testing
        else lambda p: print(f"Creating file: {p}")
    )
    Action(mkdir, str(base)).then(
        Action(mkdir, str(base / name)).then(
            Action(mkfile, base / name / "main.py"),
            Action(mkfile, base / name / "__init__.py"),
            Action(filewriter, base / name / "version.py", templates.version_py, props),
        ),
        Action(mkdir, str(req_base)).then(
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
            # req_base / pyproject.txt is generated from pyproject.
            Action(
                filewriter, req_base / "all.txt", templates.requirements_all_txt, props
            ),
        ),
        Action(mkdir, str(base / "tests")),
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
    ).execute()


if __name__ == "__main__":
    fire.Fire(Mkpyp(), name="mkpyp")
