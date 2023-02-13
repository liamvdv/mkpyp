from __future__ import print_function, unicode_literals

import os
import string
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Union, get_args

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


def get_python_version() -> str:
    """returns Major.Minor Python version, e. g. 3.10"""
    mmp = sys.version.split(" ")[0]
    mm = ".".join(mmp.split(".", 3)[:2])
    return mm


name_alphabet = string.ascii_lowercase + string.digits + "_-"


def name_validator(name: str) -> bool:
    return len(name) > 0 and name[0] in string.ascii_lowercase and all([char in name_alphabet for char in name])


def promp_user() -> templates.TemplateProps:
    git_config = get_git_config()

    questions: list[dict[str, Any]] = [
        {
            "type": "input",
            "name": "name",
            "message": "Project Name:",
            "validate": name_validator,
            "invalid_message": (
                "invalid python name: must start with a lowercase letter and only contain [{name_alphabet}]"
            ),
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
            "invalid_message": "must be Major.Minor.Patch all digits",
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
            "invalid_message": "must be Major.Minor all digits",
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
            "choices": get_args(templates.TemplateProps.__annotations__.get("license_type")),
            "message": "License:",
            "default": "MIT",
        },
        {
            "type": "input",
            "name": "source_url",
            "message": "Source Url:",
            "default": "https://github.com/",
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


def mkpyp(dry: bool = False, infile: str = None, outfile: str = None) -> None:  # type: ignore[assignment]
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
    name = str(props["name"])
    base = pwd / name
    req_base = base / "requirements"

    def mkdir(path: Union[str, Path]) -> None:
        if not testing:
            os.mkdir(path)
        else:
            print("=" * 80, file=sys.stdout)
            print(f"Creating directory: {path}", file=sys.stdout)

    def mkfile(path: Path) -> None:
        if not testing:
            path.open("x").close()
        else:
            print(f"Creating file: {path}")

    filewriter = templates.generate_file
    if not testing:
        filewriter = templates.generate_output

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
            Action(filewriter, req_base / "all.txt", templates.requirements_all_txt, props),
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


def run() -> None:
    fire.Fire(mkpyp, name="mkpyp")


if __name__ == "__main__":
    run()
