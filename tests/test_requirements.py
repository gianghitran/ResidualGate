
import pathlib
import re

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NAME_RE = re.compile(r"^\s*([A-Za-z0-9._-]+)")


def _names(path: pathlib.Path) -> set[str]:
    names = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = NAME_RE.match(line)
        if match:
            names.add(match.group(1).lower().replace("_", "-"))
    return names


@pytest.fixture(scope="module")
def pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _declared(pyproject, *groups) -> set[str]:
    specs = []
    for group in groups:
        if group == "dependencies":
            specs += pyproject["project"]["dependencies"]
        else:
            specs += pyproject["project"]["optional-dependencies"][group]
    return {NAME_RE.match(s).group(1).lower().replace("_", "-") for s in specs}


def test_base_requirements_match_pyproject(pyproject):
    declared = _declared(pyproject, "dependencies", "dev")
    listed = _names(ROOT / "requirements.txt")
    assert listed == declared, (
        f"requirements.txt only: {listed - declared}; pyproject only: {declared - listed}"
    )


def test_gpu_extras_are_not_in_the_base_install(pyproject):
    heavy = {"torch", "vllm", "transformers", "accelerate", "nvidia-ml-py"}
    assert not (_declared(pyproject, "dependencies") & heavy)
    assert not (_names(ROOT / "requirements.txt") & heavy)


def test_the_launcher_forwards_unknown_flags_verbatim():
    import importlib.util

    spec = importlib.util.spec_from_file_location("agentgate_main", ROOT / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parser_args, rest = None, None
    original = module.COMMANDS["synth-attack"]
    try:
        def capture(args, forwarded):
            nonlocal parser_args, rest
            parser_args, rest = args, forwarded
            return 0

        module.COMMANDS["synth-attack"] = capture
        module.NEEDS_CAMEL.discard("synth-attack")
        module.main(["synth-attack", "--suite", "banking", "--mode", "chain", "--limit", "20"])
    finally:
        module.COMMANDS["synth-attack"] = original
        module.NEEDS_CAMEL.add("synth-attack")

    assert rest == ["--suite", "banking", "--mode", "chain", "--limit", "20"]
    assert parser_args.model is None, "--mode must not be swallowed as --model"
