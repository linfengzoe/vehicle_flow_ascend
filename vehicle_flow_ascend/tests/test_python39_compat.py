import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_isinstance_type_groups_are_python39_compatible() -> None:
    offenders: list[str] = []
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "isinstance"
                and len(node.args) >= 2
            ):
                continue
            if isinstance(node.args[1], ast.BinOp) and isinstance(node.args[1].op, ast.BitOr):
                relative = path.relative_to(PROJECT_ROOT).as_posix()
                offenders.append(f"{relative}:{node.lineno}")

    assert offenders == []


def test_pep604_type_hints_are_deferred_for_python39() -> None:
    offenders: list[str] = []
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        if not any(isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr) for node in ast.walk(tree)):
            continue
        has_future_annotations = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            and any(alias.name == "annotations" for alias in node.names)
            for node in tree.body
        )
        if not has_future_annotations:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            offenders.append(relative)

    assert offenders == []
