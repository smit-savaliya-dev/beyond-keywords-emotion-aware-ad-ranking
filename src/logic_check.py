"""Proof that a presentation-only edit changed no logic: compare Python files as normalised syntax trees.

Normalising drops docstrings (comments never reach the tree) and maps renamed modules and script names
to their new names, so only real code differences remain. Input: source text. Output: per-statement dumps.
"""
import ast
import hashlib
import json
import re


class _Rename(ast.NodeTransformer):
    """Rewrite old module names in imports, and old module/script names inside string constants."""

    def __init__(self, modules, stems):
        self.modules, self.stems = modules, stems

    def _module(self, name):
        head, _, rest = name.partition(".")
        if head == "src" and rest:
            first, _, tail = rest.partition(".")
            return ".".join(filter(None, ["src", self.modules.get(first, first), tail]))
        return self.modules.get(name, name)

    def visit_ImportFrom(self, node):
        if node.module:
            node.module = self._module(node.module)
        if (node.level and not node.module) or node.module == "src":
            for a in node.names:
                a.name = self.modules.get(a.name, a.name)
        return node

    def visit_Import(self, node):
        for a in node.names:
            a.name = self._module(a.name)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            text = node.value
            for old, new in self.stems.items():
                text = text.replace(old, new)
            for old, new in self.modules.items():
                text = re.sub(rf"\bsrc\.{re.escape(old)}\b", f"src.{new}", text)
            node.value = text
        return node


def _drop_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return tree


def _label(i, node):
    name = getattr(node, "name", None)
    if name is None and isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        name = node.targets[0].id
    return f"{i}:{type(node).__name__}:{name or ''}"


def normalised_statements(source, modules=None, stems=None):
    """[(label, dump)] for each top-level statement, docstrings removed and old names mapped to new."""
    tree = _drop_docstrings(ast.parse(source))
    tree = ast.fix_missing_locations(_Rename(modules or {}, stems or {}).visit(tree))
    return [(_label(i, node), ast.dump(node, include_attributes=False)) for i, node in enumerate(tree.body)]


def digest(statements):
    return hashlib.sha256(json.dumps(statements).encode("utf-8")).hexdigest()


def differences(old_source, new_source, modules=None, stems=None, allowed=(), added=()):
    """Labels of top-level statements that differ once old names are mapped; names in `allowed` may differ and names in `added` may be new."""
    old = normalised_statements(old_source, modules, stems)
    new = [(l, d) for l, d in normalised_statements(new_source) if l.split(":")[2] not in added]
    old_names, new_names = [l.split(":", 1)[1] for l, _ in old], [l.split(":", 1)[1] for l, _ in new]
    if old_names != new_names:
        return [f"statements differ: {sorted(set(old_names) ^ set(new_names)) or 'order'}"]
    return [l for (l, x), (_, y) in zip(old, new) if x != y and l.split(":")[2] not in allowed]
