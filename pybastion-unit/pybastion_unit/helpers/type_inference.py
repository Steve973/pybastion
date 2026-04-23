from __future__ import annotations

import ast

from pybastion_common.models import TypeRef


def infer_literal_type(expr: ast.AST | None) -> TypeRef | None:
    if expr is None:
        return None

    if isinstance(expr, ast.Constant):
        value = expr.value
        if isinstance(value, str):
            return TypeRef(name="str")
        if isinstance(value, bool):
            return TypeRef(name="bool")
        if isinstance(value, int):
            return TypeRef(name="int")
        if isinstance(value, float):
            return TypeRef(name="float")
        if value is None:
            return TypeRef(name="None")
        return None

    if isinstance(expr, ast.List):
        return TypeRef(name="list")
    if isinstance(expr, ast.Tuple):
        return TypeRef(name="tuple")
    if isinstance(expr, ast.Set):
        return TypeRef(name="set")
    if isinstance(expr, ast.Dict):
        return TypeRef(name="dict")

    return None


def build_param_type_map_with_defaults(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, TypeRef]:
    type_map: dict[str, TypeRef] = {}

    positional_args = list(node.args.args)
    positional_defaults = list(node.args.defaults)
    positional_default_start = len(positional_args) - len(positional_defaults)

    for idx, arg in enumerate(positional_args):
        type_ref = TypeRef.from_annotation_ast(arg.annotation)

        if type_ref is None and idx >= positional_default_start:
            default_expr = positional_defaults[idx - positional_default_start]
            type_ref = infer_literal_type(default_expr)

        if type_ref is not None:
            type_map[arg.arg] = type_ref

    for arg, default_expr in zip(node.args.kwonlyargs, node.args.kw_defaults):
        type_ref = TypeRef.from_annotation_ast(arg.annotation)

        if type_ref is None:
            type_ref = infer_literal_type(default_expr)

        if type_ref is not None:
            type_map[arg.arg] = type_ref

    return type_map
