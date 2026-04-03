from __future__ import annotations

import ast

from .common import operation_eis
from .decomp_types import DecompositionContext, DecomposerResult


def _wrap(text: str) -> str:
    return f"({text})"


def _join_and(parts: list[str]) -> str:
    if not parts:
        return "True"
    if len(parts) == 1:
        return parts[0]
    return " and ".join(_wrap(part) for part in parts)


def _negate_text(text: str) -> str:
    return f"not ({text})"


def enumerate_truth_conditions(node: ast.AST) -> tuple[list[str], list[str]]:
    """
    Return:
      (true_conditions, false_conditions)

    Each item is a full path condition under which the node evaluates
    to that boolean result, preserving short-circuit behavior.

    Examples:
      a
        -> (["a"], ["not (a)"])

      a and b
        -> (["(a) and (b)"],
            ["not (a)", "(a) and (not (b))"])

      a or b
        -> (["a", "(not (a)) and (b)"],
            ["(not (a)) and (not (b))"])
    """
    match node:
        case ast.BoolOp(op=ast.And(), values=values):
            return _enumerate_and(values)

        case ast.BoolOp(op=ast.Or(), values=values):
            return _enumerate_or(values)

        case ast.UnaryOp(op=ast.Not(), operand=operand):
            operand_true, operand_false = enumerate_truth_conditions(operand)
            return operand_false, operand_true

        case _:
            expr = ast.unparse(node)
            return [expr], [_negate_text(expr)]


def _enumerate_and(values: list[ast.AST]) -> tuple[list[str], list[str]]:
    true_conditions: list[str] = []
    false_conditions: list[str] = []

    # For AND:
    # - true requires every operand true
    # - false happens at the first operand that can go false,
    #   with all previous operands true
    true_prefixes: list[str] = []
    prior_true_prefixes: list[str] = []

    for value in values:
        value_true, value_false = enumerate_truth_conditions(value)

        # Any false path here requires all prior operands true.
        if prior_true_prefixes:
            for prior in prior_true_prefixes:
                for false_cond in value_false:
                    false_conditions.append(_join_and([prior, false_cond]))
        else:
            false_conditions.extend(value_false)

        # Update the running "all previous are true" prefixes.
        if not prior_true_prefixes:
            prior_true_prefixes = list(value_true)
        else:
            new_prefixes: list[str] = []
            for prior in prior_true_prefixes:
                for true_cond in value_true:
                    new_prefixes.append(_join_and([prior, true_cond]))
            prior_true_prefixes = new_prefixes

    true_conditions = prior_true_prefixes
    return true_conditions, false_conditions


def _enumerate_or(values: list[ast.AST]) -> tuple[list[str], list[str]]:
    true_conditions: list[str] = []

    # For OR:
    # - true happens at the first operand that can go true,
    #   with all previous operands false
    # - false requires every operand false
    prior_false_prefixes: list[str] = []

    for value in values:
        value_true, value_false = enumerate_truth_conditions(value)

        # Any true path here requires all prior operands false.
        if prior_false_prefixes:
            for prior in prior_false_prefixes:
                for true_cond in value_true:
                    true_conditions.append(_join_and([prior, true_cond]))
        else:
            true_conditions.extend(value_true)

        # Update the running "all previous are false" prefixes.
        if not prior_false_prefixes:
            prior_false_prefixes = list(value_false)
        else:
            new_prefixes: list[str] = []
            for prior in prior_false_prefixes:
                for false_cond in value_false:
                    new_prefixes.append(_join_and([prior, false_cond]))
            prior_false_prefixes = new_prefixes

    false_conditions: list[str] = prior_false_prefixes
    return true_conditions, false_conditions

class BoolOpDecomposer:
    @classmethod
    def decompose_expression_boolop(
        cls,
        boolop: ast.BoolOp,
        context: DecompositionContext,
    ) -> list[DecomposerResult]:
        values = list(boolop.values)
        if not values:
            return []

        results: list[DecomposerResult] = []
        for value in values:
            results.extend(decompose_expression_node(value, context))
        return results


def decompose_expression_node(node: ast.AST, context: DecompositionContext) -> list[DecomposerResult]:
    if isinstance(node, ast.BoolOp):
        return BoolOpDecomposer.decompose_expression_boolop(node, context)
    if isinstance(node, ast.IfExp):
        results: list[DecomposerResult] = []
        results.extend(decompose_expression_node(node.test, context))
        results.extend(decompose_expression_node(node.body, context))
        results.extend(decompose_expression_node(node.orelse, context))
        return results
    if isinstance(node, ast.NamedExpr):
        return decompose_expression_node(node.value, context)
    if isinstance(node, ast.Await):
        return decompose_expression_node(node.value, context)
    if isinstance(node, ast.BinOp):
        return decompose_expression_node(node.left, context) + decompose_expression_node(node.right, context)
    if isinstance(node, ast.UnaryOp):
        return decompose_expression_node(node.operand, context)
    if isinstance(node, ast.Call):
        return operation_eis([node])
    if isinstance(node, ast.Compare):
        return operation_eis([node])
    if isinstance(node, ast.keyword):
        return decompose_expression_node(node.value, context)
    if isinstance(node, ast.Subscript):
        return decompose_expression_node(node.value, context) + decompose_expression_node(node.slice, context)
    if isinstance(node, ast.Slice):
        results: list[DecomposerResult] = []
        if node.lower:
            results.extend(decompose_expression_node(node.lower, context))
        if node.upper:
            results.extend(decompose_expression_node(node.upper, context))
        if node.step:
            results.extend(decompose_expression_node(node.step, context))
        return results
    if isinstance(node, ast.Attribute):
        return []
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        results: list[DecomposerResult] = []
        for elt in node.elts:
            results.extend(decompose_expression_node(elt, context))
        return results
    if isinstance(node, ast.Dict):
        results: list[DecomposerResult] = []
        for key in node.keys:
            if key is not None:
                results.extend(decompose_expression_node(key, context))
        for value in node.values:
            results.extend(decompose_expression_node(value, context))
        return results
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return operation_eis([node])
    if isinstance(node, (ast.Name, ast.Constant)):
        return []
    return operation_eis([node])
