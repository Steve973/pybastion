#!/usr/bin/env python3
"""
Callable ID Generation Module

Centralized ID generation for the unit ledger system.
All IDs use underscore-separated segments: {unit_id}_{segment1}_{segment2}_...
"""

from __future__ import annotations

import hashlib
import re

EI_BINDING_CHAR: str = "B"
EI_CLASS_CHAR: str = "C"
EI_EX_ITEM_CHAR: str = "E"
EI_FUNC_CHAR: str = "F"
EI_INTEGRATION_CHAR: str = "I"
EI_METHOD_CHAR: str = "M"
EI_UNIT_CHAR: str = "U"
EI_ID_DELIM: str = "_"
EI_NESTED_DELIM: str = "."
FUNC_ID_EXPR: str = rf"^{EI_UNIT_CHAR}[0-9A-F]{{10}}_(?:{EI_CLASS_CHAR}\d{{3}}_{EI_METHOD_CHAR}\d{{3}}|{EI_FUNC_CHAR}\d{{3}})"
ENTRY_EI_ID: str = "E0000"


def generate_unit_id(fully_qualified_name: str) -> str:
    """Generate a unique unit ID from FQN."""
    # Use the first 10 chars of SHA256 hash
    hash_obj = hashlib.sha256(fully_qualified_name.encode())
    return f"{EI_UNIT_CHAR}{hash_obj.hexdigest()[:10].upper()}"


def generate_binding_id(unit_id: str, binding_num: int) -> str:
    """Generate a binding ID."""
    return f"{unit_id}{EI_ID_DELIM}{EI_BINDING_CHAR}{binding_num:03d}"


def generate_class_id(unit_id: str, class_num: int) -> str:
    """
    Generate a class ID.

    Args:
        unit_id: Unit identifier (e.g., "U12AB34CD56")
        class_num: Class counter value (1-based)

    Returns:
        Class ID string (e.g., "U12AB34CD56_C001")
    """
    return f"{unit_id}{EI_ID_DELIM}{EI_CLASS_CHAR}{class_num:03d}"


def generate_nested_class_id(parent_id: str, nested_num: int) -> str:
    """
    Generate a nested class ID.

    Args:
        parent_id: Parent class or function ID
        nested_num: Nested class counter value (1-based)

    Returns:
        Nested class ID string (e.g., "U12AB34CD56_C001.C002")
    """
    return f"{parent_id}{EI_NESTED_DELIM}{EI_CLASS_CHAR}{nested_num:03d}"


def generate_function_id(unit_id: str, func_num: int) -> str:
    """
    Generate a unit-level function ID.

    Args:
        unit_id: Unit identifier (e.g., "U12AB34CD56")
        func_num: Function counter value (1-based)

    Returns:
        Function ID string (e.g., "U12AB34CD56_F001")
    """
    return f"{unit_id}{EI_ID_DELIM}{EI_FUNC_CHAR}{func_num:03d}"


def generate_nested_function_id(parent_id: str, nested_num: int) -> str:
    """
    Generate a nested function ID.

    Args:
        parent_id: Parent function or class ID
        nested_num: Nested function counter value (1-based)

    Returns:
        Nested function ID string (e.g., "U12AB34CD56_F001.F002")
    """
    return f"{parent_id}{EI_NESTED_DELIM}{EI_FUNC_CHAR}{nested_num:03d}"


def generate_method_id(class_id: str, method_num: int) -> str:
    """
    Generate a method ID.

    Args:
        class_id: Class identifier (e.g., "U12AB34CD56_C001")
        method_num: Method counter value (1-based)

    Returns:
        Method ID string (e.g., "U12AB34CD56_C001_M001")
    """
    return f"{class_id}{EI_ID_DELIM}{EI_METHOD_CHAR}{method_num:03d}"


def generate_function_entry_ei_id(function_id: str) -> str:
    """
    Generate a function entry Execution Item (EI) ID.

    Args:
        function_id: Callable entry identifier of the function

    Returns:
        Function entry EI ID string (e.g., "U12AB34CD56_F001_E0000")
    """
    pattern = re.compile(FUNC_ID_EXPR)
    if not bool(pattern.match(function_id)):
        raise ValueError(f"Invalid function ID: {function_id}")
    return f"{function_id}{EI_ID_DELIM}{EI_EX_ITEM_CHAR}{0:04d}"


def generate_ei_id(callable_id: str, ei_num: int) -> str:
    """
    Generate an Execution Item (EI) ID.

    Args:
        callable_id: Callable identifier (function, method, etc.)
        ei_num: EI counter value (1-based)

    Returns:
        EI ID string (e.g., "U12AB34CD56_F001_E0001")
    """
    return f"{callable_id}{EI_ID_DELIM}{EI_EX_ITEM_CHAR}{ei_num:04d}"


def ei_id_to_integration_id(ei_id: str) -> str:
    """
    Convert an EI ID to an integration fact ID.

    Integration IDs are simply "I" prepended to the EI ID.

    Args:
        ei_id: Execution Item ID (e.g., "U12AB34CD56_F001_E0001")

    Returns:
        Integration ID string (e.g., "IU12AB34CD56_F001_E0001")
    """
    return f"{EI_INTEGRATION_CHAR}{ei_id}"
