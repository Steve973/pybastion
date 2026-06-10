#!/usr/bin/env python3
"""
Synthetic feature-flow fixture for PyBastion Stage 4.

This file is intentionally boring. It exists to exercise feature flow expansion:

main
  -> a | b
  -> converge
  -> c | d
  -> converge
  -> end

Expected full feature cases:

- main::a::c
- main::a::d
- main::b::c
- main::b::d
"""

from io import BufferedReader
from synthetic_feature_dependency import SyntheticDependency


def normalize_input(raw: str) -> str:
    return raw.strip().lower()


def choose_first_branch(value: str) -> str:
    if value.startswith("a"):
        return "a"

    return "b"


def choose_second_branch(value: str) -> str:
    if value.endswith("c"):
        return "c"

    return "d"


def handle_a(value: str) -> str:
    return f"a:{value}"


def handle_b(value: str) -> str:
    return f"b:{value}"


def handle_c(value: str) -> str:
    return f"{value}:c"


def handle_d(value: str) -> str:
    return f"{value}:d"


def finish(value: str) -> dict[str, str]:
    return {
        "status": "ok",
        "value": value,
    }


def fixture_finish(value: str) -> dict[str, str]:
    return {
        "status": "ok",
        "value": value,
    }


################################################################################
##  Flow tests: if / else branch convergence
################################################################################


def fixture_if_branch_converge(value: int) -> dict[str, str]:
    # :: FeatureStart | name=fixture_if_branch_converge
    result = "unset"

    # :: FeatureBranch | name=fixture_if_branch_converge | branch=positive | control_polarity=true
    # :: FeatureBranch | name=fixture_if_branch_converge | branch=non_positive | control_polarity=false
    if value > 0:
        result = "positive"
    else:
        result = "non-positive"

    # :: FeatureConverge | name=fixture_if_branch_converge | branches=positive,non_positive | into=main
    converged = f"if:{result}"

    # :: FeatureEnd | name=fixture_if_branch_converge
    return fixture_finish(converged)


def fixture_if_branch_conditional_end(value: int) -> dict[str, str]:
    # :: FeatureStart | name=fixture_if_branch_conditional_end
    result = "unset"

    # :: FeatureBranch | name=fixture_if_branch_conditional_end | branch=valid | control_polarity=true
    # :: FeatureBranch | name=fixture_if_branch_conditional_end | branch=invalid | control_polarity=false
    if value > 0:
        result = f"valid:{value}"
    else:
        # :: FeatureEndConditional | name=fixture_if_branch_conditional_end | branch=invalid | on_condition=non_positive_value
        return fixture_finish("invalid")

    # :: FeatureConverge | name=fixture_if_branch_conditional_end | branches=valid | into=main
    converged = f"accepted:{result}"

    # :: FeatureEnd | name=fixture_if_branch_conditional_end
    return fixture_finish(converged)


################################################################################
##  Flow tests: nested if branch lineage
################################################################################


def fixture_nested_if_branch_converge(value: int) -> dict[str, str]:
    # :: FeatureStart | name=fixture_nested_if_branch_converge
    result = "unset"

    # :: FeatureBranch | name=fixture_nested_if_branch_converge | branch=positive | control_polarity=true
    # :: FeatureBranch | name=fixture_nested_if_branch_converge | branch=non_positive | control_polarity=false
    if value > 0:
        # :: FeatureBranch | name=fixture_nested_if_branch_converge | branch=large | control_polarity=true
        # :: FeatureBranch | name=fixture_nested_if_branch_converge | branch=small | control_polarity=false
        if value > 10:
            result = "large-positive"
        else:
            result = "small-positive"

        # :: FeatureConverge | name=fixture_nested_if_branch_converge | branches=large,small | into=positive
        result = f"positive:{result}"
    else:
        result = "non-positive"

    # :: FeatureConverge | name=fixture_nested_if_branch_converge | branches=positive,non_positive | into=main
    converged = f"nested-if:{result}"

    # :: FeatureEnd | name=fixture_nested_if_branch_converge
    return fixture_finish(converged)


################################################################################
##  Flow tests: match
################################################################################


def fixture_match_branch_converge(value: str) -> dict[str, str]:
    # :: FeatureStart | name=fixture_match_branch_converge
    normalized = value.strip().lower()

    # Match cases are all modeled as matched outcomes when a wildcard case exists.
    # These markers intentionally sit before the match statement, not before case labels.
    # :: FeatureBranch | name=fixture_match_branch_converge | branch=alpha | control_polarity=true | comment="case alpha"
    # :: FeatureBranch | name=fixture_match_branch_converge | branch=betaish | control_polarity=true | comment="case beta or bravo"
    # :: FeatureBranch | name=fixture_match_branch_converge | branch=guarded_x | control_polarity=true | comment="guarded x case"
    # :: FeatureBranch | name=fixture_match_branch_converge | branch=default | control_polarity=true | comment="wildcard case"
    match normalized:
        case "alpha":
            result = "alpha"
        case "beta" | "bravo":
            result = "betaish"
        case other if other.startswith("x"):
            result = f"guarded:{other}"
        case _:
            result = "default"

    # :: FeatureConverge | name=fixture_match_branch_converge | branches=alpha,betaish,guarded_x,default | into=main
    converged = f"match:{result}"

    # :: FeatureEnd | name=fixture_match_branch_converge
    return fixture_finish(converged)


################################################################################
##  Flow tests: for loop entry and zero-iteration branch
################################################################################


def fixture_for_loop_entry_converge(values: list[str]) -> dict[str, str]:
    # :: FeatureStart | name=fixture_for_loop_entry_converge
    total = 0

    # :: FeatureBranch | name=fixture_for_loop_entry_converge | branch=loop_entered | control_polarity=true
    # :: FeatureBranch | name=fixture_for_loop_entry_converge | branch=loop_skipped | control_polarity=false
    for value in values:
        total += len(value)

    # :: FeatureConverge | name=fixture_for_loop_entry_converge | branches=loop_entered,loop_skipped | into=main
    converged = f"for:{total}"

    # :: FeatureEnd | name=fixture_for_loop_entry_converge
    return fixture_finish(converged)


################################################################################
##  Flow tests: for loop break / continue disruptions
################################################################################


def fixture_for_loop_disruption_branches(values: list[str]) -> dict[str, str]:
    # :: FeatureStart | name=fixture_for_loop_disruption_branches
    total = 0

    # :: FeatureBranch | name=fixture_for_loop_disruption_branches | branch=loop_entered | control_polarity=true
    # :: FeatureBranch | name=fixture_for_loop_disruption_branches | branch=loop_skipped | control_polarity=false
    for value in values:
        # :: FeatureBranch | name=fixture_for_loop_disruption_branches | branch=break_requested | control_polarity=true
        # :: FeatureBranch | name=fixture_for_loop_disruption_branches | branch=no_break | control_polarity=false
        if value == "stop":
            # :: FeatureEndConditional | name=fixture_for_loop_disruption_branches | branch=break_requested | on_condition=loop_break
            break

        # :: FeatureConverge | name=fixture_for_loop_disruption_branches | branches=no_break | into=loop_entered
        checked_value = value

        # :: FeatureBranch | name=fixture_for_loop_disruption_branches | branch=continue_requested | control_polarity=true
        # :: FeatureBranch | name=fixture_for_loop_disruption_branches | branch=consume_value | control_polarity=false
        if checked_value == "skip":
            # :: FeatureEndConditional | name=fixture_for_loop_disruption_branches | branch=continue_requested | on_condition=loop_continue
            continue

        # :: FeatureConverge | name=fixture_for_loop_disruption_branches | branches=consume_value | into=loop_entered
        total += len(checked_value)

    # :: FeatureConverge | name=fixture_for_loop_disruption_branches | branches=loop_entered,loop_skipped | into=main
    converged = f"for-disruptions:{total}"

    # :: FeatureEnd | name=fixture_for_loop_disruption_branches
    return fixture_finish(converged)


################################################################################
##  Flow tests: while loop entry and zero-iteration branch
################################################################################


def fixture_while_loop_entry_converge(limit: int) -> dict[str, str]:
    # :: FeatureStart | name=fixture_while_loop_entry_converge
    count = 0

    # :: FeatureBranch | name=fixture_while_loop_entry_converge | branch=loop_entered | control_polarity=true
    # :: FeatureBranch | name=fixture_while_loop_entry_converge | branch=loop_skipped | control_polarity=false
    while count < limit:
        count += 1

    # :: FeatureConverge | name=fixture_while_loop_entry_converge | branches=loop_entered,loop_skipped | into=main
    converged = f"while:{count}"

    # :: FeatureEnd | name=fixture_while_loop_entry_converge
    return fixture_finish(converged)


################################################################################
##  Flow tests: while loop break / continue disruptions
################################################################################


def fixture_while_loop_disruption_branches(limit: int) -> dict[str, str]:
    # :: FeatureStart | name=fixture_while_loop_disruption_branches
    count = 0
    total = 0

    # :: FeatureBranch | name=fixture_while_loop_disruption_branches | branch=loop_entered | control_polarity=true
    # :: FeatureBranch | name=fixture_while_loop_disruption_branches | branch=loop_skipped | control_polarity=false
    while count < limit:
        count += 1

        # :: FeatureBranch | name=fixture_while_loop_disruption_branches | branch=continue_requested | control_polarity=true
        # :: FeatureBranch | name=fixture_while_loop_disruption_branches | branch=no_continue | control_polarity=false
        if count == 2:
            # :: FeatureEndConditional | name=fixture_while_loop_disruption_branches | branch=continue_requested | on_condition=loop_continue
            continue

        # :: FeatureConverge | name=fixture_while_loop_disruption_branches | branches=no_continue | into=loop_entered
        checked_count = count

        # :: FeatureBranch | name=fixture_while_loop_disruption_branches | branch=break_requested | control_polarity=true
        # :: FeatureBranch | name=fixture_while_loop_disruption_branches | branch=consume_count | control_polarity=false
        if checked_count == 4:
            # :: FeatureEndConditional | name=fixture_while_loop_disruption_branches | branch=break_requested | on_condition=loop_break
            break

        # :: FeatureConverge | name=fixture_while_loop_disruption_branches | branches=consume_count | into=loop_entered
        total += checked_count

    # :: FeatureConverge | name=fixture_while_loop_disruption_branches | branches=loop_entered,loop_skipped | into=main
    converged = f"while-disruptions:{total}"

    # :: FeatureEnd | name=fixture_while_loop_disruption_branches
    return fixture_finish(converged)


################################################################################
##  Flow tests: nested loop lineage
################################################################################


def fixture_nested_loop_converge(values: list[list[int]]) -> dict[str, str]:
    # :: FeatureStart | name=fixture_nested_loop_converge
    total = 0

    # :: FeatureBranch | name=fixture_nested_loop_converge | branch=outer_entered | control_polarity=true
    # :: FeatureBranch | name=fixture_nested_loop_converge | branch=outer_skipped | control_polarity=false
    for row in values:
        # :: FeatureBranch | name=fixture_nested_loop_converge | branch=inner_entered | control_polarity=true
        # :: FeatureBranch | name=fixture_nested_loop_converge | branch=inner_skipped | control_polarity=false
        for item in row:
            total += item

        # :: FeatureConverge | name=fixture_nested_loop_converge | branches=inner_entered,inner_skipped | into=outer_entered
        total += 100

    # :: FeatureConverge | name=fixture_nested_loop_converge | branches=outer_entered,outer_skipped | into=main
    converged = f"nested-loop:{total}"

    # :: FeatureEnd | name=fixture_nested_loop_converge
    return fixture_finish(converged)


################################################################################
##  Flow tests: try / except / else / finally
################################################################################


def fixture_try_except_else_finally_trace(value: str) -> dict[str, str]:
    # :: FeatureStart | name=fixture_try_except_else_finally_trace
    result = "unset"

    # :: FeatureBranch | name=fixture_try_except_else_finally_trace | branch=try_success
    # :: FeatureBranch | name=fixture_try_except_else_finally_trace | branch=value_error
    # :: FeatureBranch | name=fixture_try_except_else_finally_trace | branch=type_error
    try:
        parsed = int(value)

    except ValueError:
        result = "value-error"

    except TypeError:
        result = "type-error"

    else:
        result = f"parsed:{parsed}"

    finally:
        result = f"finally:{result}"

    # :: FeatureConverge | name=fixture_try_except_else_finally_trace | branches=try_success,value_error,type_error | into=main
    converged = result

    # :: FeatureEnd | name=fixture_try_except_else_finally_trace
    return fixture_finish(converged)


def fixture_try_with_inner_if_branch(value: str) -> dict[str, str]:
    # :: FeatureStart | name=fixture_try_with_inner_if_branch
    result = "unset"

    # :: FeatureBranch | name=fixture_try_with_inner_if_branch | branch=try_success
    # :: FeatureBranch | name=fixture_try_with_inner_if_branch | branch=value_error
    try:
        normalized = value.strip()

        # :: FeatureBranch | name=fixture_try_with_inner_if_branch | branch=direct_value | control_polarity=true
        # :: FeatureBranch | name=fixture_try_with_inner_if_branch | branch=parse_value | control_polarity=false
        if normalized == "value":
            result = "direct-value"
        else:
            result = f"parsed:{int(normalized)}"

        # :: FeatureConverge | name=fixture_try_with_inner_if_branch | branches=direct_value,parse_value | into=try_success
        result = f"try:{result}"

    except ValueError:
        result = "value-error"

    finally:
        result = f"finally:{result}"

    # :: FeatureConverge | name=fixture_try_with_inner_if_branch | branches=try_success,value_error | into=main
    converged = result

    # :: FeatureEnd | name=fixture_try_with_inner_if_branch
    return fixture_finish(converged)


################################################################################
##  Flow tests: with statement and post-with branch
################################################################################


def fixture_with_trace_and_content_branch(path: str) -> dict[str, str]:
    # :: FeatureStart | name=fixture_with_trace_and_content_branch
    handle: BufferedReader

    # :: FeatureTrace | name=fixture_with_trace_and_content_branch | comment="with statement entered"
    with open(path, encoding="utf-8") as handle:
        # :: FeatureTrace | name=fixture_with_trace_and_content_branch | comment="with body entered"
        content = handle.read()

    # :: FeatureBranch | name=fixture_with_trace_and_content_branch | branch=has_content | control_polarity=true
    # :: FeatureBranch | name=fixture_with_trace_and_content_branch | branch=empty_content | control_polarity=false
    if content:
        result = "has-content"
    else:
        result = "empty"

    # :: FeatureConverge | name=fixture_with_trace_and_content_branch | branches=has_content,empty_content | into=main
    converged = f"with:{result}"

    # :: FeatureEnd | name=fixture_with_trace_and_content_branch
    return fixture_finish(converged)


################################################################################
##  Flow tests: mixed control without pretending try/with regions are branches
################################################################################


def fixture_mixed_control_feature(values: list[str], path: str) -> dict[str, str]:
    # :: FeatureStart | name=fixture_mixed_control_feature
    result = "unset"
    handle: BufferedReader

    # :: FeatureBranch | name=fixture_mixed_control_feature | branch=try_success
    # :: FeatureBranch | name=fixture_mixed_control_feature | branch=os_error
    try:
        total = 0

        # :: FeatureBranch | name=fixture_mixed_control_feature | branch=loop_entered | control_polarity=true
        # :: FeatureBranch | name=fixture_mixed_control_feature | branch=loop_skipped | control_polarity=false
        for value in values:
            # :: FeatureBranch | name=fixture_mixed_control_feature | branch=skip_value | control_polarity=true
            # :: FeatureBranch | name=fixture_mixed_control_feature | branch=use_value | control_polarity=false
            if value == "skip":
                continue

            total += len(value)

        # :: FeatureConverge | name=fixture_mixed_control_feature | branches=loop_entered,loop_skipped | into=try_success
        loop_result = total

        with open(path, encoding="utf-8") as handle:
            content = handle.read()

        # :: FeatureBranch | name=fixture_mixed_control_feature | branch=has_content | control_polarity=true
        # :: FeatureBranch | name=fixture_mixed_control_feature | branch=empty_content | control_polarity=false
        if content:
            result = f"content:{loop_result}"
        else:
            result = f"empty:{loop_result}"

        # :: FeatureConverge | name=fixture_mixed_control_feature | branches=has_content,empty_content | into=try_success
        result = f"mixed:{result}"

    except OSError:
        result = "os-error"

    finally:
        result = f"finally:{result}"

    # :: FeatureConverge | name=fixture_mixed_control_feature | branches=try_success,os_error | into=main
    converged = result

    # :: FeatureEnd | name=fixture_mixed_control_feature
    return fixture_finish(converged)


################################################################################
##  Fixtures to test control statement analysis
################################################################################


def fixture_if_control(value: str) -> str:
    if value.startswith("a"):
        return "starts-a"

    if value.endswith("z"):
        return "ends-z"

    return "other"


def fixture_match_control(value: str) -> str:
    match value:
        case "alpha":
            return "matched-alpha"
        case "beta" | "bravo":
            return "matched-betaish"
        case other if other.startswith("x"):
            return "matched-guarded-x"
        case _:
            return "matched-default"


def fixture_for_control(values: list[str]) -> str:
    result = "empty"

    for value in values:
        if value == "stop":
            break

        if value == "skip":
            continue

        result = f"seen:{value}"
    else:
        result = "completed"

    return result


def fixture_while_control(limit: int) -> str:
    count = 0

    while count < limit:
        count += 1

        if count == 2:
            continue

        if count == 4:
            break
    else:
        return "exhausted"

    return f"stopped:{count}"


def fixture_try_control(value: str) -> str:
    try:
        normalized = value.strip()

        if normalized == "value":
            return "direct-value"

        parsed = int(normalized)
    except ValueError:
        return "value-error"
    except TypeError:
        return "type-error"
    else:
        return f"parsed:{parsed}"
    finally:
        cleanup_marker = "cleanup"
        len(cleanup_marker)


def fixture_with_control(path: str) -> str:
    handle: BufferedReader
    with open(path, encoding="utf-8") as handle:
        content = handle.read()

    if content:
        return "has-content"

    return "empty"


def fixture_mixed_control(values: list[str], fallback: str) -> str:
    result = fallback
    text: str
    value: str

    try:
        for value in values:
            match value:
                case "skip":
                    continue
                case "stop":
                    break
                case text if text.startswith("a"):
                    result = text.upper()
                case _:
                    result = value.lower()
        else:
            result = "loop-completed"
    except AttributeError:
        result = "attribute-error"
    finally:
        result = f"final:{result}"

    if result.endswith("error"):
        return "failed"

    return result


def fixture_full_control_flow_probe(
    value: int,
    items: list[int],
    mapping: dict[str, int],
    path,
) -> int:
    total = 0
    if value > 10:
        total += value
    elif value == 10:
        total += 100
    else:
        total -= value
    match value:
        case 0:
            total += 1
        case 1 | 2:
            total += 2
        case other if other < 0:
            total -= 10
        case _:
            total += 5
    for item in items:
        if item < 0:
            continue
        if item == 0:
            break
        total += item
    else:
        total += 50
    while total < 100:
        total += 10
        if total == 70:
            break
    else:
        total += 7
    try:
        handle: BufferedReader
        raw: bytes
        with path.open("r") as handle:
            raw = handle.read()

        if raw.strip():
            total += len(raw)
        else:
            total -= 1
    except FileNotFoundError:
        total = -404
    except OSError:
        total = -1
    else:
        total += mapping.get("bonus", 0)
    finally:
        total += 3
    return total


def fixture_if_normal_completion_probe(value: int) -> int:
    total = 0
    if value > 0:
        total += 1
    else:
        total -= 1
    return total


def fixture_if_partial_disruption_probe(value: int) -> int:
    total = 0
    if value > 0:
        if value == 10:
            return 10
        total += 1
    return total


def fixture_loop_direct_disruptions(items: list[int]) -> int:
    total = 0
    for item in items:
        total += item
        continue
    while total < 10:
        total += 1
        break
    return total


def fixture_if_direct_return_raise(value: int) -> int:
    if value > 0:
        return value
    else:
        raise ValueError("value must be positive")


def fixture_match_direct_return_raise(value: int) -> int:
    match value:
        case 0:
            return 0
        case _:
            raise ValueError("unsupported value")


def fixture_loop_direct_return_raise(items: list[int]) -> int:
    for item in items:
        if item < 0:
            break
        return item
    while True:
        raise RuntimeError("no usable item")
    return -1


def fixture_nested_if_return_raise(value: int) -> int:
    total = 0
    if value > 0:
        if value == 10:
            return 10
        total += 1
    if value < -10:
        raise ValueError("too negative")
    return total


def fixture_try_loop_disruptions(values: list[str]) -> int:
    total = 0
    for value in values:
        try:
            if value == "skip":
                continue
            if value == "stop":
                break
            total += int(value)
        except ValueError:
            continue
    return total


def fixture_with_loop_disruptions(values: list[str], path) -> int:
    total = 0
    handle: BufferedReader
    for value in values:
        with path.open("r") as handle:
            handle.read()
            if value == "skip":
                continue
            if value == "stop":
                break
            total += 1
    return total


def fixture_nested_loop_transfer_binding(values: list[list[int]]) -> int:
    total = 0
    for row in values:
        for item in row:
            if item < 0:
                continue
            if item == 0:
                break
            total += item
        total += 100
    return total


def fixture_try_direct_return_raise(value: int) -> int:
    try:
        if value == 0:
            raise ValueError("zero")
        return value
    except ValueError:
        raise RuntimeError("wrapped")
    finally:
        value + 1


def fixture_with_direct_return_raise(path, mode: str) -> str:
    handle: BufferedReader
    if mode == "raise":
        with path.open("r") as handle:
            raise RuntimeError("forced failure")
    with path.open("r") as handle:
        return handle.read()


def fixture_try_direct_return_suppresses_else(value: int) -> int:
    try:
        return value
    except ValueError:
        return -1
    else:
        return 100


def fixture_try_handler_else_normal_completion(value: str) -> int:
    result = 0
    try:
        parsed = int(value)
    except ValueError:
        result = -1
    else:
        result = parsed
    return result


def fixture_try_finally_normal_resume_target(value: int) -> int:
    result = value

    try:
        result += 1
    finally:
        result += 10

    result += 100
    return result


def calls_other_unit(dependency: SyntheticDependency, value: str) -> str:
    return dependency.transform(value)
