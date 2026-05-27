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


################################################################################
##  Flow tests: if/elif/else
################################################################################


def synthetic_branch_multiply_feature(raw: str) -> dict[str, str]:
    # :: FeatureStart | name=synthetic_branch_multiply
    value = normalize_input(raw)

    first_branch = choose_first_branch(value)

    # :: FeatureBranch | name=synthetic_branch_multiply | branch=a | control_polarity=true
    # :: FeatureBranch | name=synthetic_branch_multiply | branch=b | control_polarity=false
    if first_branch == "a":
        first_result = handle_a(value)
    else:
        first_result = handle_b(value)

    # :: FeatureConverge | name=synthetic_branch_multiply | branches=a,b | into=main
    after_first_converge = f"first:{first_result}"

    second_branch = choose_second_branch(after_first_converge)

    # :: FeatureBranch | name=synthetic_branch_multiply | branch=c | control_polarity=true
    # :: FeatureBranch | name=synthetic_branch_multiply | branch=d | control_polarity=false
    if second_branch == "c":
        second_result = handle_c(after_first_converge)
    else:
        second_result = handle_d(after_first_converge)

    # :: FeatureConverge | name=synthetic_branch_multiply | branches=c,d | into=main
    after_second_converge = f"second:{second_result}"

    # :: FeatureEnd | name=synthetic_branch_multiply
    return finish(after_second_converge)


def synthetic_nested_branch_feature(raw: str) -> dict[str, str]:
    # :: FeatureStart | name=synthetic_nested_branch
    value = normalize_input(raw)

    first_branch = choose_first_branch(value)

    # :: FeatureBranch | name=synthetic_nested_branch | branch=a | control_polarity=true
    # :: FeatureBranch | name=synthetic_nested_branch | branch=b | control_polarity=false
    if first_branch == "a":
        nested_branch = choose_second_branch(value)

        # :: FeatureBranch | name=synthetic_nested_branch | branch=a_c | control_polarity=true
        # :: FeatureBranch | name=synthetic_nested_branch | branch=a_d | control_polarity=false
        if nested_branch == "c":
            branch_result = handle_c(handle_a(value))
        else:
            branch_result = handle_d(handle_a(value))
    else:
        branch_result = handle_b(value)

    # :: FeatureConverge | name=synthetic_nested_branch | branches=a_c,a_d,b | into=main
    converged = f"nested:{branch_result}"

    # :: FeatureEnd | name=synthetic_nested_branch
    return finish(converged)


################################################################################
##  Flow tests: match
################################################################################


def synthetic_match_feature(value: str) -> dict[str, str]:
    # :: FeatureStart | name=synthetic_match
    normalized = normalize_input(value)

    match normalized:
        # :: FeatureBranch | name=synthetic_match | branch=alpha
        case "alpha":
            result = handle_a(normalized)
        # :: FeatureBranch | name=synthetic_match | branch=betaish
        case "beta" | "bravo":
            result = handle_b(normalized)
        # :: FeatureBranch | name=synthetic_match | branch=guarded_x
        case other if other.startswith("x"):
            result = handle_c(other)
        # :: FeatureBranch | name=synthetic_match | branch=default
        case _:
            result = handle_d(normalized)

    # :: FeatureConverge | name=synthetic_match | branches=alpha,betaish,guarded_x,default | into=main
    converged = f"match:{result}"

    # :: FeatureEnd | name=synthetic_match
    return finish(converged)


################################################################################
##  Flow tests: loop completion and disruption
################################################################################


def synthetic_for_loop_feature(values: list[str]) -> dict[str, str]:
    # :: FeatureStart | name=synthetic_for_loop
    result = "empty"

    # :: FeatureBranch | name=synthetic_for_loop | branch=loop_body | control_polarity=true
    # :: FeatureBranch | name=synthetic_for_loop | branch=loop_else | control_polarity=false
    for value in values:
        # :: FeatureBranch | name=synthetic_for_loop | branch=loop_break | control_polarity=true
        # :: FeatureBranch | name=synthetic_for_loop | branch=not_break | control_polarity=false
        if value == "stop":
            result = "stopped"
            break

        # :: FeatureBranch | name=synthetic_for_loop | branch=loop_continue | control_polarity=true
        # :: FeatureBranch | name=synthetic_for_loop | branch=loop_consume | control_polarity=false
        if value == "skip":
            result = "skipped"
            continue

        result = f"seen:{value}"
    else:
        result = "completed"

    # :: FeatureConverge | name=synthetic_for_loop | branches=loop_break,loop_continue,loop_consume,loop_else | into=main
    converged = f"for:{result}"

    # :: FeatureEnd | name=synthetic_for_loop
    return finish(converged)


def synthetic_while_loop_feature(limit: int) -> dict[str, str]:
    # :: FeatureStart | name=synthetic_while_loop
    count = 0
    result = "initial"

    # :: FeatureBranch | name=synthetic_while_loop | branch=loop_body | control_polarity=true
    # :: FeatureBranch | name=synthetic_while_loop | branch=loop_else | control_polarity=false
    while count < limit:
        count += 1

        # :: FeatureBranch | name=synthetic_while_loop | branch=loop_continue | control_polarity=true
        # :: FeatureBranch | name=synthetic_while_loop | branch=not_continue | control_polarity=false
        if count == 2:
            result = "continued"
            continue

        # :: FeatureBranch | name=synthetic_while_loop | branch=loop_break | control_polarity=true
        # :: FeatureBranch | name=synthetic_while_loop | branch=loop_consume | control_polarity=false
        if count == 4:
            result = "stopped"
            break

        result = f"seen:{count}"
    else:
        result = "exhausted"

    # :: FeatureConverge | name=synthetic_while_loop | branches=loop_continue,loop_break,loop_consume,loop_else | into=main
    converged = f"while:{result}"

    # :: FeatureEnd | name=synthetic_while_loop
    return finish(converged)


def synthetic_nested_loop_feature(values: list[list[int]]) -> dict[str, str]:
    # :: FeatureStart | name=synthetic_nested_loop
    total = 0

    # :: FeatureBranch | name=synthetic_nested_loop | branch=outer_body | control_polarity=true
    # :: FeatureBranch | name=synthetic_nested_loop | branch=outer_empty | control_polarity=false
    for row in values:
        # :: FeatureBranch | name=synthetic_nested_loop | branch=inner_body | control_polarity=true
        # :: FeatureBranch | name=synthetic_nested_loop | branch=inner_empty | control_polarity=false
        for item in row:
            # :: FeatureBranch | name=synthetic_nested_loop | branch=inner_continue | control_polarity=true
            # :: FeatureBranch | name=synthetic_nested_loop | branch=not_inner_continue | control_polarity=false
            if item < 0:
                continue

            # :: FeatureBranch | name=synthetic_nested_loop | branch=inner_break | control_polarity=true
            # :: FeatureBranch | name=synthetic_nested_loop | branch=inner_consume | control_polarity=false
            if item == 0:
                break

            total += item

        # :: FeatureConverge | name=synthetic_nested_loop | branches=inner_continue,inner_break,inner_consume,inner_empty | into=outer_body
        total += 100

    # :: FeatureConverge | name=synthetic_nested_loop | branches=outer_body,outer_empty | into=main
    converged = f"nested-loop:{total}"

    # :: FeatureEnd | name=synthetic_nested_loop
    return finish(converged)


################################################################################
##  Flow tests: Try / except / else / finally
################################################################################


def synthetic_try_except_else_finally_feature(value: str) -> dict[str, str]:
    # :: FeatureStart | name=synthetic_try_except_else_finally
    result = "unset"

    try:
        # :: FeatureBranch | name=synthetic_try_except_else_finally | branch=try_body
        normalized = value.strip()

        # :: FeatureBranch | name=synthetic_try_except_else_finally | branch=direct_value | control_polarity=true
        # :: FeatureBranch | name=synthetic_try_except_else_finally | branch=parse_attempt | control_polarity=false
        if normalized == "value":
            result = "direct-value"
        else:
            parsed = int(normalized)
            result = f"parsed:{parsed}"

    except ValueError:
        # :: FeatureBranch | name=synthetic_try_except_else_finally | branch=value_error
        result = "value-error"

    except TypeError:
        # :: FeatureBranch | name=synthetic_try_except_else_finally | branch=type_error
        result = "type-error"

    else:
        # :: FeatureBranch | name=synthetic_try_except_else_finally | branch=try_else
        result = f"else:{result}"

    finally:
        # :: FeatureTrace | name=synthetic_try_except_else_finally
        result = f"finally:{result}"

    # :: FeatureConverge | name=synthetic_try_except_else_finally | branches=try_body,direct_value,parse_attempt,value_error,type_error,try_else | into=main
    converged = f"try:{result}"

    # :: FeatureEnd | name=synthetic_try_except_else_finally
    return finish(converged)


def synthetic_try_finally_resume_feature(value: int) -> dict[str, str]:
    # :: FeatureStart | name=synthetic_try_finally_resume
    result = value

    # :: FeatureBranch | name=synthetic_try_finally_resume | branch=try_body
    try:
        result += 1

    finally:
        # :: FeatureTrace | name=synthetic_try_finally_resume
        result += 10

    # :: FeatureConverge | name=synthetic_try_finally_resume | branches=try_body | into=main
    result += 100

    # :: FeatureEnd | name=synthetic_try_finally_resume
    return finish(str(result))


def synthetic_try_direct_disruption_feature(value: int) -> dict[str, str]:
    # :: FeatureStart | name=synthetic_try_direct_disruption
    result = "unset"

    try:
        # :: FeatureBranch | name=synthetic_try_direct_disruption | branch=zero | control_polarity=true
        # :: FeatureBranch | name=synthetic_try_direct_disruption | branch=nonzero | control_polarity=false
        if value == 0:
            raise ValueError("zero")

        result = f"value:{value}"

    except ValueError:
        # :: FeatureBranch | name=synthetic_try_direct_disruption | branch=value_error
        result = "wrapped"

    finally:
        # :: FeatureTrace | name=synthetic_try_direct_disruption
        result = f"finally:{result}"

    # :: FeatureConverge | name=synthetic_try_direct_disruption | branches=zero,nonzero,value_error | into=main
    converged = f"try-direct:{result}"

    # :: FeatureEnd | name=synthetic_try_direct_disruption
    return finish(converged)


################################################################################
##  Flow tests: With + disruption
################################################################################


def synthetic_with_feature(path: str) -> dict[str, str]:
    # :: FeatureStart | name=synthetic_with
    handle: BufferedReader

    # :: FeatureTrace | name=synthetic_with
    with open(path, encoding="utf-8") as handle:
        # :: FeatureBranch | name=synthetic_with | branch=with_body
        content = handle.read()

    # :: FeatureBranch | name=synthetic_with | branch=has_content | control_polarity=true
    # :: FeatureBranch | name=synthetic_with | branch=empty | control_polarity=false
    if content:
        result = "has-content"
    else:
        result = "empty"

    # :: FeatureConverge | name=synthetic_with | branches=with_body,has_content,empty | into=main
    converged = f"with:{result}"

    # :: FeatureEnd | name=synthetic_with
    return finish(converged)


def synthetic_with_disruption_feature(path, mode: str) -> dict[str, str]:
    # :: FeatureStart | name=synthetic_with_disruption
    handle: BufferedReader
    result = "unset"

    # :: FeatureBranch | name=synthetic_with_disruption | branch=raise_mode | control_polarity=true
    # :: FeatureBranch | name=synthetic_with_disruption | branch=read_mode | control_polarity=false
    if mode == "raise":
        try:
            # :: FeatureTrace | name=synthetic_with_disruption
            with path.open("r") as handle:
                # :: FeatureBranch | name=synthetic_with_disruption | branch=with_raise_body
                handle.read()
                raise RuntimeError("forced failure")

        except RuntimeError:
            # :: FeatureBranch | name=synthetic_with_disruption | branch=runtime_error
            result = "raised"

    else:
        # :: FeatureTrace | name=synthetic_with_disruption
        with path.open("r") as handle:
            # :: FeatureBranch | name=synthetic_with_disruption | branch=with_read_body
            result = handle.read()

    # :: FeatureConverge | name=synthetic_with_disruption | branches=raise_mode,read_mode,with_raise_body,runtime_error,with_read_body | into=main
    converged = f"with-disruption:{result}"

    # :: FeatureEnd | name=synthetic_with_disruption
    return finish(converged)


def synthetic_with_loop_disruption_feature(values: list[str], path) -> dict[str, str]:
    # :: FeatureStart | name=synthetic_with_loop_disruption
    total = 0
    handle: BufferedReader

    for value in values:
        # :: FeatureTrace | name=synthetic_with_loop_disruption
        with path.open("r") as handle:
            # :: FeatureBranch | name=synthetic_with_loop_disruption | branch=with_body
            handle.read()

            # :: FeatureBranch | name=synthetic_with_loop_disruption | branch=loop_continue | control_polarity=true
            # :: FeatureBranch | name=synthetic_with_loop_disruption | branch=not_continue | control_polarity=false
            if value == "skip":
                continue

            # :: FeatureBranch | name=synthetic_with_loop_disruption | branch=loop_break | control_polarity=true
            # :: FeatureBranch | name=synthetic_with_loop_disruption | branch=loop_consume | control_polarity=false
            if value == "stop":
                break

            total += 1

    # :: FeatureConverge | name=synthetic_with_loop_disruption | branches=with_body,loop_continue,not_continue,loop_break,loop_consume | into=main
    converged = f"with-loop:{total}"

    # :: FeatureEnd | name=synthetic_with_loop_disruption
    return finish(converged)


################################################################################
##  Flow tests: interunit call
################################################################################


def synthetic_interunit_call_feature(
    dependency: SyntheticDependency,
    value: str,
) -> dict[str, str]:
    # :: FeatureStart | name=synthetic_interunit_call
    normalized = normalize_input(value)

    # :: FeatureTrace | name=synthetic_interunit_call | operation=dependency_transform
    transformed = dependency.transform(normalized)

    # :: FeatureEnd | name=synthetic_interunit_call
    return finish(transformed)


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
