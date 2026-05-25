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
    with open(path, encoding="utf-8") as handle:
        content = handle.read()

    if content:
        return "has-content"

    return "empty"


def fixture_mixed_control(values: list[str], fallback: str) -> str:
    result = fallback

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
