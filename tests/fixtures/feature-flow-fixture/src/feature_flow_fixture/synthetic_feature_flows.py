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
