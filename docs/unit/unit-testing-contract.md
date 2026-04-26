# Unit Testing Contract

## Overview

This document defines the rules and workflow for writing unit tests from the
artifacts produced by the Pybastion unit analysis pipeline. It is a contract:
**if a test violates this contract, it is not a unit test**.

This contract is designed to be used with a completed unit inventory. The unit
inventory is the authoritative source for the unit's structural entries,
unit-scoped bindings, callable-owned execution items (EIs), integration
candidates, execution paths, and analysis metadata.

**Language Note:** This contract currently targets Python tests, primarily using
pytest. Other languages may adopt the same concepts later, but this document
should be read as Python-focused unless a language-specific contract says
otherwise.

---

## Agent Posture

The unit analysis pipeline is automated. The agent does not generate the
inventory or reinterpret the analysis. The agent reads the generated inventory
together with the source code and writes tests that satisfy the reachable EIs
described by the inventory.

This is a mechanical transformation exercise. The agent's job is to turn
inventory facts into runnable test cases with inputs, mocks, fixtures, and
assertions. The source code may be inspected to understand how to trigger and
assert an inventory fact, but it must not replace the inventory as the source of
test obligations.

The agent should:

- read the inventory
- inspect the source only to clarify how to exercise the inventory facts
- create case rows that cover every reachable EI
- mock unit boundaries and external influences
- write runnable tests
- mark genuinely blocked EIs instead of guessing

The agent should not redesign the analysis, skip reachable EIs, invent expected
behavior, or substitute subjective judgment for inventory-derived obligations.

---

## 1. Definitions

### 1.1 Unit Test

A **unit test** exercises code within a single compilation unit in isolation
from:

- Other units in the same project
- External systems, such as network, filesystem, database, subprocesses, or
  message buses
- Non-deterministic influences, such as time, randomness, and environment

In the current Python implementation, the unit is a module.

A unit test:

- **Focuses on one unit** – the code defined in the target module only
- **Mocks all dependencies** – everything outside the unit boundary
- **Runs fast** – no I/O, no network, no subprocess calls
- **Is deterministic** – same inputs always produce the same results
- **Is isolated** – tests can run in any order, including in parallel

### 1.2 Coverage Goal

**Primary goal:** Achieve 100% reachable EI coverage for callable-owned
execution items in the unit inventory.

EI coverage means every reachable execution item enumerated for the unit is
exercised at least once. This includes:

- Every conditional branch/path, such as if/else, match/case, and ternary
  outcomes
- Every loop outcome represented by the inventory, such as zero iterations and
  one-or-more iterations
- Every exception path represented by the inventory
- Every early exit, such as return, raise, yield, break, or continue
- Every sequential executable statement represented by an EI

**Secondary goal:** Verify exposed unit contracts that do not own EIs but still
matter to callers, such as enum values, dataclass defaults, public constants, or
other language-level declarations.

Do not mix these categories. EI coverage proves callable execution behavior.
Contract verification protects exposed unit contracts that are not represented
as callable execution flow.

**Scope:** Coverage applies only to the unit under test. Coverage gained by
exercising external code does not count and should be avoided through mocking.

### 1.3 Branch Coverage vs EI Coverage

**Industry term:** "Branch coverage" traditionally means exercising all
conditional branches.

**This contract:** Uses "EI coverage" to be precise about what is being
enumerated. An EI, or Execution Item, is a callable-owned execution fact that
represents a distinct executable outcome.

For communication with traditional coverage tools, "branch coverage" may still
be used. Internally, use "EI coverage" for precision.

### 1.4 Key Terms

**Unit Inventory:** The per-unit artifact produced by the unit analysis
pipeline. It contains unit metadata, unit-scoped bindings, structural entries,
callable-owned EIs, integration candidates, execution paths, and summary
information.

**Entry:** A structural item in the unit inventory, such as a class, function,
method, nested class, or nested function.

**Binding:** A unit-scoped name used for resolution or traceability. Bindings
do not own EIs and are not callable entries. They may still be useful when
understanding callable behavior.

**EI ID:** Unique identifier for an execution item, such as
`U1234567890_F001_E0003` or `U1234567890_C001_M001_E0003`. Every reachable EI
in the inventory must be covered by at least one test.

**Case Row:** The smallest unit of test intent. Specifies:

- Input values
- Expected outcome, such as return value or exception
- `covers`: list of EI IDs exercised by this case
- Patch targets or fixtures needed for the case

**Bucket:** A group of case rows that share the same harness key. Each bucket
becomes one test function.

**Harness Key:** A tuple of facts used to partition case rows into buckets:

- Callable ID
- Outcome kind, such as returns or raises
- Patch targets, represented as a sorted list of mocks needed

**Integration Candidate:** Information from the unit inventory about an
operation that may cross a unit, project, library, or external system boundary.
Integration candidates are used to determine what to mock and how to reach the
operation.

**Execution Path:** A sequence of EI IDs that must execute to reach a particular
point in the code. Execution paths are used to trace how to trigger specific EIs
or integration candidates.

**Blocked EI:** An EI that cannot be tested without information not present in
the unit, inventory, source code, project conventions, or provided context.
Blocked EIs do not prevent progress on other EIs.

**Unreachable EI:** An EI that cannot be executed without violating unit
boundaries or creating an impossible state. Unreachable EIs are excluded from
the coverage target when the inventory or analysis findings identify them as
unreachable.

---

## 2. Hard Rules (Non-Negotiable)

### 2.1 Unit Boundaries

**What counts as "the unit":**

- Code defined in the target module only
- Classes, functions, methods, and nested callables in that module
- Unit-scoped bindings in that module, as context

**Forbidden:**

- Calling upstream orchestrators
- Calling adjacent units because it is convenient
- Letting external code execute unmocked
- Using integration tests to satisfy unit test obligations

**Required:**

- Mock or stub everything outside the target unit
- Use integration candidates from the inventory to identify what needs mocking
- Patch at the call site used by the unit under test

### 2.2 External Influences Must Be Mocked

Mock or stub all external influences, including but not limited to:

**Boundary crossings:**

- **Network:** HTTP, sockets, APIs
- **Filesystem:** File I/O, directory operations
- **Database:** Queries, transactions
- **Subprocess:** Shell commands, external programs
- **Message Bus:** Queue publish/consume
- **Clock:** Current time, timestamps
- **Randomness:** Random numbers, UUIDs
- **Environment:** Environment variables

**Interunit calls:**

- Calls to other project modules
- Calls to classes or functions in different files
- Dynamic discovery mechanisms, such as plugins, services, and entrypoints

**Third-party libraries:**

- External package registries
- HTTP clients such as requests or httpx
- Database drivers
- Async scheduling
- Thread pools

If you are about to introduce an unmocked external influence, stop and refactor
the test to mock it.

### 2.3 Patch Target Rule

The integration candidate target describes the operation being invoked. The
mock patch target must be the name as resolved at the call site in the unit
under test.

For Python, patch where the unit looks up the symbol, not where the symbol was
originally defined. If the inventory provides both an observed target expression
and a resolved target, prefer the observed call-site expression for patch
placement and the resolved target for understanding what behavior is being
substituted.

Example:

```python
# myunit.py
import requests

def fetch_data(url: str) -> dict:
    return requests.get(url).json()
```

Patch:

```python
@patch("myunit.requests.get")
def test_fetch_data(mock_get):
    ...
```

Do not patch:

```python
@patch("requests.get")
def test_fetch_data(mock_get):
    ...
```

unless the unit under test looks up the symbol through that name.

### 2.4 Unit-Scoped Bindings

Unit-scoped bindings are not test targets by themselves. They do not own EIs and
should not receive unit tests merely because they are listed in the inventory.

Bindings should be used as context for understanding callable behavior. For
example, a compiled regex binding may explain why a callable invokes
`_SHA256_RE.match(value)`, but the test target is the callable behavior that
uses the binding, not the binding assignment itself.

If a binding establishes a public contract, such as an exported constant, enum
value, dataclass default, or TypeVar constraint, that may be tested as contract
verification rather than EI coverage.

### 2.5 No Invented Expectations

The agent must not invent expected return values, exception messages, mock
contracts, fixture behavior, or domain rules that are not supported by the unit
inventory, source code, project conventions, or provided context.

When the expected behavior is unclear, either:

- create a blocked case, or
- use a minimal assertion that proves the EI outcome without overclaiming

A minimal assertion is acceptable when it is sufficient to prove the relevant
execution outcome. Do not make tests look stronger by guessing details that are
not supported by evidence.

### 2.6 Test Structure Rules

**Function/Method Structure:**

- Do not nest test classes by default
- Use plain `def test_...():` functions unless project conventions say
  otherwise
- Projects may override this rule with explicit written instructions

**Parameterization:**

- Prefer pytest parameterization for multiple cases
- Do not copy/paste test blocks or functions for multiple cases
- Do not manually loop over test cases inside a single test function

**Example:**

```python
# Good: framework parameterization
@pytest.mark.parametrize("input_value,expected", [
    (5, "positive"),
    (-3, "negative"),
    (0, "zero"),
])
def test_classify_value(input_value, expected):
    # covers: see parameter row
    assert classify_value(input_value) == expected


# Bad: manual loop
def test_classify_value():
    for input_value, expected in [(5, "positive"), (-3, "negative"), (0, "zero")]:
        assert classify_value(input_value) == expected
```

### 2.7 Selection Behavior Rule

If code selects the best item or uses ranked selection:

**Forbidden:**

- Relying on lexical sorting as a proxy for preference
- Asserting on incidental or arbitrary ordering

**Required:**

- Selection must follow the explicit preference order in the code under test
- Assert based on the code's ordering rules
- Use the actual comparison or ranking logic

### 2.8 Unreachable EIs

If you cannot reach an EI without violating unit boundaries or requiring an
impossible state:

- The inventory or related findings should mark it unreachable
- Do not write fake tests just to hit the coverage number
- Document why it is unreachable

Abstract methods and interface-like signatures are not automatically
unreachable. Test them by introducing a test implementation that exercises the
abstract contract when that contract belongs to the unit.

---

## 3. Mandatory Workflow

### 3.1 Prerequisites

**Before writing any tests:**

1. Generate the unit inventory using the unit analysis pipeline.
2. Validate the inventory.
3. Review any summary, findings, ambiguity notes, unresolved analysis facts, or
   blocked/unreachable indicators.

If you start writing tests before the inventory exists, stop and generate the
inventory first.

The inventory is the authoritative source for:

- Callables in the unit
- EI IDs and their outcomes
- Unit-scoped bindings
- Integration candidates requiring mocks
- Execution paths to integration candidates
- Any unresolved or ambiguous analysis facts

### 3.2 Artifact Reading Order

Before writing tests, inspect the available unit analysis artifacts in this
order:

1. The unit inventory for the target unit.
2. The `entries` hierarchy to identify callables, classes, methods, enums, and
   contract-like constructs.
3. The top-level `bindings` collection to understand unit-scoped names used by
   callables.
4. Each callable's `analysis_info.branches` to identify reachable EIs and
   outcomes.
5. Each callable's `analysis_info.integration_candidates` to identify required
   mocks, fixtures, and patch targets.
6. Execution paths attached to integration candidates to understand how to
   reach those operations.
7. Any summary, findings, ambiguity notes, or unresolved analysis facts.

Do not infer missing behavior from source alone when the inventory already
contains a more specific analyzed fact. Use source inspection to clarify or
confirm, not to contradict the inventory without evidence.

### 3.3 The Test Generation Procedure

After the unit inventory exists, generate unit tests using this deterministic
7-stage procedure. The procedure is mechanical and must not involve searching
for an optimal layout.

The procedure produces a runnable test skeleton early. Later stages refine
assertions and reduce duplication, but must not change the inventory-derived
EI-to-case mapping without a specific reason.

This procedure is mechanical:

- transform inventory facts into test cases
- implement those cases via stages explained below

Do not use this stage to redesign the analysis, search for a better strategy,
or replace inventory-derived obligations with your own (subjective) judgment.

---

### Stage 1: Create Case Rows

**Purpose:** Create a set of test cases that covers every reachable EI at least
once.

**Process:**

For each callable in inventory order:

1. Inspect the callable's reachable EIs in EI ID order.
2. Use EI outcomes, execution paths, and integration candidates to determine
   how each reachable EI can be triggered.
3. Create the smallest clear set of case rows that covers every reachable EI at
   least once without obscuring intent.
4. Record which EI IDs each case row covers.
5. Determine the expected outcome, such as return value or exception.
6. Determine patch targets and fixtures needed for the case.

A case row may cover multiple EIs when a single execution path naturally
exercises them together. Do not create one test per EI if a smaller set of
clear path-based cases covers the same behavior.

**Case Row Structure:**

```yaml
{
  "callable_id": "U1234567890_F001",
  "ei_ids": ["U1234567890_F001_E0003", "U1234567890_F001_E0005"],
  "inputs": {"x": -5},
  "outcome_kind": "returns",
  "expected": "negative",
  "patch_targets": []
}
```

**Rules:**

- Every reachable EI ID must appear in at least one case row.
- Skip unreachable EIs identified by the inventory or related findings.
- If an EI cannot be mapped to inputs with certainty, mark it BLOCKED.
- Use integration candidates from the inventory to determine patch targets.
- Use execution paths to trace how to reach specific EIs and integration
  candidates.
- Do not invent expected behavior unsupported by the code and inventory.
- Treat the inventory as the test obligation source: do not skip reachable EIs
  because they seem unimportant, redundant, or unlikely.

**Using Execution Paths:**

Integration candidates in the inventory may contain execution paths: sequences
of EI IDs that lead to that integration point. Use these to:

- Trace what conditions must be true to reach an integration
- Determine which EIs must execute before the integration
- Set up mocks at the right point in the execution flow
- Identify natural path-based case rows

Example:

```yaml
integration_candidates:
  - id: U1234567890_F002_I0007
    ei_id: U1234567890_F002_E0007
    target: requests.get
    execution_paths:
      - [
          U1234567890_F002_E0001,
          U1234567890_F002_E0003,
          U1234567890_F002_E0007
        ]
```

This tells you that to reach the `requests.get` call at `E0007`, the test input
and mocks must trigger `E0001`, then `E0003`, and then the integration EI
itself.

**Output:** Complete list of case rows for the entire unit.

**Verification Gate:**

- [ ] Every reachable EI ID appears in at least one case row
- [ ] Every case row has inputs, expected outcome, and `covers` list
- [ ] Every required patch target is identified or marked blocked
- [ ] Blocked EIs are marked and do not prevent progress

---

### Stage 2: Partition Into Buckets

**Purpose:** Group case rows that can share test setup and teardown.

**Process:**

For each case row, compute its harness key:

```python
harness_key = (
    callable_id,
    outcome_kind,
    tuple(sorted(patch_targets)),
)
```

Group case rows by identical harness keys. Each group is a bucket.

**Why This Key:**

- **callable_id:** Tests for different functions should be separate.
- **outcome_kind:** Mixing returns and raises in one test is awkward.
- **patch_targets:** Shared mock setup is useful; different mocks often require
  different harnesses.

**Rules:**

- Do not backtrack without a specific reason.
- Keep the key small to avoid combinatorial explosion.
- If the inventory provides different patch targets through integration
  candidates, respect them.
- Do not move a case into a bucket where its required mocks would be wrong or
  misleading.

**Example:**

Case rows:

```yaml
[
  {
    "callable_id": "U1234567890_F001",
    "outcome_kind": "returns",
    "patch_targets": [],
    "covers": ["U1234567890_F001_E0001"]
  },
  {
    "callable_id": "U1234567890_F001",
    "outcome_kind": "returns",
    "patch_targets": [],
    "covers": ["U1234567890_F001_E0002"]
  },
  {
    "callable_id": "U1234567890_F001",
    "outcome_kind": "raises",
    "patch_targets": [],
    "covers": ["U1234567890_F001_E0003"]
  },
  {
    "callable_id": "U1234567890_F002",
    "outcome_kind": "returns",
    "patch_targets": ["myunit.requests.get"],
    "covers": ["U1234567890_F002_E0005"]
  }
]
```

Buckets:

```python
{
    ("U1234567890_F001", "returns", ()): [case1, case2],
    ("U1234567890_F001", "raises", ()): [case3],
    ("U1234567890_F002", "returns", ("myunit.requests.get",)): [case4],
}
```

**Output:** Buckets of case rows, keyed by harness key.

**Verification Gate:**

- [ ] Every case row is in exactly one bucket
- [ ] Bucket keys are computed correctly
- [ ] No case rows are lost or duplicated

---

### Stage 3: Realize Test Functions

**Purpose:** Convert buckets into test functions.

**Process:**

For each bucket:

- If a bucket has multiple case rows, create a parameterized test function.
- If a bucket has one case row, create a non-parameterized test function.

**Naming Convention:**

Use names that identify the callable, outcome, and meaningful distinguisher.

Examples:

- `test_classify_value_returns`
- `test_classify_value_raises_for_invalid_input`
- `test_fetch_user_data_returns_remote_payload`
- `test_fetch_user_data_raises_on_api_error`

**Coverage Annotation:**

Every test function must reference the EI IDs it covers:

```python
def test_classify_value_returns(input_value, expected):
    # covers: U1234567890_F001_E0001, U1234567890_F001_E0002
    assert classify_value(input_value) == expected
```

For parameterized tests, include coverage per case row when the rows cover
different EIs:

```python
@pytest.mark.parametrize("input_value,expected,covers", [
    (-3, "negative", ["U1234567890_F001_E0001"]),
    (0, "zero", ["U1234567890_F001_E0002", "U1234567890_F001_E0003"]),
])
def test_classify_value_returns(input_value, expected, covers):
    # covers: see parameter
    assert classify_value(input_value) == expected
```

**Output:** Test skeleton where all test functions are defined and runnable,
even if some assertions are still minimal.

**Verification Gate:**

- [ ] Every bucket has exactly one test function
- [ ] Parameterized tests use pytest parameterization
- [ ] Every test has a coverage annotation
- [ ] Test skeleton is syntactically valid and runnable

---

### Stage 4: Micro Review (Optional, Bounded)

**Purpose:** Improve clarity without introducing churn.

**When:** Only if no case rows are blocked.

**Allowed Operations:**

- Merge buckets when harness keys are identical
- Split a bucket to isolate an uncertain case
- Rename test functions for readability and consistency
- Adjust parameterization for clarity

**Forbidden Operations:**

- Searching for an optimal grouping
- Revisiting EI-to-case mapping without a specific reason
- Inventing new EIs or case rows
- Deep semantic analysis
- Major refactoring

**Process:** Single pass over the bucket list. After one pass, stop.

**Output:** Refined test skeleton.

**Verification Gate:**

- [ ] No EI IDs were added or removed
- [ ] Every EI ID is still covered by the same test or an intentionally
      equivalent replacement
- [ ] Changes improve clarity only

---

### Stage 5: Implementation (Coverage First)

**Purpose:** Implement the test functions to achieve full reachable EI coverage.

**Process:**

Implement test functions bucket by bucket, in inventory order.

1. **Set up mocks**
   - Use integration candidates to identify mock targets.
   - Mock at the call site used by the unit under test.
   - Use project-provided fakes or fixtures when available.

2. **Call the function under test**
   - Use the case row inputs.
   - Configure mocks according to the execution path being exercised.

3. **Assert the outcome**
   - For return cases: assert return value or observable behavior.
   - For exception cases: assert exception type and key message substrings.
   - For state mutation cases: assert the externally visible unit-local state
     change.

4. **Verify coverage**
   - Run tests with a coverage tool capable of tracking lines and branches.
   - Confirm all EI IDs in the case row's `covers` list are exercised.

**Constraints:**

- Follow mocking rules.
- Do not skip to later callables unless project workflow requires it.
- Prefer minimal assertions sufficient to prove the EI outcome.
- Do not invent expectations.
- If expected behavior is unclear, mark the EI blocked or use a minimal
  evidence-backed assertion.

**Example Implementation:**

```python
from unittest.mock import Mock, patch

import pytest

from myunit import fetch_user_data


@patch("myunit.requests.get")
def test_fetch_user_data_returns(mock_requests_get):
    # covers: U1234567890_F004_E0002, U1234567890_F004_E0004, U1234567890_F004_E0006
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "123", "name": "Alice"}
    mock_requests_get.return_value = mock_response

    result = fetch_user_data("123")

    assert result == {"id": "123", "name": "Alice"}
    mock_requests_get.assert_called_once_with("https://api.example.com/users/123")


def test_fetch_user_data_raises_on_empty_id():
    # covers: U1234567890_F004_E0001, U1234567890_F004_E0003
    with pytest.raises(ValueError, match="user_id required"):
        fetch_user_data("")


@patch("myunit.requests.get")
def test_fetch_user_data_raises_on_api_error(mock_requests_get):
    # covers: U1234567890_F004_E0002, U1234567890_F004_E0004, U1234567890_F004_E0005
    mock_response = Mock()
    mock_response.status_code = 404
    mock_requests_get.return_value = mock_response

    with pytest.raises(RuntimeError, match="API error: 404"):
        fetch_user_data("999")
```

**Deliverable:** Runnable test module where all test functions execute and cover
all reachable EI IDs.

**Note:** If using an AI agent and the agent cannot run tests itself, manual
intervention is required:

1. Run the tests.
2. Provide the error output to the agent.
3. The agent makes corrections.
4. Repeat until the tests pass.

**Verification Gate:**

- [ ] All test functions are implemented
- [ ] Tests are runnable
- [ ] All reachable EI IDs are covered
- [ ] All mocks are set up correctly
- [ ] Unit isolation is maintained

---

### Stage 6: Refinement (Bounded)

**Purpose:** Strengthen assertions and reduce duplication without changing the
structure unnecessarily.

**When:** After Stage 5 is complete and all tests pass.

**Process:** Single-pass refinement, bounded to one sweep.

**Allowed Improvements:**

- Stronger assertions based on explicit unit behavior
- Reduced duplication through parameterization or helpers
- Clearer naming and readability
- Improved mock clarity and call site correctness
- Better exception assertions, such as more specific message substrings

**Forbidden Changes:**

- Changing EI-to-case mapping without a specific reason
- Removing EI coverage
- Structural rewrites
- Coupling to implementation details
- Adding expectations that are not supported by the unit inventory, source, or
  project conventions

**Stop Condition:** Stop when improvements require guessing, broad refactoring,
or subjective redesign.

**Verification Gate:**

- [ ] No EI coverage lost
- [ ] Tests still pass
- [ ] Assertions are stronger only where supported
- [ ] Duplication reduced only where it improves clarity

---

### Stage 7: Stop Condition

Stop when:

- All reachable EI IDs are covered
- Unit isolation constraints are satisfied
- All external influences are mocked
- Assertions provide sufficient proof without brittleness or invention
- Blocked EIs, if any, are clearly reported

**Verification:** Run the coverage tool and confirm that the tests provide 100%
reachable EI coverage for the unit.

**If coverage is less than 100%:**

- Check the inventory for unreachable EIs and exclude them from the coverage
  target.
- Check for blocked EIs.
- Identify which EI IDs are missing coverage.
- Add case rows for missing EIs.
- Return to Stage 2 to integrate the new case rows.

**Output:** Complete unit test module, ready for commit.

---

## 4. Blocked EI Protocol

### 4.1 Purpose

A blocked EI must not prevent progress. If any EI IDs cannot be tested without
guessing, complete all unblocked EIs first, then report blocked EIs explicitly.

### 4.2 Definition of Blocked

An EI ID is **BLOCKED** if reaching it, or asserting its outcome, requires
missing information that cannot be derived from the unit inventory, source code,
project conventions, available fixtures, or provided context without guessing.

**Common causes:**

- Unknown callable signature required to call the code
- Branch trigger cannot be mapped from inventory facts, source, or available
  fixtures
- Expected exception type or message substring is unavailable
- Required fake, fixture, or project test utility is referenced but unavailable
- The observed call-site patch target is ambiguous
- The integration candidate lacks enough contract information to construct a
  safe mock response

Blocked status is determined per EI ID.

### 4.3 Required Output

When any EI IDs are blocked, the test file must include a blocked EI comment
block and placeholder tests.

**1. BLOCKED EI IDs Comment Block:**

```python
# BLOCKED EI IDs
# - <EI_ID>:
#     why: <reason the EI is blocked>
#     need: <concrete artifact needed to unblock>
#     impact: <scope of impact>
#     info: <optional brief additional context>
#     action: <what user can provide to unblock>
```

Fields:

- `why`: Reason the EI is blocked
- `need`: Minimal missing input, such as code snippet, signature, exception
  type, fixture name, or patch target
- `impact`: "Localized to this EI" or list of other impacted EI IDs
- `info`: Optional additional context
- `action`: Single-sentence unblock action

**2. Placeholder Test:**

Create a pytest xfail test for each blocked EI:

```python
import pytest


@pytest.mark.xfail(reason="Blocked: missing trigger mapping for U1234567890_F002_E0004")
def test_blocked_U1234567890_F002_E0004():
    # covers: U1234567890_F002_E0004
    assert False, "EI blocked - see BLOCKED EI IDs comment"
```

Purpose: Keep blocked work visible and allow the user to supply missing inputs
and regenerate or revise tests later.

### 4.4 Example

```python
# BLOCKED EI IDs
# - U1234567890_F002_E0004:
#     why: trigger mapping is not derivable from inventory facts
#     need: code snippet or fixture contract showing which input controls the branch
#     impact: all other reachable EIs covered; this one pending
#     info: inventory does not identify which input parameter controls this branch
#     action: provide the branch trigger mapping or the missing fixture contract

import pytest


@pytest.mark.xfail(reason="Blocked: missing trigger mapping for U1234567890_F002_E0004")
def test_blocked_U1234567890_F002_E0004():
    # covers: U1234567890_F002_E0004
    assert False, "EI blocked - see BLOCKED EI IDs comment"
```

### 4.5 No-Stall Requirement

If blocked EIs exist:

- Do not stop test generation.
- Do not expand analysis indefinitely.
- Emit runnable tests for all unblocked EIs first.
- Then emit the blocked EI report and placeholders.
- Move forward.

---

## 5. Test Writing Rules

### 5.1 Mocking Strategy

**Using integration candidates from the inventory:**

The unit inventory contains integration candidates that indicate what may need
mocking:

```yaml
analysis_info:
  integration_candidates:
    - id: U1234567890_F001_I0004
      ei_id: U1234567890_F001_E0004
      target: validate_typed_dict
      classification: interunit
      execution_paths:
        - [U1234567890_F001_E0004]

    - id: U1234567890_F001_I0007
      ei_id: U1234567890_F001_E0007
      target: requests.get
      classification: boundary
      boundary:
        kind: network
        protocol: http
      execution_paths:
        - [U1234567890_F001_E0003, U1234567890_F001_E0007]
```

This tells you:

1. **What to mock:** `validate_typed_dict` as an interunit operation and
   `requests.get` as a boundary operation.
2. **Where to mock:** At the call site in the unit under test.
3. **How to reach it:** Follow `execution_paths` to determine what conditions
   trigger these calls.

**Mocking Rules:**

1. Mock at the call site used by the unit under test.
2. Use the project's standard mocking mechanism.
3. Use project-provided fakes and fixtures when available.
4. Mock all boundaries identified in integration candidates.
5. Mock all interunit calls unless the project explicitly allows same-process
   collaboration for that case.
6. Do not let adjacent units execute accidentally.

**Example:**

```python
from unittest.mock import Mock, patch


@patch("myunit.requests.get")
def test_fetch_data(mock_get):
    # Set up mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": "success"}
    mock_get.return_value = mock_response

    # Execute
    result = fetch_data("test")

    # Assert
    assert result == {"result": "success"}
    mock_get.assert_called_once()
```

### 5.2 Assertions

**For return cases:**

- Assert return value matches expected.
- Use equality checks for simple values.
- Use structural assertions for complex objects.
- Assert key fields, not entire object graphs, unless the object graph is the
  explicit contract.

**For exception cases:**

- Assert the exception type.
- Assert key substrings in the message.
- Avoid full-message matching unless the full message is a stable public
  contract.
- Prefer user-meaningful wording, such as project names, version numbers,
  strategy names, URL patterns, or policy modes.

**Examples:**

```python
# Good: key substring
with pytest.raises(ValueError, match="Invalid environment key"):
    validate_config(bad_config)


# Bad: brittle full message
with pytest.raises(
    ValueError,
    match="Invalid environment key: DEBUG_MODE not in allowed keys: ['LOG_LEVEL', 'API_URL']",
):
    validate_config(bad_config)


# Good: behavior assertion
assert len(result) == 3
assert all(isinstance(item, Config) for item in result)


# Bad: implementation detail
assert result._internal_cache is not None
```

**Assertion Principles:**

- Assert behavior, not implementation details, unless EI coverage requires a
  narrow implementation-facing proof.
- Prefer minimal assertions sufficient to prove the EI outcome.
- Strengthen assertions in Stage 6 if deterministic improvements exist.
- Stop strengthening when improvements require guessing.

### 5.3 EI Coverage Labeling

Every test must state which EI IDs it covers.

**Format:** Comment with `covers:` prefix.

```python
def test_classify_value_positive():
    # covers: U1234567890_F001_E0001, U1234567890_F001_E0003
    assert classify_value(5) == "positive"
```

**For parameterized tests:**

Option 1: Include `covers` in parameter data.

```python
@pytest.mark.parametrize("input_value,expected,covers", [
    (5, "positive", ["U1234567890_F001_E0001", "U1234567890_F001_E0003"]),
    (-3, "negative", ["U1234567890_F001_E0002", "U1234567890_F001_E0004"]),
])
def test_classify_value(input_value, expected, covers):
    # covers: see parameter
    assert classify_value(input_value) == expected
```

Option 2: List all covered IDs in the function comment when every parameter row
belongs to the same clear coverage bucket.

```python
@pytest.mark.parametrize("input_value,expected", [
    (5, "positive"),
    (-3, "negative"),
])
def test_classify_value(input_value, expected):
    # covers: U1234567890_F001_E0001, U1234567890_F001_E0002, U1234567890_F001_E0003, U1234567890_F001_E0004
    assert classify_value(input_value) == expected
```

**Why this matters:**

- Enables verification that all EI IDs are covered.
- Makes coverage gaps visible during code review.
- Allows regeneration of specific tests when code changes.
- Documents the inventory-to-test mapping.

---

## 6. Complete Worked Example

This section shows the full workflow from inventory to tests.

### 6.1 The Unit Under Test

```python
# validation.py
def classify_value(x: int) -> str:
    """Classify an integer as negative, zero, or positive."""
    if x < 0:
        return "negative"
    if x == 0:
        return "zero"
    return "positive"
```

### 6.2 The Unit Inventory Excerpt

```yaml
unit: validation
fully_qualified_name: validation
unit_id: U1234567890
entries:
  - id: U1234567890_F001
    kind: function
    name: classify_value
    signature_info:
      signature: "classify_value(x: int) -> str"
    analysis_info:
      branches:
        - id: U1234567890_F001_E0001
          condition: "x < 0 true"
          description: "returns negative"
        - id: U1234567890_F001_E0002
          condition: "x < 0 false"
          description: "continues to zero check"
        - id: U1234567890_F001_E0003
          condition: "x == 0 true"
          description: "returns zero"
        - id: U1234567890_F001_E0004
          condition: "x == 0 false"
          description: "continues to final return"
        - id: U1234567890_F001_E0005
          condition: "final return"
          description: "returns positive"
```

### 6.3 Stage 1: Case Rows

```python
case_rows = [
    {
        "callable_id": "U1234567890_F001",
        "ei_ids": ["U1234567890_F001_E0001"],
        "inputs": {"x": -5},
        "outcome_kind": "returns",
        "expected": "negative",
        "patch_targets": [],
    },
    {
        "callable_id": "U1234567890_F001",
        "ei_ids": ["U1234567890_F001_E0002", "U1234567890_F001_E0003"],
        "inputs": {"x": 0},
        "outcome_kind": "returns",
        "expected": "zero",
        "patch_targets": [],
    },
    {
        "callable_id": "U1234567890_F001",
        "ei_ids": [
            "U1234567890_F001_E0002",
            "U1234567890_F001_E0004",
            "U1234567890_F001_E0005",
        ],
        "inputs": {"x": 10},
        "outcome_kind": "returns",
        "expected": "positive",
        "patch_targets": [],
    },
]
```

All 5 reachable EI IDs are covered.

### 6.4 Stage 2: Buckets

All case rows have the same harness key:

```python
("U1234567890_F001", "returns", ())
```

Bucket result:

```python
buckets = {
    ("U1234567890_F001", "returns", ()): [case_row_1, case_row_2, case_row_3]
}
```

### 6.5 Stage 3: Test Function

```python
@pytest.mark.parametrize("x,expected,covers", [
    (-5, "negative", ["U1234567890_F001_E0001"]),
    (0, "zero", ["U1234567890_F001_E0002", "U1234567890_F001_E0003"]),
    (
        10,
        "positive",
        [
            "U1234567890_F001_E0002",
            "U1234567890_F001_E0004",
            "U1234567890_F001_E0005",
        ],
    ),
])
def test_classify_value_returns(x, expected, covers):
    # covers: see parameter
    assert classify_value(x) == expected
```

### 6.6 Stage 5: Implementation

The simple example is already implemented by the Stage 3 realization.

### 6.7 Stage 6: Refinement

Additional values may be added if they strengthen behavior verification without
creating new EI obligations:

```python
@pytest.mark.parametrize("x,expected,covers", [
    (-5, "negative", ["U1234567890_F001_E0001"]),
    (-1, "negative", ["U1234567890_F001_E0001"]),
    (0, "zero", ["U1234567890_F001_E0002", "U1234567890_F001_E0003"]),
    (
        1,
        "positive",
        [
            "U1234567890_F001_E0002",
            "U1234567890_F001_E0004",
            "U1234567890_F001_E0005",
        ],
    ),
    (
        10,
        "positive",
        [
            "U1234567890_F001_E0002",
            "U1234567890_F001_E0004",
            "U1234567890_F001_E0005",
        ],
    ),
])
def test_classify_value_returns(x, expected, covers):
    # covers: see parameter
    assert classify_value(x) == expected
```

### 6.8 Final Test File

```python
# test_validation.py
import pytest

from validation import classify_value


@pytest.mark.parametrize("x,expected,covers", [
    (-5, "negative", ["U1234567890_F001_E0001"]),
    (-1, "negative", ["U1234567890_F001_E0001"]),
    (0, "zero", ["U1234567890_F001_E0002", "U1234567890_F001_E0003"]),
    (
        1,
        "positive",
        [
            "U1234567890_F001_E0002",
            "U1234567890_F001_E0004",
            "U1234567890_F001_E0005",
        ],
    ),
    (
        10,
        "positive",
        [
            "U1234567890_F001_E0002",
            "U1234567890_F001_E0004",
            "U1234567890_F001_E0005",
        ],
    ),
])
def test_classify_value_returns(x, expected, covers):
    """Test classify_value with representative inputs.

    Covers all 5 reachable EI IDs for classify_value.
    """
    # covers: see parameter
    assert classify_value(x) == expected
```

---

## 7. Common Testing Patterns

This section identifies common EI categories to assist in creating case rows. If
supplemental project-specific or domain-specific instructions exist, they take
precedence.

### 7.1 Input Classification Units

Common EI buckets:

- Multiple accepted shapes or encodings, such as dict vs JSON string or
  absolute vs relative path
- Input validation, such as valid, invalid characters, empty, or null
- Normalization, such as case-insensitive matching, whitespace trimming, or path
  normalization
- Type coercion, such as string to int or flexible vs strict parsing

Example EIs:

- E0001: input is dict, parse as dict
- E0002: input is string, parse as JSON
- E0003: input is None, raise ValueError
- E0004: input is empty string, raise ValueError

### 7.2 Conditional Logic Units

Common EI buckets:

- Mode selection, such as strict vs permissive, debug vs normal, dry-run vs
  apply
- Feature gates, such as feature enabled or feature disabled
- Filtering, such as keep vs drop, empty results vs some results vs all results
- Early exits, such as precondition fails or precondition passes

Example EIs:

- E0001: strict mode, validate all fields
- E0002: permissive mode, validate required fields only
- E0003: validation passes, continue
- E0004: validation fails, raise exception

### 7.3 Collection Processing Units

Common EI buckets:

- Empty collection
- Single item
- Multiple items
- All filtered out
- Some pass filter

Example EIs:

- E0001: collection empty, return empty list
- E0002: collection has items, all filtered, return an empty list
- E0003: collection has items, some pass, return filtered list

### 7.4 Error Handling Units

Common EI buckets:

- Success path
- Expected errors
- Unexpected errors
- Retry logic

Example EIs:

- E0001: operation succeeds, return result
- E0002: operation raises ValueError, catch and handle
- E0003: operation raises TypeError, catch and handle
- E0004: operation raises other exception, propagate

### 7.5 Integration/Boundary Units

Common EI buckets:

- External call succeeds
- External call fails
- Fallback paths
- Caching paths

Example EIs:

- E0001: API call returns 200, parse and return data
- E0002: API call returns 404, raise NotFoundError
- E0003: API call times out, raise TimeoutError
- E0004: use cached data, skip API call

### 7.6 Strategy/Selection Units

Common EI buckets:

- Strategy selection
- Preference ordering
- Best match
- Ambiguity

Example EIs:

- E0001: no candidates, raise NoMatchError
- E0002: exactly one match, return it
- E0003: multiple matches, select by priority, return best
- E0004: multiple matches, ambiguous, raise AmbiguousMatchError

### 7.7 State Transition Units

Common EI buckets:

- Valid transitions
- Invalid transitions
- Idempotency
- Terminal states

Example EIs:

- E0001: current state is PENDING, transition to RUNNING, allowed
- E0002: current state is PENDING, transition to COMPLETE, forbidden
- E0003: current state is COMPLETE, any transition forbidden
- E0004: transition called twice, idempotent, no error

### 7.8 Language Construct Verification Units

Some language constructs establish contracts that should be verified even though
they do not create EIs in the unit inventory.

#### 7.8.1 Enum Value Contracts

**What to test:**

- Enum members exist
- Enum values match expected strings or integers
- No accidental deletions or modifications

**Why:** Protects against refactoring errors and typos that break the enum
contract.

**Example:**

```python
def test_requires_dist_url_policy_enum_values():
    """Verify RequiresDistUrlPolicy enum contract."""
    assert RequiresDistUrlPolicy.HONOR.value == "honor"
    assert RequiresDistUrlPolicy.IGNORE.value == "ignore"
    assert RequiresDistUrlPolicy.RAISE.value == "raise"
    assert set(RequiresDistUrlPolicy) == {
        RequiresDistUrlPolicy.HONOR,
        RequiresDistUrlPolicy.IGNORE,
        RequiresDistUrlPolicy.RAISE,
    }
```

**Inventory relationship:** Enum definitions create no callable-owned EIs unless
there is explicit executable source associated with them. This test verifies the
enum contract, not execution paths.

#### 7.8.2 Data Class Contracts

**What to test:**

- Field existence and default values
- Immutability constraints, such as `frozen=True`
- Basic construction with and without arguments

**Why:** Data class constructs may generate behavior outside the explicit source
body. Tests should verify the public contract established by the decorator.

**Example:**

```python
def test_resolution_result_dataclass_contract():
    """Verify ResolutionResult dataclass contract."""
    result = ResolutionResult()
    assert result.requirements_by_env == {}
    assert result.resolved_wheels_by_env == {}

    result = ResolutionResult(
        requirements_by_env={"env1": "reqs"},
        resolved_wheels_by_env={"env1": ["wheel1"]},
    )
    assert result.requirements_by_env == {"env1": "reqs"}
    assert result.resolved_wheels_by_env == {"env1": ["wheel1"]}

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.requirements_by_env = {}
```

**Inventory relationship:** Dataclass-generated methods do not create
callable-owned EIs in the unit inventory unless they are explicitly represented
in source. This test verifies the dataclass contract, not execution paths in the
unit.

#### 7.8.3 Constants and Unit-Scoped Binding Contracts

**What to test:**

- Public constants have expected values
- Exported maps or tables contain expected keys
- Stable public aliases point to the expected objects
- Publicly meaningful defaults remain unchanged

**Why:** Some unit-scoped bindings establish public behavior even though the
binding itself is not an EI owner.

**Example:**

```python
def test_supported_hash_algorithms_contract():
    assert SUPPORTED_HASH_ALGORITHMS == {"sha256", "sha384", "sha512"}
```

**Inventory relationship:** The binding appears in the inventory as unit-scoped
context. This kind of test verifies a public contract, not EI coverage.

#### 7.8.4 When to Use Construct Verification

Use construct verification tests when:

- Language features auto-generate behavior
- A unit-scoped binding establishes a public contract
- Changes to the contract would break callers
- The behavior is not represented as callable-owned EIs

Do not use construct verification for:

- Methods explicitly written in the source file
- Internal implementation details that are not exposed as contracts
- Constructs where the language guarantee is enough and the project has no
  reason to protect a specific value or shape

---

## Appendix A: Quick Reference

### A.1 Workflow Checklist

- [ ] Generate unit inventory
- [ ] Validate inventory
- [ ] Review summary, findings, ambiguity notes, and unresolved analysis facts
- [ ] Read entries, bindings, branches, integration candidates, and execution
      paths
- [ ] Stage 1: Create case rows
- [ ] Stage 2: Partition into buckets
- [ ] Stage 3: Realize test functions
- [ ] Stage 4: Micro review, if useful
- [ ] Stage 5: Implement tests
- [ ] Stage 6: Refine
- [ ] Stage 7: Verify 100% reachable EI coverage
- [ ] Run tests and confirm all pass
- [ ] Commit test file

### A.2 Harness Key Formula

```python
harness_key = (
    callable_id,
    outcome_kind,
    tuple(sorted(patch_targets)),
)
```

### A.3 Case Row Template

```python
{
    "callable_id": "<callable_id>",
    "ei_ids": ["<ei_id_1>", "<ei_id_2>", ...],
    "inputs": {"<param_name>": "<value>"},
    "outcome_kind": "returns | raises",
    "expected": "<return_value_or_exception>",
    "patch_targets": ["<mock_target_1>", ...],
}
```

### A.4 Coverage Comment Format

```python
# covers: U1234567890_F001_E0001, U1234567890_F001_E0002, U1234567890_F001_E0005
```

### A.5 Blocked EI Comment Format

```python
# BLOCKED EI IDs
# - <EI_ID>:
#     why: <reason>
#     need: <concrete artifact>
#     impact: <scope>
#     action: <what user can provide>
```

### A.6 Common Mock Patterns

```python
# Mock HTTP call
@patch("myunit.requests.get")
def test_fetch(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"data": "value"}


# Mock filesystem
@patch("myunit.open", mock_open(read_data="file contents"))
def test_read_file(mock_file):
    ...


# Mock time
@patch("myunit.time.time", return_value=1234567890)
def test_timestamp(mock_time):
    ...


# Mock random
@patch("myunit.random.random", return_value=0.5)
def test_random(mock_random):
    ...
```

---

## Appendix B: Terminology Map

- Branch → Execution Item, or EI
- Branch ID → EI ID
- Branch coverage → EI coverage, or branch coverage when talking to external
  tools
- Callable branches → Callable EIs
- Integration fact → Integration candidate
- Unit Ledger → Unit Inventory

When communicating with coverage tools or team members unfamiliar with the unit
inventory model, "branch coverage" is acceptable and understood. Internally, use
"EI" for precision.

---

## Appendix C: Validation Checklist

Before committing tests, verify:

**Inventory Alignment:**

- [ ] Test file exists for the unit
- [ ] Every reachable EI ID from the inventory has coverage
- [ ] Unreachable EIs are excluded from the coverage target
- [ ] Blocked EIs have placeholder tests
- [ ] Coverage comments reference current EI IDs

**Unit Isolation:**

- [ ] No unmocked external calls
- [ ] No unmocked interunit calls
- [ ] All boundaries are mocked
- [ ] Tests run fast
- [ ] Tests do not require network, filesystem, database, subprocess, clock, or
      randomness unless those influences are controlled

**Test Quality:**

- [ ] All tests pass
- [ ] Coverage tool reports 100% reachable EI coverage
- [ ] Every test has a coverage comment
- [ ] Parameterization is used where appropriate
- [ ] Assertions are specific and non-brittle
- [ ] Mock setup is clear and correct
- [ ] No expectations were invented without evidence

**Code Quality:**

- [ ] Test names are descriptive
- [ ] Duplication is reasonable or extracted to helpers/fixtures
- [ ] Code is readable and maintainable
- [ ] Tests follow project conventions

### C.1 Coverage Calculation Notes

**What counts toward EI coverage:**

- Only EIs enumerated in the unit inventory
- EIs in source code files within the unit boundary
- Reachable callable-owned EIs

**What does not count toward EI coverage:**

- Unit-scoped bindings
- Auto-generated code from decorators or language features
- Code in external libraries or frameworks
- Language runtime behavior
- Metaprogramming-generated methods not represented as callable-owned EIs

**Example scenarios:**

| Scenario                            | Has EIs? | Should Test?  | Test Type             |
|-------------------------------------|----------|---------------|-----------------------|
| Explicit method in source file      | Yes      | Yes           | EI coverage           |
| Dataclass `__init__`                | No       | Maybe         | Contract verification |
| Enum class definition               | No       | Maybe         | Contract verification |
| Public constant binding             | No       | Maybe         | Contract verification |
| Enum constructor call `MyEnum(val)` | Yes      | Yes           | EI coverage           |
| `@property` decorated method        | Yes      | Yes           | EI coverage           |
| Standard library call               | No       | Mock in tests | Not directly tested   |

Coverage tools may report branches in auto-generated code. These do not need to
be covered to meet the 100% EI coverage goal because they are not enumerated in
the unit inventory. If you specifically want to test auto-generated code or
language-level contracts, add construct verification tests separately.

---

**END OF CONTRACT**