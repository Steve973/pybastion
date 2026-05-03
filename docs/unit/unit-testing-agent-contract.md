# Agent Unit Test Generation Contract

This document defines how an AI agent must generate Python unit tests from a
PyBastion unit inventory.

The task is not open-ended test design. The task is a constrained
inventory-to-test transformation.

The unit inventory already performs the hard analysis. It identifies the unit's
structural entries, callable-owned execution items, integration candidates,
execution paths, unit-scoped bindings, and analysis metadata. The agent's job is
to turn those facts into runnable tests.

The agent must not reanalyze the whole project, redesign the inventory, skip
reachable execution items, or invent behavior. The agent reads the inventory,
uses the source code only to implement inventory-derived case rows, and writes
tests that cover every reachable EI.

## Table of Contents

- [Primary Rule](#primary-rule)
- [Agent Priority Rules](#agent-priority-rules)
- [Inputs](#inputs)
- [Definitions](#definitions)
- [Hard Constraints](#hard-constraints)
- [Required Workflow](#required-workflow)
  - [Step 1: Read the Inventory](#step-1-read-the-inventory)
  - [Step 2: Build the EI Obligation Set](#step-2-build-the-ei-obligation-set)
  - [Step 3: Create Case Rows](#step-3-create-case-rows)
  - [Step 4: Partition Case Rows into Buckets](#step-4-partition-case-rows-into-buckets)
  - [Step 5: Realize Test Functions](#step-5-realize-test-functions)
  - [Step 6: Implement Tests](#step-6-implement-tests)
  - [Step 7: Report Blocked EIs](#step-7-report-blocked-eis)
  - [Step 8: Stop](#step-8-stop)
- [Patch Target Rules](#patch-target-rules)
- [Assertion Rules](#assertion-rules)
- [Blocked EI Protocol](#blocked-ei-protocol)
- [Required Final Output](#required-final-output)
- [Forbidden Behaviors](#forbidden-behaviors)
- [Checklist](#checklist)

## Primary Rule

The inventory is the work order. Do not audit the work order unless explicitly
instructed by the user.

The agent must not reinterpret, redesign, correct, or second-guess the
inventory. The agent must transform the inventory into tests.

Use the source code only to answer operational questions required to implement
a case row:

- how to import the callable
- how to construct inputs
- how to instantiate required classes
- how to assert the specified outcome
- where to patch an integration candidate

Do not use source inspection to create a competing analysis of what should be
tested.

If the inventory appears inconsistent with the source, do not resolve the
conflict independently. Report the inconsistency as blocked or ambiguous and
continue with all unaffected EIs.

## Agent Priority Rules

Follow these rules in order:

1. Treat the inventory as the work order.
2. Do not reinterpret the inventory.
3. Do not perform independent coverage analysis.
4. Do not decide that an inventory-listed reachable EI is unnecessary.
5. Use source only to implement inventory-derived case rows.
6. Preserve unit boundaries.
7. Mock all interunit calls and external influences.
8. Patch where the unit under test looks up the symbol.
9. Do not invent expected behavior.
10. If an EI cannot be tested without guessing, mark it blocked and continue.
11. Produce runnable tests for all unblocked EIs.
12. Stop when the work order is satisfied.

This is not an analysis task; it is a mechanical task. Do not search for the
perfect testing philosophy. Do not redesign the pipeline. Generate the tests.

## Allowed Source Inspection

Source inspection is allowed only to implement inventory-derived obligations.

Allowed uses:

- confirm import names
- inspect callable signatures
- construct valid inputs
- identify stable exception types or message substrings
- determine call-site patch targets
- understand how to instantiate objects required by the target callable
- identify project test conventions already present in nearby tests

Forbidden uses are hard stops, not suggestions. The agent must not do any of
the following:

- Do not replace the inventory's reachable EI set.
- Do not decide an EI is not worth testing.
- Do not invent new obligations not represented by the inventory.
- Do not reclassify integration candidates.
- Do not change patch requirements because another design seems cleaner.
- Do not perform any architecture review.
- Do not audit or correct the inventory unless explicitly asked.
- Do not override this contract with any other testing strategy.
- Do not reinterpret the task as an opportunity to redesign the workflow.
- Do not provide general advice instead of generating the requested tests.
- Do not perform general analysis of any kind.

**Do not** treat this as, "If it is not forbidden, it is allowed."

If any of these activities seem necessary, stop that activity and continue with
the inventory-to-test workflow. If it blocks a specific EI, mark that EI as
blocked or ambiguous and continue with unaffected EIs.

## Inputs

The agent should be given:

- the unit inventory for the target unit
- the source file for the target unit
- any available project test conventions
- any existing fixtures or fakes, if relevant

The unit inventory is the required input. If no inventory is available, stop and
ask for the inventory.

## Definitions

### Unit

For the current Python implementation, a unit is a module.

A unit test exercises code defined in that module while isolating it from other
project units and external systems.

### Execution Item

An Execution Item, or EI, is a callable-owned execution fact. It represents a
distinct executable outcome in a function or method.

Every reachable EI in the inventory must be covered by at least one test.

### Integration Candidate

An integration candidate is an operation that may cross a unit, project,
library, or external system boundary.

Integration candidates are used to determine what must be mocked and how to
reach the operation.

### Execution Path

An execution path is a sequence of EI IDs that reaches a specific EI or
integration candidate.

Use execution paths to determine which inputs, mocks, and conditions are needed
to trigger a case.

### Case Row

A case row is the smallest unit of test intent.

A case row specifies:

- callable ID
- covered EI IDs
- inputs
- expected outcome
- patch targets
- fixtures or mocks needed

### Bucket

A bucket is a group of case rows that can share one test function or harness.

Buckets are based on:

- callable ID
- outcome kind
- patch targets

### Blocked EI

A blocked EI is reachable in principle, but cannot be safely tested with the
available inventory, source, fixtures, and context without guessing.

Blocked EIs must be reported. They must not stop progress on unblocked EIs.

## Hard Constraints

### Unit Boundary

A unit test must not execute other project units or external systems unless
that behavior is explicitly part of the unit and controlled through a test
double.

Mock or stub:

- calls to other project modules
- network access
- filesystem access
- database access
- subprocess calls
- message bus interactions
- current time
- randomness
- environment access
- third-party services
- dynamic plugin or entrypoint discovery

### Inventory Authority

Do not skip reachable EIs because they look redundant, unimportant, obvious, or
annoying.

Do not add test obligations that are not supported by the inventory, source, or
explicit user instructions.

### No Invented Expectations

Do not invent:

- return values
- exception messages
- fixture behavior
- mock contracts
- domain rules
- ordering expectations
- hidden state expectations

When behavior is unclear, use a minimal evidence-backed assertion or mark the
EI blocked.

### No Wandering

Do not perform broad project analysis unless needed to satisfy a specific
inventory-derived test obligation.

Do not explain testing theory unless asked.

Do not produce a long plan and stop. The deliverable is runnable test code.

## Required Workflow

### Step 1: Read the Inventory

Read the inventory before writing tests.

Inspect, in order:

1. unit metadata
2. `entries`
3. unit-scoped `bindings`
4. callable analysis data
5. execution items
6. integration candidates
7. execution paths
8. summary or findings

Do not start writing tests until the reachable EI set is known.

### Step 2: Build the EI Obligation Set

Create the set of reachable callable-owned EI IDs.

Exclude only EIs explicitly identified as unreachable by the inventory or
related findings.

Do not exclude an EI because it seems covered by implication. Coverage must be
mapped to a concrete test case.

Output of this step:

```yaml
reachable_eis:
  - <EI_ID>
  - <EI_ID>
  - <EI_ID>

unreachable_eis:
  - id: <EI_ID>
    reason: <inventory-supported reason>
```

### Step 3: Create Case Rows

Create the smallest clear set of case rows that covers every reachable EI at
least once.

A single case row may cover multiple EIs when one natural execution path
exercises them together.

Do not create one test per EI unless that is actually the clearest mapping.

Case row shape:

```yaml
callable_id: <callable_id>
covers:
  - <EI_ID>
  - <EI_ID>
inputs:
  <name>: <value>
outcome_kind: returns | raises | mutates | calls | yields
expected: <expected result or exception>
patch_targets:
  - <patch target>
fixtures:
  - <fixture name>
notes: <optional>
```

Rules:

- Every reachable EI must appear in at least one case row.
- Every case row must have a reason to exist.
- Every integration candidate used by the path must have a mock or fixture.
- Every blocked EI must be recorded instead of guessed.
- Source code may be inspected to implement inputs and assertions required by the case row.
- Source code must not override the inventory obligations silently.

### Step 4: Partition Case Rows into Buckets

Partition case rows by harness key:

```python
harness_key = (
    callable_id,
    outcome_kind,
    tuple(sorted(patch_targets)),
)
```

Each bucket becomes one test function.

Rules:

- Do not endlessly search for a better grouping.
- Do not manually loop over cases inside a test.
- Use pytest parameterization for multiple rows in one bucket.
- Do not merge buckets with incompatible mocks or outcomes.

### Step 5: Realize Test Functions

For each bucket:

- create one test function
- parameterize it when the bucket has multiple case rows
- include EI coverage comments
- set up the required mocks and fixtures
- call the function or method under test
- assert the expected outcome

Every test must include a `covers:` comment.

Example:

```python
def test_classify_value_returns_positive():
    # covers: U1234567890_F001_E0002, U1234567890_F001_E0004
    assert classify_value(10) == "positive"
```

For parameterized tests:

```python
@pytest.mark.parametrize("value, expected, covers", [
    (-1, "negative", ["U1234567890_F001_E0001"]),
    (0, "zero", ["U1234567890_F001_E0002", "U1234567890_F001_E0003"]),
])
def test_classify_value_returns(value, expected, covers):
    # covers: see parameter
    assert classify_value(value) == expected
```

### Step 6: Implement Tests

Implement the tests bucket by bucket.

For each case row:

1. configure mocks
2. provide inputs
3. call the unit code
4. assert the result
5. verify relevant mock calls
6. preserve the EI coverage comment

Mock setup must follow the inventory's integration candidates and the patch
target rules below.

### Step 7: Report Blocked EIs

If an EI cannot be tested without guessing, mark it blocked.

Do not stop. Finish all unblocked tests first.

Blocked EIs require:

1. a blocked EI comment block
2. a pytest `xfail` placeholder

Example:

```python
# BLOCKED EI IDs
# - U1234567890_F002_E0004:
#     why: trigger mapping is not derivable from inventory or source
#     need: fixture contract showing which input controls this branch
#     impact: localized to this EI
#     action: provide the missing fixture contract


@pytest.mark.xfail(
    reason="Blocked: missing trigger mapping for U1234567890_F002_E0004"
)
def test_blocked_U1234567890_F002_E0004():
    # covers: U1234567890_F002_E0004
    assert False, "EI blocked - see BLOCKED EI IDs comment"
```

### Step 8: Stop

Stop when:

- every reachable EI is covered
- every blocked EI is reported
- every test preserves unit isolation
- every external influence is mocked
- tests are runnable
- assertions are evidence-backed
- no invented behavior remains

Do not keep refining indefinitely.

## Patch Target Rules

Patch where the unit under test looks up the symbol.

Use the inventory's observed target expression for patch placement when
available. Use the resolved target to understand what behavior is being
substituted.

Example source:

```python
# myunit.py
import requests

def fetch_data(url: str) -> dict:
    return requests.get(url).json()
```

Correct:

```python
@patch("myunit.requests.get")
def test_fetch_data(mock_get):
    ...
```

Wrong:

```python
@patch("requests.get")
def test_fetch_data(mock_get):
    ...
```

unless the unit under test actually looks up `requests.get` through that name.

## Assertion Rules

Assertions must be strong enough to prove the intended behavior, but not
stronger than the evidence supports.

For return cases:

- assert the return value when clear
- assert key fields for complex objects
- avoid full object graph assertions unless that is the public contract

For exception cases:

- assert exception type
- assert stable key message substrings
- avoid brittle full-message matches unless the full message is a contract

For mock interactions:

- assert calls that are part of the behavior
- do not assert incidental internal calls unless they are the relevant outcome

For unclear behavior:

- use a minimal assertion that proves the EI outcome, or
- mark the EI blocked

## Blocked EI Protocol

An EI is blocked when reaching it or asserting its outcome requires information
not available from:

- the inventory
- the source code
- project conventions
- available fixtures
- provided context

Common blocked reasons:

- missing trigger mapping
- ambiguous patch target
- unknown expected exception
- unavailable fixture or fake
- missing external contract
- unclear mock return shape
- impossible state not marked unreachable

Blocked EIs do not block progress on other EIs.

The agent must complete all unblocked tests first.

## Required Final Output

The final answer should provide:

1. the complete test file content
2. a concise coverage summary
3. a blocked EI summary, if any
4. any assumptions that affected the generated tests

The final answer should not include a long essay about the process.

Coverage summary format:

```text
Reachable EI coverage:
- covered: <n>
- blocked: <n>
- unreachable: <n>
- total reachable: <n>
```

Blocked summary format:

```text
Blocked EIs:
- <EI_ID>: <reason>; need <specific missing input>
```

If there are no blocked EIs, say:

```text
Blocked EIs: none
```

## Forbidden Behaviors

The agent must not:

- start writing tests before reading the inventory
- ignore reachable EIs
- use external code execution to satisfy unit obligations
- let interunit calls execute unmocked
- patch where the dependency is defined instead of where it is looked up
- invent expected behavior
- invent missing fixture contracts
- silently reinterpret the inventory
- turn blocked EIs into fake passing tests
- manually loop over test cases instead of parameterizing
- perform broad philosophical analysis before generating tests
- keep asking for permission to proceed once the required inputs are present
- stop after producing only a plan

## Checklist

Before finalizing, verify:

- [ ] Inventory was read first.
- [ ] Reachable EI set was identified.
- [ ] Every reachable EI is covered or blocked.
- [ ] Every case row maps to inventory facts.
- [ ] Every test has a `covers:` comment.
- [ ] All integration candidates are mocked or controlled.
- [ ] Patch targets use call-site lookup names.
- [ ] No external systems are contacted.
- [ ] No adjacent project units execute unmocked.
- [ ] Assertions are evidence-backed.
- [ ] Blocked EIs have `xfail` placeholders.
- [ ] Final output includes complete test file content.