# Unit Testing Contract

## Overview

This document defines the rules and workflow for writing unit tests. It is a
contract: **if a test violates this contract, it is not a unit test**.

This contract is designed to be used with a Unit Ledger as described in the
[Unit Ledger Generation Procedure](./unit-ledger-spec.md). The ledger is the
authoritative inventory of callables and execution items (EIs). This contract
is the authoritative set of constraints for deriving unit tests from that
ledger.

**Prerequisites:** You must read and understand the Unit Ledger specification
before using this contract. This document assumes you have a completed Unit
Ledger for the unit under test.

**Language Note:** Examples in this document are primarily in Python (using
pytest). However, the principles, rules, and workflow are **language-agnostic**
and apply to any programming language. Adapt the syntax and testing framework
to your language (JUnit for Java, xUnit for C#, RSpec for Ruby, etc.), but
follow the same procedure and constraints.

---

## 1. Definitions

### 1.1 Unit Test

A **unit test** exercises code within a single compilation unit (module, file,
class) in isolation from:
- Other units in the same project
- External systems (network, filesystem, database, etc.)
- Non-deterministic influences (time, randomness, environment)

A unit test:
- **Focuses on one unit** – the code defined in the target file/module only
- **Mocks all dependencies** – everything outside the unit boundary
- **Runs fast** – no I/O, no network, no subprocess calls
- **Is deterministic** – same inputs always produce the same results
- **Is isolated** – tests can run in any order, in parallel

### 1.2 Coverage Goal

**Goal:** Achieve 100% execution item (EI) coverage for the unit under test.

**EI coverage** means every execution item enumerated in the Unit Ledger is
exercised at least once. This includes:
- Every conditional branch/path (if/else, switch/case, ternary)
- Every loop outcome (zero iterations, 1+ iterations)
- Every exception path (success, each exception handler)
- Every early exit (return, raise, yield, break)
- Every sequential statement

**Scope:** Coverage applies only to the unit under test. Coverage gained by
exercising external code does not count and should be avoided through mocking.

### 1.3 Branch Coverage vs EI Coverage

**Industry term:** "Branch coverage" traditionally means exercising all
conditional branches.

**This contract:** Uses "EI coverage" to be precise about what we are
enumerating. An EI (Execution Item) is an atomic unit of execution – a distinct
outcome path for a single line of code.

For communication with traditional tools, "branch coverage" and "EI coverage"
are equivalent in this context. The Unit Ledger enumerates EIs, tests exercise
them, and coverage tools report them as branches.

### 1.4 Key Terms

**Unit Ledger:** Three-document YAML file containing:
- Document 1: Derived IDs (all EI IDs in the unit)
- Document 2: Ledger (detailed callable and EI specifications)
- Document 3: Review (findings from ledger generation)

**EI ID:** Unique identifier for an execution item (e.g., `C000F001E0003`).
Every reachable EI in the ledger must be covered by at least one test.

**Case Row:** The smallest unit of test intent. Specifies:
- Input values
- Expected outcome (return value or exception)
- `covers`: list of EI IDs exercised by this case

**Bucket:** A group of case rows that share the same harness key. Each bucket
becomes one test function.

**Harness Key:** A tuple of facts used to partition case rows into buckets:
- Callable ID
- Outcome kind (returns vs raises)
- Patch targets (sorted list of mocks needed)

**Integration Fact:** Information from the ledger about inter-unit calls and
boundary crossings. Used to determine what to mock and how to reach integration
points.

**Execution Path:** A sequence of EI IDs that must execute to reach a particular
point in the code. Used to trace how to trigger specific EIs or integrations.

**Blocked EI:** An EI that cannot be tested without information not present in
the unit or ledger. Blocked EIs do not prevent progress on other EIs.

**Unreachable EI:** An EI that cannot be executed without violating unit
boundaries or creating an impossible state. Marked UNREACHABLE in the ledger.

---

## 2. Hard Rules (Non-Negotiable)

### 2.1 Unit Boundaries

**What counts as "the unit":**
- Code defined in the target file/module only
- Classes, functions, methods in that file
- Nothing outside that file

**Forbidden:**
- Calling upstream orchestrators
- Calling adjacent units "because it's convenient"
- Letting external code execute unmocked

**Required:**
- Mock or stub everything outside the target unit
- Use integration facts from the ledger to identify what needs mocking

### 2.2 External Influences Must Be Mocked

Mock or stub all external influences, including but not limited to:

**Boundary Crossings (see ledger integration facts):**
- **Network:** HTTP, sockets, APIs (boundary.kind = network)
- **Filesystem:** File I/O, directory operations (boundary.kind = filesystem)
- **Database:** Queries, transactions (boundary.kind = database)
- **Subprocess:** Shell commands, external programs
  (boundary.kind = subprocess)
- **Message Bus:** Queue publish/consume (boundary.kind = message_bus)
- **Clock:** Current time, timestamps (boundary.kind = clock)
- **Randomness:** Random numbers, UUIDs (boundary.kind = randomness)
- **Environment:** Environment variables (boundary.kind = env)

**Interunit Calls (see ledger integration facts):**
- Calls to other project modules
- Calls to other classes in different files
- Dynamic discovery mechanisms (plugins, services, entrypoints)

**Third-Party Libraries:**
- External package registries
- HTTP clients (requests, httpx, etc.)
- Database drivers
- Async scheduling
- Thread pools

**If you are about to introduce an unmocked external influence, stop and
refactor the test to mock it.**

### 2.3 Test Structure Rules

**Function/Method Structure:**
- Do not nest test classes, by default
- Python: Use plain `def test_...():` functions only
- Java/Kotlin: Prefer single-level test classes over nested classes
- Projects may override this rule with explicit written instructions

**Parameterization:**
- Prefer the test framework's parameterization mechanism
- Do not copy/paste test blocks or functions for multiple cases
- Do not manually loop over test cases inside a single test function

**Language-Specific Adaptations:**
- Python: `@pytest.mark.parametrize` or `@parameterized.expand`
- Java: `@ParameterizedTest` with `@MethodSource` or `@ValueSource`
- C#: `[Theory]` with `[InlineData]` or `[MemberData]`
- Ruby: RSpec's `where` or `it` blocks with iteration
- Adapt to your language's idiomatic test patterns while following the spirit
  of these rules

**Example (Python with pytest):**
```python
# ✓ Good: Framework parameterization
@pytest.mark.parametrize("input,expected", [
    (5, "positive"),
    (-3, "negative"),
    (0, "zero"),
])
def test_classify_value(input, expected):
    # covers: C000F001E0001, C000F001E0002, C000F001E0003
    assert classify_value(input) == expected

# ✗ Bad: Manual loop
def test_classify_value():
    for input, expected in [(5, "positive"), (-3, "negative"), (0, "zero")]:
        assert classify_value(input) == expected
```

### 2.4 Selection Behavior Rule

If code selects the best item or uses ranked selection:

**Forbidden:**
- Relying on lexical sorting as a proxy for preference
- Asserting on incidental/arbitrary ordering

**Required:**
- Selection must follow the explicit preference order in the code under test
- Assert based on the code's ordering rules
- Use the actual comparison/ranking logic

### 2.5 Unreachable EIs

If you cannot reach an EI without violating unit boundaries or requiring
impossible state:
- The ledger should mark it UNREACHABLE
- Do not write fake tests just to hit the coverage number
- Document why it's unreachable

**Note:** Abstract methods and interface signatures are NOT unreachable. Test
them by introducing a test implementation that exercises the abstract contract.

---

## 3. Mandatory Workflow

### 3.1 Prerequisites

**Before writing any tests:**

1. **Generate the Unit Ledger** following the Unit Ledger Generation Procedure
2. **Validate the ledger** against the schema
3. **Review Document 3** for any generation findings

**If you start writing tests before the ledger exists, stop and generate the
ledger first.**

The ledger is the authoritative source for:
- All callables in the unit (Document 2: unit.children)
- All EI IDs and their outcomes (Document 2: callable.branches)
- Integration points requiring mocks (Document 2: callable.integration)
- Execution paths to each integration (integration.executionPaths)

### 3.2 The Test Generation Procedure

After the Unit Ledger exists, generate unit tests using this deterministic
7-stage procedure. This procedure is mechanical and must not involve searching
for an optimal layout.

The procedure produces a runnable test skeleton early. Later stages refine
assertions and reduce duplication, but must not change the ledger-derived
EI-to-case mapping.

---

#### Stage 1: Create Case Rows

**Purpose:** Map every reachable EI to at least one test case.

**Process:**

For each callable in the ledger (in ledger order):
1. For each reachable EI ID (in EI ID order):
   - Create one case row
   - Determine inputs that will trigger this EI
   - Determine expected outcome (return value or exception)
   - Record which EI IDs this case covers

**Case Row Structure:**
```yaml
{
    "callable_id": "C000F001",
    "ei_ids": ["C000F001E0003", "C000F001E0005"],  # EIs covered
    "inputs": {"x": -5},
    "outcome_kind": "returns",  # or "raises"
    "expected": "negative",
    "patch_targets": [],  # mocks needed (from integration facts)
}
```

**Rules:**
- Every reachable EI ID must appear in at least one case row
- Skip UNREACHABLE EIs (marked in ledger)
- If an EI cannot be mapped to inputs with certainty, mark it BLOCKED
  (see Section 4)
- Use integration facts from the ledger to determine patch_targets
- Use executionPaths to trace how to reach specific EIs

**Using Execution Paths:**

Integration facts in the ledger contain `executionPaths` – sequences of EI IDs
that lead to that integration point. Use these to:
- Trace what conditions must be true to reach an integration
- Determine which EIs must execute before the integration
- Set up mocks at the right point in the execution flow

Example:
```yaml
# From ledger
integration:
  boundaries:
    - id: IC000F002E0007
      target: requests.get
      executionPaths:
        # must hit E0001, E0003 first
        - [C000F002E0001, C000F002E0003, C000F002E0007]
```

This tells you: to reach the `requests.get` call (E0007), your test inputs must:
- Trigger E0001 (e.g., pass validation check)
- Trigger E0003 (e.g., take the "fetch remote" branch)
- Then you'll hit E0007 (the actual HTTP call to mock)

**Output:** Complete list of case rows for the entire unit.

**Verification Gate:**
- [ ] Every reachable EI ID appears in at least one case row
- [ ] Every case row has inputs, expected outcome, and covers the list
- [ ] Blocked EIs are marked (don't prevent progress)

---

#### Stage 2: Partition Into Buckets

**Purpose:** Group case rows that can share test setup/teardown.

**Process:**

For each case row, compute its harness key:
```python
harness_key = (
    callable_id,
    outcome_kind,  # "returns" or "raises"
    tuple(sorted(patch_targets))  # sorted list of mocks
)
```

Group case rows by identical harness keys. Each group is a bucket.

**Why This Key:**
- **callable_id:** Tests for different functions should be separate
- **outcome_kind:** Mixing returns and raises in one test is awkward
- **patch_targets:** Sharing mock setup is efficient; different mocks mean
  different buckets

**Rules:**
- Do not backtrack – once assigned, a case row stays in its bucket
- Keep the key small to avoid combinatorial explosion
- If the ledger provides different patch targets via integration facts, then
  respect them

**Example:**

Case rows:
```yaml
[
    {"callable": "C000F001", "outcome": "returns", "patches": [], "covers": ["E0001"]},
    {"callable": "C000F001", "outcome": "returns", "patches": [], "covers": ["E0002"]},
    {"callable": "C000F001", "outcome": "raises", "patches": [], "covers": ["E0003"]},
    {"callable": "C000F002", "outcome": "returns", "patches": ["requests.get"], "covers": ["E0005"]},
]
```

Buckets:
```
{
    ("C000F001", "returns", ()): [case1, case2],  # 2 cases, can parameterize
    ("C000F001", "raises", ()): [case3],          # 1 case, single test
    ("C000F002", "returns", ("requests.get",)): [case4],  # 1 case, needs mock
}
```

**Output:** Buckets of case rows, keyed by harness key.

**Verification Gate:**
- [ ] Every case row is in exactly one bucket
- [ ] Bucket keys are computed correctly
- [ ] No case rows lost or duplicated

---

#### Stage 3: Realize Test Functions

**Purpose:** Convert buckets into actual test functions.

**Process:**

For each bucket:
- **If a bucket has multiple case rows:** Create parameterized test function
- **If a bucket has one case row:** Create non-parameterized test function

**Naming Convention:**
- `test_<callable_name>_<outcome>_<distinguisher>`
- Example: `test_classify_value_returns`, `test_classify_value_raises`
- If multiple buckets for the same callable/outcome, add a distinguisher

**Coverage Annotation:**

Every test function must reference the EI IDs it covers:
```python
def test_classify_value_returns(input, expected):
    # covers: C000F001E0001, C000F001E0002, C000F001E0005
    assert classify_value(input) == expected
```

**Example Realization:**

Bucket: `("C000F001", "returns", ())` with 2 case rows

Becomes:
```python
@pytest.mark.parametrize("input,expected,covers", [
    (5, "positive", ["C000F001E0001", "C000F001E0003"]),
    (-3, "negative", ["C000F001E0002", "C000F001E0004"]),
])
def test_classify_value_returns(input, expected, covers):
    # covers: see parameter
    assert classify_value(input) == expected
```

**Output:** Test skeleton – all test functions defined, runnable (even if
assertions are minimal).

**Verification Gate:**
- [ ] Every bucket has exactly one test function
- [ ] Parameterized tests use a framework mechanism
- [ ] Every test has a coverage annotation
- [ ] Test skeleton is syntactically valid and runnable

---

#### Stage 4: Micro Review (Optional, Bounded)

**Purpose:** Improve clarity without introducing churn.

**When:** Only if no case rows are BLOCKED.

**Allowed Operations:**
- Merge buckets when harness keys are identical
- Split a bucket to isolate a blocked or uncertain case
- Rename test functions for readability and consistency
- Adjust parameterization for clarity

**Forbidden Operations:**
- Searching for an optimal grouping
- Revisiting EI-to-case mapping
- Inventing new EIs or case rows
- Deep semantic analysis
- Major refactoring

**Process:** Single pass over the bucket list. After one pass, stop.

**Output:** Refined test skeleton (optional, skip if all unblocked).

**Verification Gate:**
- [ ] No EI IDs were added or removed
- [ ] Every EI ID is still covered by the same test(s)
- [ ] Changes improve clarity only

---

#### Stage 5: Implementation (Coverage First)

**Purpose:** Implement the test functions to achieve full EI coverage.

**Process:**

Implement test functions bucket by bucket, in ledger order:

1. **Set up mocks** (from patch_targets in case rows)
   - Use integration facts to identify mock targets
   - Mock at the call site used by the unit under test
   - Use project-provided fakes/fixtures when available

2. **Call the function under test** with the case row inputs

3. **Assert the outcome**
   - For return cases: assert return value matches expected
   - For exception cases: assert exception type and key message substrings

4. **Verify coverage**
   - Run tests with a coverage tool capable of tracking lines *and* branches
   - Confirm all EI IDs in case row's `covers` list are hit

**Constraints:**
- Follow mocking rules (Section 2.2 and 5.1)
- Do not skip to later callables – work in ledger order
- Prefer minimal assertions sufficient to prove the EI outcome
- Do not invent expectations – if uncertain, use minimal proof

**Example Implementation:**

```python
from unittest.mock import patch, Mock

def test_fetch_user_data_returns(mock_requests_get):
    # covers: C000F004E0002, C000F004E0004, C000F004E0006
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "123", "name": "Alice"}
    mock_requests_get.return_value = mock_response
    
    result = fetch_user_data("123")
    
    assert result == {"id": "123", "name": "Alice"}
    mock_requests_get.assert_called_once_with("https://api.example.com/users/123")

def test_fetch_user_data_raises_on_empty_id():
    # covers: C000F004E0001, C000F004E0003
    with pytest.raises(ValueError, match="user_id required"):
        fetch_user_data("")

def test_fetch_user_data_raises_on_api_error(mock_requests_get):
    # covers: C000F004E0002, C000F004E0004, C000F004E0005
    mock_response = Mock()
    mock_response.status_code = 404
    mock_requests_get.return_value = mock_response
    
    with pytest.raises(RuntimeError, match="API error: 404"):
        fetch_user_data("999")
```

**Deliverable:** Runnable test unit where all test functions execute and cover
all reachable EI IDs.

**Note:** If using generative AI and the AI cannot run tests itself, manual
intervention is required:
1. Run the tests yourself
2. Provide error output to AI
3. AI makes corrections
4. Repeat until tests pass

**Verification Gate:**
- [ ] All test functions are implemented
- [ ] Tests are runnable (syntax valid, imports correct)
- [ ] All reachable EI IDs covered
- [ ] All mocks set up correctly
- [ ] Unit isolation maintained (no external calls leak through)

---

#### Stage 6: Refinement (Bounded)

**Purpose:** Strengthen assertions and reduce duplication without changing the
structure.

**When:** After Stage 5 is complete and all tests pass.

**Process:** Single-pass refinement, bounded to one sweep.

**Allowed Improvements:**
- Stronger assertions based on an explicit unit contract
- Reduced duplication through parameterization or helpers
- Clearer naming and readability
- Improved mock clarity and call site correctness
- Better error message assertions (more specific substrings)

**Forbidden Changes:**
- Changing EI-to-case mapping from Stage 1
- Changing which EI IDs are covered by which tests
- Structural rewrites
- New test functions or removed test functions
- Coupling to implementation details

**Stop Condition:** When improvements lack determinism or require guessing.

**Example Refinement:**

Before:
```python
def test_classify_value_positive():
    assert classify_value(5) == "positive"

def test_classify_value_negative():
    assert classify_value(-3) == "negative"
```

After (reduced duplication via parameterization):
```python
@pytest.mark.parametrize("input,expected", [
    (5, "positive"),
    (-3, "negative"),
])
def test_classify_value_returns(input, expected):
    assert classify_value(input) == expected
```

**Verification Gate:**
- [ ] No EI coverage lost
- [ ] Tests still pass
- [ ] Assertions stronger (if deterministic improvement exists)
- [ ] Duplication reduced (if possible without churn)

---

#### Stage 7: Stop Condition

**Stop when:**
- All reachable EI IDs are covered, AND
- Unit isolation constraints are satisfied (all external influences mocked), AND
- Assertions provide sufficient proof without brittleness or invention

**Verification:** Run the coverage tool and confirm that the tests provide 100%
EI coverage for the unit.

**If coverage < 100%:**
- Check the ledger for UNREACHABLE EIs (and exclude from the coverage target)
- Check for BLOCKED EIs (see Section 4)
- Identify which EI IDs are missing coverage
- Add case rows for missing EIs
- Return to Stage 2 to integrate new case rows

**Output:** Complete unit test file, ready for commit.

---

## 4. Blocked EI Protocol

### 4.1 Purpose

A blocked EI must not prevent progress. If any EI IDs cannot be tested without
guessing, continue and complete all unblocked EIs first, then report blocked
EIs explicitly.

### 4.2 Definition of Blocked

An EI ID is **BLOCKED** if reaching it, or asserting its outcome, requires
missing information that cannot be derived from the unit and ledger without
guessing.

**Common causes:**
- Unknown callable signature required to call the code
- Branch trigger cannot be mapped to inputs or mocks from ledger facts
- Expected exception type or message substring is unavailable
- Required fakes, fixtures, or utilities are referenced but not provided
- The required patch target is ambiguous from unit and ledger

**Blocked status is determined per EI ID.**

### 4.3 Required Output

When any EI IDs are blocked, the test file must include:

**1. BLOCKED EI IDs Comment Block:**

Format (adapt comment syntax to your language):
```python
# BLOCKED EI IDs
# - <EI_ID>:
#     why: <reason the EI is blocked>
#     need: <concrete artifact needed to unblock>
#     impact: <scope of impact>
#     info: <optional: brief additional context>
#     action: <what user can provide to unblock>
```

Fields:
- `why`: Reason the EI is blocked
- `need`: Minimal missing input (must name concrete artifact: code snippet,
  signature, exception type, fixture name, patch target)
- `impact`: "Localized to this EI" or list other impacted EI IDs
- `info` (optional): Brief additional context (no invented expectations)
- `action`: Single-sentence unblock action (clear, unambiguous)

**2. Placeholder Test:**

Create a test function for each blocked EI, marked as expected failure (adapt
this mechanism to your test framework):

**Python (pytest):**
```python
import pytest

@pytest.mark.xfail(reason="Blocked: missing trigger mapping for C000F002E0004")
def test_blocked_C000F002E0004():
    # covers: C000F002E0004
    assert False, "EI blocked - see BLOCKED EI IDs comment"
```

**Java (JUnit 5):**
```java
@Test
@Disabled("Blocked: missing trigger mapping for C000F002E0004")
void test_blocked_C000F002E0004() {
    // covers: C000F002E0004
    fail("EI blocked - see BLOCKED EI IDs comment");
}
```

**C# (xUnit):**
```csharp
[Fact(Skip = "Blocked: missing trigger mapping for C000F002E0004")]
public void Test_Blocked_C000F002E0004()
{
    // covers: C000F002E0004
    Assert.True(false, "EI blocked - see BLOCKED EI IDs comment");
}
```

Purpose: Keep blocked work visible and allow user to supply missing inputs and
regenerate tests later.

### 4.4 Example

```python
# BLOCKED EI IDs
# - C000F002E0004:
#     why: trigger mapping not derivable from ledger facts
#     need: code snippet for parse_config showing the branch condition at line 47
#     impact: all other reachable EIs covered; this one pending
#     info: ledger does not identify which input parameter controls this branch
#     action: paste parse_config function body or add ledger note mapping
#             the condition to specific inputs or mock configurations

import pytest

@pytest.mark.xfail(reason="Blocked: missing trigger mapping for C000F002E0004")
def test_blocked_C000F002E0004():
    # covers: C000F002E0004
    assert False, "EI blocked - see BLOCKED EI IDs comment"
```

### 4.5 No-Stall Requirement

If blocked EIs exist:
- Do NOT stop test generation
- Do NOT expand analysis or search for solutions
- Emit runnable tests for all unblocked EIs first
- Then emit the blocked EI report and placeholders
- Move forward

---

## 5. Test Writing Rules

### 5.1 Mocking Strategy

**Using Integration Facts from the Ledger:**

The Unit Ledger contains integration facts (Document 2: callable.integration)
that tell you exactly what needs mocking:

```yaml
# From ledger
integration:
  interunit:
    - id: IC000F001E0004
      target: validate_typed_dict
      kind: call
      executionPaths: [[C000F001E0004]]
  boundaries:
    - id: IC000F001E0007
      target: requests.get
      kind: call
      boundary:
        kind: network
        protocol: http
      executionPaths: [[C000F001E0003, C000F001E0007]]
```

This tells you:
1. **What to mock:** `validate_typed_dict` (interunit),
  `requests.get` (boundary)
2. **Where to mock:** At the call site in the unit under test
3. **How to reach it:** Follow executionPaths to determine what conditions
  trigger these calls

**Mocking Rules:**

1. **Mock at the call site** used by the unit under test
2. **Use the project's standard mocking mechanism** (pytest fixtures,
   unittest.mock, etc.)
3. **Use project-provided fakes and fixtures** when available – do not recreate
   them
4. **Mock all boundaries** identified in integration facts (network, filesystem,
   database, etc.)
5. **Mock all interunit calls** - don't let adjacent units execute

**Example:**

```python
from unittest.mock import patch, Mock

@patch('myunit.requests.get')  # Mock at the call site in myunit
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

**For Return Cases:**
- Assert return value matches expected
- Use equality checks (`==`) for simple values
- Use structural assertions (`isinstance`, `len`) for complex objects
- Assert key fields, not entire object graphs (avoid brittleness)

**For Exception Cases:**
- Assert the exception type
- Assert key substrings in the message (not the full message – too brittle)
- Prefer user-meaningful wording: project names, version numbers, strategy
  names, URL patterns, policy modes

**Examples:**

```python
# Good: Key substring
with pytest.raises(
        ValueError,
        match="Invalid environment key"):
    validate_config(bad_config)

# Bad: Full message (brittle)
with pytest.raises(
        ValueError,
        match="Invalid environment key: DEBUG_MODE not in allowed keys: ['LOG_LEVEL', 'API_URL']"):
    validate_config(bad_config)

# Good: Behavior assertion
assert len(result) == 3
assert all(isinstance(item, Config) for item in result)

# Bad: Implementation detail
assert result._internal_cache is not None
```

**Assertion Principles:**
- Assert behavior, not implementation details (unless EI coverage requires it)
- Prefer minimal assertions sufficient to prove the EI outcome
- Strengthen assertions in Stage 6 if deterministic improvements exist
- Stop strengthening when improvements require guessing

### 5.3 EI Coverage Labeling

**Every test must state which EI IDs it covers.**

**Format:** Comment with `covers:` prefix

```python
def test_classify_value_positive():
    # covers: C000F001E0001, C000F001E0003
    assert classify_value(5) == "positive"
```

**For parameterized tests:**

Option 1: Include in parameter data
```python
@pytest.mark.parametrize("input,expected,covers", [
    (5, "positive", ["C000F001E0001", "C000F001E0003"]),
    (-3, "negative", ["C000F001E0002", "C000F001E0004"]),
])
def test_classify_value(input, expected, covers):
    # covers: see parameter
    assert classify_value(input) == expected
```

Option 2: List all in function comment
```python
@pytest.mark.parametrize("input,expected", [
    (5, "positive"),
    (-3, "negative"),
])
def test_classify_value(input, expected):
    # covers: C000F001E0001, C000F001E0002, C000F001E0003, C000F001E0004
    assert classify_value(input) == expected
```

**Why this matters:**
- Enables verification that all EI IDs are covered
- Makes coverage gaps visible during code review
- Allows regeneration of specific tests when code changes
- Documents the ledger-to-test mapping

---

## 6. Complete Worked Example

This section shows the full workflow from ledger to tests.

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

### 6.2 The Unit Ledger (Excerpts)

**Document 2 (Ledger) - Relevant Section:**

```yaml
- id: C000F001
  kind: callable
  name: classify_value
  signature: 'classify_value(x: int) -> str'
  callable:
    params:
      - name: x
        type:
          name: int
    returnType:
      name: str
    branches:
      - id: C000F001E0001
        condition: 'if x < 0'
        outcome: 'returns "negative"'
      - id: C000F001E0002
        condition: 'else'
        outcome: 'continues to next check'
      - id: C000F001E0003
        condition: 'if x == 0'
        outcome: 'returns "zero"'
      - id: C000F001E0004
        condition: 'else'
        outcome: 'continues to final return'
      - id: C000F001E0005
        condition: 'final return'
        outcome: 'returns "positive"'
```

### 6.3 Stage 1: Case Rows

```python
case_rows = [
    {
        "callable_id": "C000F001",
        "ei_ids": ["C000F001E0001"],
        "inputs": {"x": -5},
        "outcome_kind": "returns",
        "expected": "negative",
        "patch_targets": [],
    },
    {
        "callable_id": "C000F001",
        "ei_ids": ["C000F001E0002", "C000F001E0003"],
        "inputs": {"x": 0},
        "outcome_kind": "returns",
        "expected": "zero",
        "patch_targets": [],
    },
    {
        "callable_id": "C000F001",
        "ei_ids": ["C000F001E0002", "C000F001E0004", "C000F001E0005"],
        "inputs": {"x": 10},
        "outcome_kind": "returns",
        "expected": "positive",
        "patch_targets": [],
    },
]
```

All 5 EI IDs covered: ✓

### 6.4 Stage 2: Buckets

All case rows have same harness key: `("C000F001", "returns", ())`

```python
buckets = {
    ("C000F001", "returns", ()): [case_row_1, case_row_2, case_row_3]
}
```

### 6.5 Stage 3: Test Function

```python
@pytest.mark.parametrize("x,expected,covers", [
    (-5, "negative", ["C000F001E0001"]),
    (0, "zero", ["C000F001E0002", "C000F001E0003"]),
    (10, "positive", ["C000F001E0002", "C000F001E0004", "C000F001E0005"]),
])
def test_classify_value(x, expected, covers):
    # covers: see parameter
    assert classify_value(x) == expected
```

### 6.6 Stage 5: Implementation

(Already implemented in Stage 3 for this simple example)

### 6.7 Stage 6: Refinement

Could add more test cases for better validation:

```python
@pytest.mark.parametrize("x,expected,covers", [
    (-5, "negative", ["C000F001E0001"]),
    (-1, "negative", ["C000F001E0001"]),  # Additional case
    (0, "zero", ["C000F001E0002", "C000F001E0003"]),
    (1, "positive", ["C000F001E0002", "C000F001E0004", "C000F001E0005"]),  # Additional
    (10, "positive", ["C000F001E0002", "C000F001E0004", "C000F001E0005"]),
])
def test_classify_value(x, expected, covers):
    # covers: see parameter
    assert classify_value(x) == expected
```

### 6.8 Final Test File

```python
# test_validation.py
import pytest
from validation import classify_value


@pytest.mark.parametrize("x,expected,covers", [
    (-5, "negative", ["C000F001E0001"]),
    (-1, "negative", ["C000F001E0001"]),
    (0, "zero", ["C000F001E0002", "C000F001E0003"]),
    (1, "positive", ["C000F001E0002", "C000F001E0004", "C000F001E0005"]),
    (10, "positive", ["C000F001E0002", "C000F001E0004", "C000F001E0005"]),
])
def test_classify_value(x, expected, covers):
    """Test classify_value with various inputs.
    
    Covers all 5 EI IDs for classify_value:
    - C000F001E0001: x < 0 → negative
    - C000F001E0002: x >= 0 → continue
    - C000F001E0003: x == 0 → zero
    - C000F001E0004: x > 0 → continue
    - C000F001E0005: return positive
    """
    # covers: see parameter
    assert classify_value(x) == expected
```

Coverage: 100% ✓

---

## 7. Common Testing Patterns

This section identifies common branch categories to assist in creating case
rows. If supplemental project- or domain-specific instructions exist, they take
precedence.

### 7.1 Input Classification Units

Common EI buckets:
- **Multiple accepted shapes or encodings** (dict vs. JSON string, absolute vs.
  a relative path)
- **Input validation** (valid, invalid characters, empty, null)
- **Normalization** (case-insensitive, whitespace trimming, path normalization)
- **Type coercion** (string to int, flexible vs. strict parsing)

**Example EIs:**
- E0001: input is dict → parse as dict
- E0002: input is string → parse as JSON
- E0003: input is None → raise ValueError
- E0004: input is empty string → raise ValueError

### 7.2 Conditional Logic Units

Common EI buckets:
- **Mode selection** (strict vs permissive, debug vs normal, dry-run vs apply)
- **Feature gates** (feature enabled, feature disabled)
- **Filtering** (keep vs. drop, empty results vs. some results vs. all results)
- **Early exits** (precondition fails, precondition passes)

**Example EIs:**
- E0001: strict mode → validate all fields
- E0002: permissive mode → validate required fields only
- E0003: validation passes → continue
- E0004: validation fails → raise exception

### 7.3 Collection Processing Units

Common EI buckets:
- **Empty collection** (no items to process)
- **Single item** (special case handling)
- **Multiple items** (batch processing)
- **All filtered out** (filter returns empty)
- **Some pass filter** (partial results)

**Example EIs:**
- E0001: collection empty → return empty list
- E0002: a collection has items, all filtered → return an empty list
- E0003: a collection has items, some pass → return filtered list

### 7.4 Error Handling Units

Common EI buckets:
- **Success path** (no errors)
- **Expected errors** (handled exceptions with specific messages)
- **Unexpected errors** (catch-all handlers)
- **Retry logic** (the first attempt fails, retry succeeds)

**Example EIs:**
- E0001: operation succeeds → return result
- E0002: operation raises ValueError → catch and handle
- E0003: operation raises TypeError → catch and handle
- E0004: operation raises other exception → propagate

### 7.5 Integration/Boundary Units

Common EI buckets:
- **External call succeeds** (boundary returns expected data)
- **External call fails** (boundary raises exception or returns error)
- **Fallback paths** (primary fails, fallback succeeds)
- **Caching** (cache hit, cache miss, cache invalidation)

**Example EIs:**
- E0001: API call returns 200 → parse and return data
- E0002: API call returns 404 → raise NotFoundError
- E0003: API call times out → raise TimeoutError
- E0004: use cached data → skip API call

### 7.6 Strategy/Selection Units

Common EI buckets:
- **Strategy selection** (which strategy applies based on conditions)
- **Preference ordering** (stable deterministic tie-breaking)
- **Best match** (multiple candidates, select best)
- **Ambiguity** (no match, exactly one match, multiple matches)

**Example EIs:**
- E0001: no candidates → raise NoMatchError
- E0002: exactly one match → return it
- E0003: multiple matches, select by priority → return best
- E0004: multiple matches, ambiguous → raise AmbiguousMatchError

### 7.7 State Transition Units

Common EI buckets:
- **Valid transitions** (state A → state B allowed)
- **Invalid transitions** (state A → state C forbidden)
- **Idempotency** (repeated transition has no effect)
- **Terminal states** (no transitions allowed)

**Example EIs:**
- E0001: current state is PENDING, transition to RUNNING → allowed
- E0002: current state is PENDING, transition to COMPLETE → forbidden
- E0003: current state is COMPLETE, any transition → forbidden
- E0004: transition called twice → idempotent, no error

### 7.8 Language Construct Verification Units

Some language constructs establish contracts that should be verified even though
they don't create execution items (EIs) in the ledger.

#### 7.8.1 Enum Value Contracts

**What to test:**
- Enum members exist
- Enum values match expected strings/integers
- No accidental deletions or modifications

**Why:** Protects against refactoring errors and typos that break the enum contract.

**Example (Python):**
```python
def test_requires_dist_url_policy_enum_values():
    """Verify RequiresDistUrlPolicy enum contract."""
    assert RequiresDistUrlPolicy.HONOR.value == "honor"
    assert RequiresDistUrlPolicy.IGNORE.value == "ignore"
    assert RequiresDistUrlPolicy.RAISE.value == "raise"
    # Verify all expected members exist
    assert set(RequiresDistUrlPolicy) == {
        RequiresDistUrlPolicy.HONOR,
        RequiresDistUrlPolicy.IGNORE,
        RequiresDistUrlPolicy.RAISE
    }
```

**Ledger relationship:** Enum definitions create no EIs. This test verifies the
contract, not execution paths. As such, these tests will not be a direct result
of enumerated EI IDs. They will be a result of parsing the unit at test
creation time.

#### 7.8.2 Data Class Contracts

**What to test:**
- Field existence and default values
- Immutability constraints
  - Python: `frozen=True`
  - Java: `@Value` when using Lombok
- Basic construction with and without arguments

**Why:** Data class constructs auto-generate code outside the unit. Tests should
verify the contract established by the decorator. Constructs vary by language.
For example:
- Python: `dataclass`
- Java: `@Value` when using Lombok, or record classes

**Example (Python):**
```python
def test_resolution_result_dataclass_contract():
    """Verify ResolutionResult dataclass contract."""
    # Test default construction
    result = ResolutionResult()
    assert result.requirements_by_env == {}
    assert result.resolved_wheels_by_env == {}
    
    # Test construction with values
    result = ResolutionResult(
        requirements_by_env={"env1": "reqs"},
        resolved_wheels_by_env={"env1": ["wheel1"]}
    )
    assert result.requirements_by_env == {"env1": "reqs"}
    assert result.resolved_wheels_by_env == {"env1": ["wheel1"]}
    
    # Test immutability (if frozen=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.requirements_by_env = {}
```

**Ledger relationship:** Dataclass-generated methods create no EIs in the source
file. This test verifies the dataclass contract, not execution paths in the
unit. As such, these tests will not be a direct result of enumerated EI IDs.
They will be a result of parsing the unit at test creation time.

#### 7.8.3 When to Use Construct Verification

**Use construct verification tests when:**
- Language features auto-generate code (dataclasses, enums, properties)
- The generated code establishes a contract
- Changes to the contract would break callers
- The auto-generated code is not in the unit's source file

**Do NOT use construct verification for:**
- Methods explicitly written in the source file (these have EIs)
- Internal implementation details that are not exposed as contracts
- Constructs where the language guarantees the behavior

## Appendix A: Quick Reference

### A.1 Workflow Checklist

- [ ] Generate Unit Ledger (read spec first)
- [ ] Validate ledger against schema
- [ ] Review Document 3 for findings
- [ ] Stage 1: Create case rows (one per reachable EI)
- [ ] Stage 2: Partition into buckets (by harness key)
- [ ] Stage 3: Realize test functions (parameterize or single)
- [ ] Stage 4: Micro review (optional, bounded)
- [ ] Stage 5: Implement tests (coverage first)
- [ ] Stage 6: Refine (strengthen assertions, reduce duplication)
- [ ] Stage 7: Verify 100% EI coverage
- [ ] Run tests and confirm all pass
- [ ] Commit test file

### A.2 Harness Key Formula

```python
harness_key = (
    callable_id,           # Which function/method
    outcome_kind,          # "returns" or "raises"
    tuple(sorted(patch_targets))  # Sorted list of mocks
)
```

### A.3 Case Row Template

```
{
    "callable_id": "<callable_id>",
    "ei_ids": ["<ei_id_1>", "<ei_id_2>", ...],
    "inputs": {<param_name>: <value>, ...},
    "outcome_kind": "returns" | "raises",
    "expected": <return_value> | <exception_type>,
    "patch_targets": ["<mock_target_1>", ...],
}
```

### A.4 Coverage Comment Format

```python
# covers: C000F001E0001, C000F001E0002, C000F001E0005
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
@patch('myunit.requests.get')
def test_fetch(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"data": "value"}

# Mock filesystem
@patch('myunit.open', mock_open(read_data="file contents"))
def test_read_file(mock_file):
    ...

# Mock time
@patch('myunit.time.time', return_value=1234567890)
def test_timestamp(mock_time):
    ...

# Mock random
@patch('myunit.random.random', return_value=0.5)
def test_random(mock_random):
    ...
```

---

## Appendix B: Terminology Map

**Old Term → New Term**

- Branch → Execution Item (EI)
- Branch ID → EI ID
- Branch coverage → EI coverage (or branch coverage for external tools)
- Callable branches → Callable EIs
- Integration target → Integration fact

**Backward Compatibility:**

When communicating with coverage tools or team members unfamiliar with the Unit
Ledger spec, "branch coverage" is acceptable and understood. Internally, use
"EI" for precision.

---

## Appendix C: Validation Checklist

Before committing tests, verify:

**Ledger Alignment:**
- [ ] Test file exists for the unit
- [ ] Every reachable EI ID from the ledger has coverage
- [ ] UNREACHABLE EIs excluded from the coverage target
- [ ] BLOCKED EIs have placeholder tests

**Unit Isolation:**
- [ ] No unmocked external calls (network, filesystem, database, etc.)
- [ ] No unmocked interunit calls
- [ ] All boundaries are mocked (check ledger integration facts)
- [ ] Tests run fast (no I/O delays)

**Test Quality:**
- [ ] All tests pass
- [ ] Coverage tool reports 100% (excluding UNREACHABLE)
- [ ] Every test has a coverage comment
- [ ] Parameterization used where appropriate
- [ ] Assertions are specific and non-brittle
- [ ] Mock setup is clear and correct

**Code Quality:**
- [ ] Test names are descriptive
- [ ] No duplication (extracted to helpers/fixtures)
- [ ] Code is readable and maintainable
- [ ] Follows project conventions

### C.1 Coverage Calculation Notes

**What counts toward EI coverage:**
- Only EIs enumerated in the Unit Ledger
- EIs must be in source code files within the unit boundary
- UNREACHABLE EIs are excluded from the coverage target

**What does NOT count toward EI coverage:**
- Auto-generated code from decorators (dataclasses, enums, properties)
- Code in external libraries or frameworks
- Language runtime behavior
- Metaprogramming-generated methods

**Example scenarios:**

| Scenario                            | Has EIs? | Should Test?  | Test Type             |
|-------------------------------------|----------|---------------|-----------------------|
| Explicit method in source file      | Yes      | Yes           | EI coverage           |
| Dataclass `__init__`                | No       | Yes           | Contract verification |
| Enum class definition               | No       | Yes           | Contract verification |
| Enum constructor call `MyEnum(val)` | Yes      | Yes           | EI coverage           |
| `@property` decorated method        | Yes      | Yes           | EI coverage           |
| Standard library call               | No       | Mock in tests | Not directly tested   |

**Coverage tool interpretation:**
Coverage tools may report branches in auto-generated code. These do NOT need to
be covered to meet the 100% EI coverage goal, since they are not enumerated in
the Unit Ledger. Focus on the ledger as the authoritative coverage inventory. If
you specifically want to test auto-generated code, then you will need to add
tests manually, since that is not a function of the ledger, or of this unit
testing augmentation process.

---

**END OF CONTRACT**
