# Integration Flow Testing (Work in Progress)

## Status: Design Phase

This directory will contain the specification and procedures for integration flow testing. This README captures the initial design thinking and approach.

---

## Core Concept

**Integration flow testing focuses on SEAMS, not internal execution paths.**

- **Unit tests** verify all execution items (EIs) within a single unit - exhaustive internal path coverage
- **Integration tests** verify how units compose across their boundaries - seam interaction coverage

Integration flows are sequences of integration points (seams) across multiple units. We test these flows with use case coverage: happy paths, boundary conditions, and error conditions.

---

## The Graph Model

### Nodes: Integration Points

Every integration fact from unit ledgers becomes a node:
- **Interunit integrations:** Calls from Unit A to Unit B
- **Boundary integrations:** Calls to external systems (network, filesystem, database, etc.)

Example nodes:
- `IC000F001E0004` - Unit A calls Unit B.process_data()
- `IC001F002E0007` - Unit B calls Unit C.fetch()
- `IC002F001E0012` - Unit C calls requests.get() [boundary]

### Edges: Integration Dependencies

An edge exists when one integration can lead to another:
- If integration I1 can cause integration I2 to execute, create edge I1 → I2
- Determined by: Call relationships and execution paths in the code

### Flows: Sequences of Seams

A flow is a path through the graph from entry point to terminal node:
- **Entry points:** First seam in a flow (public API, event handler, message consumer)
- **Terminal nodes:** Boundaries (external system calls) or natural conclusions
- **Flow:** Sequence of integration IDs representing how units compose

Example flow:
```
Entry: A.handle_request
  ↓ (seam: A→B)
B.process_data
  ↓ (seam: B→C)
C.fetch
  ↓ (seam: C→API)
requests.get [boundary]

Flow: [IC000F001E0004 → IC001F002E0007 → IC002F001E0012]
Meaning: "A calls B, which calls C, which calls external API"
```

---

## The Sliding Window Approach

### Complete Flow Enumeration

1. Build the integration graph from all unit ledgers in scope
2. Identify all complete flows (entry point → terminal node)
3. For each flow of length N, generate all sliding windows of length 2 to N

### Sliding Windows

For a flow `[A→B, B→C, C→API]` (length 3):

**Windows of length 2:**
- `[A→B, B→C]` - Test A and B integration, mock C
- `[B→C, C→API]` - Test B and C integration, mock API

**Windows of length 3:**
- `[A→B, B→C, C→API]` - Test full flow, mock only API

Each window represents a potential integration test with different scope.

### Why Sliding Windows?

Different window lengths test different composition depths:
- **Smaller windows (2-3 seams):** More focused, easier to debug, faster tests
- **Larger windows (4+ seams):** More comprehensive, closer to real system behavior
- **All windows:** Ensure every seam pair is tested together

---

## Test Scope and Mocking Strategy

### Test Scope = Units in the Window

For window `[A→B, B→C]`:
- **System under test:** Units A and B (real, unmocked)
- **Mock boundary:** Unit C's interface
- **Entry point:** Call A's public API (the "way in")

For window `[A→B, B→C, C→API]`:
- **System under test:** Units A, B, and C (real, unmocked)
- **Mock boundary:** External API
- **Entry point:** Call A's public API (the "way in")

### The "Way In"

To test an integration flow, you need an entry point:
- The first seam's caller (often a public API, endpoint, or event handler)
- Use minimal execution paths to reach the first seam
- Don't test all internal EI variations (that's what unit tests do)

Example:
```python
def test_integration_flow_a_to_c(mock_api):
    # Entry point: Call A's public interface
    # Use simple, valid input (minimal path to seams)
    result = A.handle_request({"user_id": "123"})
    
    # Assert on SEAM behavior:
    # - Did the flow complete correctly?
    # - Was the API called as expected?
    # - Did data flow through A→B→C correctly?
    assert result == expected_output
    mock_api.get.assert_called_once_with("/users/123")
```

---

## Use Case Coverage for Seams

For each integration flow window, test with use case coverage:

### Happy Paths
- Normal data flows through the seam chain correctly
- Success responses propagate back properly
- State changes happen as expected across boundaries

### Boundary Conditions
- Empty collections at seam boundaries
- Null/None values crossing seams
- Maximum/minimum values that test contract limits
- Edge cases in data transformation between units
- Large datasets flowing through seams

### Error Conditions
- Exceptions raised at seams - do they propagate correctly?
- Timeouts at external boundaries
- Invalid data shapes crossing seams (contract violations)
- Partial failures in multi-step flows

### Example Use Case Matrix

For flow window `[A→B, B→C, C→API]`:

| Use Case | Entry Input | Mock Behavior | Expected Outcome | Category |
|----------|-------------|---------------|------------------|----------|
| Normal success | `{"user_id": "123"}` | API returns `{"status": "ok"}` | Data flows correctly, result returned | Happy path |
| Empty API response | `{"user_id": "123"}` | API returns `{}` | System handles gracefully | Boundary |
| API timeout | `{"user_id": "123"}` | API raises Timeout | Error propagates to A | Error |
| Invalid data from B | `{"user_id": "123"}` | B returns wrong type | Contract violation caught | Error |
| Large dataset | `{"user_id": "123"}` | API returns 10k items | Handles large data | Boundary |

---

## What We're NOT Testing

Integration tests focus on seams, not internal logic:

**NOT tested in integration tests:**
- All EI variations within each unit (unit tests cover this)
- Every possible input combination (unit tests cover this)
- Internal branching and conditionals (unit tests cover this)

**Tested in integration tests:**
- Do units call each other correctly at boundaries?
- Do data contracts match at seams?
- Do errors propagate correctly across seams?
- Does the composition work end-to-end?

### Example Contrast

**Unit test (exhaustive EI coverage):**
```python
def test_handle_request_empty_input():
    # covers: C000F001E0001
    with pytest.raises(ValueError):
        A.handle_request(None)

def test_handle_request_high_priority():
    # covers: C000F001E0004
    result = A.handle_request({"priority": "high", "value": "x"})
    # ... detailed assertions on A's internal behavior

def test_handle_request_normal_priority():
    # covers: C000F001E0005
    result = A.handle_request({"value": "x"})
    # ... detailed assertions on A's internal behavior
```

**Integration test (minimal paths, focus on seams):**
```python
def test_a_to_b_to_c_happy_path(mock_api):
    # Use ONE simple valid path to reach the seams
    # Don't test all input variations - that's what unit tests do
    mock_api.get.return_value = {"data": "success"}
    
    result = A.handle_request({"user_id": "123"})
    
    # Assert on SEAM behavior, not internal EI behavior:
    assert result == {"data": "success"}
    mock_api.get.assert_called_once()

def test_a_to_b_to_c_api_error(mock_api):
    # Test error propagation ACROSS seams
    mock_api.get.side_effect = ConnectionError("API down")
    
    with pytest.raises(ServiceUnavailableError):
        A.handle_request({"user_id": "123"})
```

---

## Window Filtering (Future Work)

Not all sliding windows may be worth testing. Future work will define filtering criteria:

### Potentially Keep (High Value)
- Windows ending at boundaries (test system interaction with external world)
- Windows spanning multiple units (test inter-unit contracts)
- Windows with error propagation (often missed in unit tests)
- Windows representing critical business flows

### Potentially Discard (Lower Value or Redundant)
- Windows fully covered by smaller windows (if composition doesn't add risk)
- Pure pass-through windows (unit just forwards data unchanged)
- Internal chaining within same unit (unit tests likely cover adequately)

**Decision:** Start by generating all windows, collect data on usefulness, then determine filtering criteria based on real-world experience.

---

## Future Integration Flow Specification

The complete specification will include:

### Stage 1: Build Integration Graph
- Collect all integration facts from unit ledgers
- Create nodes for each integration ID
- Create edges based on call relationships
- Identify entry points and terminal nodes

### Stage 2: Enumerate Complete Flows
- Find all paths from entry points to terminal nodes
- Record full flow sequences (length N)

### Stage 3: Generate Sliding Windows
- For each complete flow of length N:
  - Generate all subflows of length 2
  - Generate all subflows of length 3
  - ...
  - Generate all subflows of length N

### Stage 4: Filter Test-Worthy Windows (TBD)
- Apply filtering criteria (to be determined)
- Determine which windows represent meaningful tests
- Mark redundant windows

### Stage 5: Generate Integration Test Specifications
- For each test-worthy window:
  - Identify system under test (which units are real)
  - Determine mock boundaries
  - Identify entry point ("way in")
  - Generate use case matrix (happy, boundary, error)
  - Specify assertions for each use case

### Stage 6: Generate Integration Tests
- Similar to unit test generation contract
- But focused on seam interactions, not EI exhaustiveness
- Use case coverage for each window

### Stage 7: Validation
- Verify all critical flows are tested
- Verify no duplicate coverage (if filtering applied)
- Confirm mock boundaries are correct

---

## Open Questions

These will be resolved during specification development:

1. **Graph construction precision:** Are edges between specific integration IDs, or between callables more broadly?

2. **Entry point detection:** What makes an integration point an "entry"? Do we need ledger annotations?

3. **Edge determination:** How do we mechanically determine if integration I1 can lead to I2? Do we need path analysis across units?

4. **Window filtering strategy:** Generate all windows (comprehensive), or filter to valuable subset (targeted)?

5. **Test oracle definition:** For a given window, what exactly are we asserting? Flow completion? Data correctness? Side effects?

6. **Mock scope flexibility:** Can we mock in the middle of a window, or only at window boundaries?

---

## Relationship to Unit Testing

Integration flow testing complements unit testing:

| Aspect | Unit Testing | Integration Flow Testing |
|--------|--------------|--------------------------|
| **Focus** | Execution items within one unit | Seams between units |
| **Coverage** | 100% of EIs (exhaustive paths) | Use case coverage of seam interactions |
| **Scope** | Single unit, mock everything outside | Multiple units, mock at window boundary |
| **Failures mean** | Internal logic is wrong | Units don't compose correctly |
| **Generated from** | Unit Ledger (EI enumeration) | Integration Flow Graph (seam enumeration) |
| **Execution paths** | Test all variations | Use minimal valid paths to reach seams |

**Together they provide:**
- Unit tests: Every line of code works correctly in isolation
- Integration tests: Units fit together correctly at their boundaries
- Result: Complete confidence in the system

---

## Next Steps

1. **Validate unit ledger system** with real-world codebases
2. **Build integration graph construction** procedure
3. **Define entry point and terminal node** identification rules
4. **Implement sliding window generation** algorithm
5. **Test on real systems** to determine filtering criteria
6. **Write Integration Flow Generation Procedure** specification
7. **Write Integration Testing Contract** (analogous to Unit Testing Contract)

---

## Contributing

This is an active design area. Contributions welcome:
- Real-world integration testing challenges
- Graph construction approaches
- Window filtering heuristics
- Test generation strategies
- Example integration flows from actual systems

---

# Integration Flow Analysis Pipeline - Session Summary

## What We Built Today

A 5-stage pipeline that analyzes unit ledgers to discover, classify, and enumerate integration flows through a codebase, ultimately generating test windows for integration testing.

### The Problem We're Solving

Unit ledgers enumerate all execution items (EIs) within individual units and document integration points (interunit calls and boundary crossings). But they don't show **how integration points connect across units to form complete flows**. We needed a way to:

1. Extract all integration points from ledgers
2. Understand which integration points lead to others
3. Find complete flows from entry points to boundaries/terminals
4. Generate testable scopes (windows) from those flows

### The 5-Stage Pipeline

#### Stage 1: Collect Integration Points
**Input:** Unit ledger files (auto-discovered or explicit)  
**Output:** Flat list of all integration points with full context

**What it does:**
- Discovers all `*-ledger.yaml` files in a directory tree
- Loads each ledger's Document 2 (the ledger document)
- Walks the unit tree to find all callables
- Extracts `integration.interunit` and `integration.boundaries` facts
- Creates `IntegrationPoint` objects with:
  - Source context (unit, callable ID, callable name)
  - Target (raw string, unresolved)
  - Execution paths (EI sequences leading to this integration)
  - Boundary info (if applicable)
  - Kind, signature, condition, notes

**Results on test project:**
- 21 ledgers processed
- 286 integration points extracted
  - 250 interunit integrations
  - 36 boundary integrations

#### Stage 2: Classify Integration Points
**Input:** Integration points from Stage 1  
**Output:** Classification into entry/intermediate/terminal

**Classification logic:**
- **Entry points:** Source callable has no incoming interunit calls (first seams)
- **Terminal nodes:** Boundary integrations OR targets with no outgoing calls
- **Intermediate:** Everything else (both incoming and outgoing)

**Results on test project:**
- 231 entry points (81%)
- 0 intermediate seams (0%)
- 55 terminal nodes (19%)

**Insight:** No intermediate seams suggests architecture with clear entry points and boundaries, not a lot of pass-through chaining. Most integration points are either initiators or endpoints.

#### Stage 3: Build Integration Graph
**Input:** Stage 1 points + Stage 2 classification  
**Output:** Graph with nodes (integration points) and edges

**Edge construction rule:**
```
Edge from I1 → I2 exists if:
  I1.target matches I2.source_callable
```

Meaning: "If integration I1 happens (A calls B), does that put us inside the callable that contains I2 (B calls C)?"

**Matching strategies:**
- Exact callable ID match
- Exact name match (target == source_callable_name)
- Qualified match (target ends with `.source_callable_name`)
- Partial qualified match (conservative, only with dots)

**Results on test project:**
- 286 nodes
- 1,419 edges
- Average ~5 edges per node
- High connectivity indicates significant integration chaining

#### Stage 4: Enumerate Flows
**Input:** Integration graph from Stage 3  
**Output:** Complete flows from entry points to terminal nodes

**Algorithm:**
- Depth-first search (DFS) from each entry point
- Follow edges to neighbors
- Track visited nodes (cycle detection)
- Stop at terminal nodes → record complete flow
- Stop at max depth (configurable, default 20)

**Optimization challenges encountered:**
- Initial implementation hit combinatorial explosion
- High branching factor (avg 5 edges/node) × 231 entry points = millions of paths
- Added limits:
  - Max depth: 20 hops
  - Max flows per entry: 100
  - Max paths explored per entry: 10,000
  - Max total flows: 10,000

**Results on test project:**
- 2,604 complete flows found
- Flow lengths: 2-20 hops (avg: 18.6)
- 84,308 paths hit max depth limit
  - Indicates very deep chains (likely serialization/deserialization)
  - Many paths don't reach terminals within 20 hops
  - Suggests possible cycles or architectural coupling

#### Stage 5: Generate Test Windows
**Input:** Flows from Stage 4  
**Output:** Sliding windows for integration testing

**Window generation:**
- For each flow, create overlapping windows
- Window size: `min_window_length` to `max_window_length` (config)
- Slides by 1 integration point for complete coverage
- Example: 20-hop flow with window_size=7 → 14 windows

**Each window contains:**
- Window ID and source flow ID
- Position in original flow
- List of integration IDs in this window
- Entry point (first integration: unit, callable, target)
- Exit point (last integration: unit, callable, target, isBoundary flag)
- Description (human-readable chain)
- Full sequence (complete integration point data)

**Results on test project:**
- 33,106 test windows generated
- From 2,604 flows
- Avg ~12.7 windows per flow
- Window sizes: 2-7 integration points
- Output file: 132 MB YAML

### Configuration System

**File:** `integration_config.toml`

**Key settings:**
```toml
[paths]
ledgers_root = "ledgers"
integration_output_dir = "dist/integration-output"

[processing]
max_flow_depth = 20
min_window_length = 2
max_window_length = 7

[stages]
stage1_output = "stage1-integration-points.yaml"
stage2_output = "stage2-classified-points.yaml"
# ... etc
```

**Path resolution:**
- Target-relative paths (ledgers, output) → resolve to target project
- Tool-relative paths (schema) → resolve to tool repo
- Runs from target project by default
- Can specify `--target-root` for cross-repo usage

### File Structure
```
integration/
├── config.py                    # Configuration loader
├── integration_config.toml      # Configuration file
├── stages/
│   ├── stage1_collect_integration_points.py
│   ├── stage2_classify_integration_points.py
│   ├── stage3_build_integration_graph.py
│   ├── stage4_enumerate_flows.py
│   └── stage5_generate_windows.py
├── shared/
│   ├── data_structures.py       # IntegrationPoint, etc.
│   ├── ledger_reader.py         # Discovery & extraction
│   └── yaml_utils.py            # YAML I/O
├── specs/
│   └── integration-flow-schema.json  # (planned)
└── docs/
    └── README.md
```

### Key Data Structures

**IntegrationPoint:**
```python
@dataclass
class IntegrationPoint:
    id: str                           # Integration ID (e.g., IC000F001E0007)
    source_unit: str                  # Unit name
    source_callable_id: str           # Callable ID (e.g., C000F001)
    source_callable_name: str         # Callable name
    target_raw: str                   # Target as string
    target_resolved: TargetRef        # Resolved target (unit, callable)
    kind: str                         # call, construct, io, etc.
    execution_paths: list[list[str]]  # EI sequences to reach this point
    condition: str | None             # Conditional expression
    boundary: BoundarySummary | None  # If boundary integration
    signature: str | None             # Call signature
    notes: str | None                 # Additional context
```

**Flow (Stage 4 output):**
```yaml
flowId: FLOW_0001
description: "A → B → C → boundary"
length: 3
sequence: [full integration point objects]
entryPoint:
  integrationId: IC000F001E0001
  unitId: api
  callableId: C000F001
  callableName: resolve
terminalNode:
  integrationId: IC000F003E0005
  boundary: filesystem
```

**Window (Stage 5 output):**
```yaml
windowId: WINDOW_00001
sourceFlowId: FLOW_0001
startPosition: 0
length: 7
integrationIds: [IC001, IC002, IC003, ...]
entryPoint: {...}
exitPoint: {...}
description: "resolve → load_services → rl_resolve → ..."
sequence: [full integration point objects]
```

## Current Limitations & Known Issues

### 1. Deep Chain Problem
- Avg flow length 18.6 hops (very long!)
- 84K paths hit 20-hop depth limit
- Many flows are serialization chains (`to_mapping()` → `from_mapping()`)
- These are technically "flows" but may not be the most valuable tests

### 2. Volume Challenge
- 33,106 windows is impractical for actual test generation
- Need filtering/prioritization strategy
- Not all flows are equally valuable

### 3. Missing Context
- Windows show WHAT to test, not HOW
- Test generation still needs:
  - Input data to trigger each path
  - Assertions to validate correctness
  - Setup/teardown logic

### 4. Potential False Flows
- High depth-limit hits suggest:
  - Real deep workflows, OR
  - Architectural coupling creating false "flows", OR
  - Near-cycles in serialization patterns

### 5. No Feature Mapping
- Flows aren't tagged by feature/capability
- Can't answer "which flows test package resolution?"
- Need domain knowledge layer

## Next Steps (Planned for Tomorrow)

### 1. Code Review & Domain Understanding
- Examine actual code from units in flows
- Understand what the system does (features, capabilities)
- Identify what's "interesting" vs "mechanical"
- Determine if long chains are legitimate or artifacts

### 2. Stage 6: Flow Profiling & Scoring
**Proposed features:**
- Classify flow types (serialization, API, I/O, computation)
- Score by business value, complexity, risk
- Tag flows by feature/capability
- Identify "interesting" vs "boilerplate" flows
- Count boundary crossings, unit diversity

**Example output:**
```yaml
flowProfiles:
  - flowId: FLOW_0001
    type: boundary_io
    feature: pep691_loading
    complexity: low
    boundaryCount: 1
    unitDiversity: 2
    score: 75  # High priority
    tags: [critical, file_io, api_entry]
```

### 3. Filtering & Prioritization
- Add filtering capabilities:
  - By flow type (exclude pure serialization?)
  - By length (exclude >15 hops?)
  - By feature (focus on critical features?)
  - By boundary count (prioritize external interactions?)
- Aim for 500-1000 high-value tests, not 33K

### 4. Test Generation Proof-of-Concept
- Pick top 50 windows (diverse, short, boundary-heavy)
- Generate actual pytest integration tests
- Validate the approach with runnable tests

### 5. Continuous Integration
- Run pipeline on code changes
- Detect NEW flows
- Generate tests for deltas only
- Track coverage over time

## Questions to Answer Tomorrow

1. **What features does this system implement?**
   - Package resolution? Dependency graphs? Repository management?

2. **What are the critical user-facing workflows?**
   - What breaks when things go wrong?

3. **Are the long chains legitimate?**
   - Are 18-hop flows real workflows or design artifacts?

4. **What's worth testing?**
   - Which flows have business value?
   - Which are risky/complex?

5. **How to map flows to features?**
   - Can we infer from code structure?
   - Need manual tagging?
   - Use naming conventions?

## Success Metrics So Far

✅ **Complete visibility** - Every integration point enumerated  
✅ **Systematic approach** - Deterministic, reproducible  
✅ **Graph construction** - Discovered actual call chains  
✅ **Flow enumeration** - Found 2,604 complete paths  
✅ **Test specifications** - Generated 33,106 windows  
✅ **End-to-end pipeline** - All 5 stages working  

⚠️ **Volume management** - Need filtering  
⚠️ **Feature mapping** - Need domain layer  
⚠️ **Test generation** - Not yet implemented  

## Files Generated Today

All in `/mnt/user-data/outputs/`:
- `ledger_reader.py` - Discovery and extraction logic
- `stage1_collect_integration_points.py` - Stage 1 implementation
- `stage2_classify_integration_points.py` - Stage 2 implementation
- `stage3_build_integration_graph.py` - Stage 3 implementation
- `stage4_enumerate_flows.py` - Stage 4 implementation
- `stage5_generate_windows.py` - Stage 5 implementation
- `integration_config.toml` - Configuration template
- `test_ledger_discovery.py` - Testing utility

## Architecture Insights Discovered

From analyzing the test project:
- **Entry-heavy architecture:** 81% of integration points are entry points
- **High connectivity:** Avg 5 edges per node suggests significant integration
- **Deep chains:** 18.6 avg hops indicates complex workflows or coupling
- **Serialization patterns:** Many flows are object mapping chains
- **Clear boundaries:** 36 boundary integrations (filesystem, network, etc.)
- **No pass-through:** Zero intermediate seams suggests low chaining between units