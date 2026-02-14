# Unit Ledger Generation Procedure

## 1. Introduction and Overview

### 1.1 Motivation and Need

Software testing faces a persistent challenge: **how do we know we've tested
everything?** Code coverage tools report percentages, but they don't provide a
systematic inventory of what needs testing or a deterministic path to 100%
coverage.

The Unit Ledger addresses this by creating a **complete, language-agnostic
inventory** of all execution items (EIs) in a unit. Instead of hoping tests
cover everything, the ledger provides:

- **Enumeration**: Every distinct outcome path gets a unique identifier
- **Verification**: Test coverage becomes a checklist - did we exercise every
  EI?
- **Traceability**: Links between code, tests, requirements, and cross-language
  ports
- **Speed**: AI can generate comprehensive tests from a complete inventory

This procedure emerged from observing that AI systems (and humans) can be prone
to making consistent errors when left to interpret "be thorough." By making the
enumeration process largely **procedural rather than interpretive**, we achieve
deterministic, repeatable results.

### 1.2 Goals and Intent

**Primary Goal**: Enable 100% branch coverage testing through the exhaustive
enumeration of execution items.

**Secondary Goals**:
- Support integration test planning by capturing all boundary crossings and
  inter-unit calls
- Enable cross-language porting with preserved test coverage
- Provide requirement-to-code-to-test traceability
- Create machine-parseable, human-readable artifacts

**Key Principles**:
- **Thoroughness over efficiency**: Enumerate every outcome path, even if tests
  will consolidate them
- **Observation over inference**: Capture what's literally present in the code
- **Determinism**: Same input results in the same ledger, every time
- **Language agnostic**: Works for Python, Java, Kotlin, Rust, etc.

### 1.3 Scope: What This *Is* and What This Is *Not*

**This procedure IS:**
- A systematic method for enumerating all execution items in a unit
- A specification for capturing integration facts (inter-unit calls, external
  libraries, and boundary crossings)
- A foundation for generating comprehensive unit tests
- A language-agnostic inventory format
- An inventory of *observed* content

**This procedure IS NOT:**
- A test generation procedure (that comes after the ledger exists)
- An integration test specification (flows are derived from multiple ledgers)
- A code quality analysis tool
- A replacement for coverage measurement tools
- Analysis or interpretation of content

**Key Distinction**: The ledger enumerates "what can happen" (outcome paths),
not "what values to test" (test parameterization). A single outcome path might
require dozens of test cases to verify correctness across different inputs.

### 1.4 Brief Stage Overview

The procedure consists of five sequential stages, executed per callable:

**Stage 1: Outcome Path Analysis**
- Input: Callable source code (compilation unit)
- Process: Line-by-line analysis to identify distinct outcome paths
- Output: Hierarchical/nested dict mapping line numbers to outcome descriptions
- Gate: Verify every executable line has been analyzed

**Stage 2: EI ID Assignment**
- Input: Outcome maps from Stage 1
- Process: Assign sequential EI IDs to each outcome path
- Output: Mapping of EI IDs to line numbers and outcome descriptions
- Gate: Verify ID count matches total outcome count

**Stage 3: Integration Fact Enumeration**
- Input: Code and EI mappings from Stage 2
- Process: Identify inter-unit calls, external library calls, and boundary
  crossings
- Output: Integration facts with executionPaths referencing EI IDs
- Gate: Verify every integration fact references valid EI IDs

**Stage 4: Document Generation**
- Input: All data from Stages 1-3
- Process: Generate the three-document YAML (IDs, Ledger, Review)
- Output: Multi-document YAML file
- Gate: Structural validity check

**Stage 5: Schema Validation**
- Input: Generated YAML from Stage 4
- Process: Validate against JSON schema
- Output: Validation report
- Gate: Schema compliance confirmed

Each stage builds on the previous, and verification gates ensure correctness
before proceeding.

## 2. Definitions and Shapes

### 2.1 Core Concepts

#### Execution Item (EI)

An **Execution Item (EI)** is an atomic unit of execution used for enumeration
and coverage verification. Each EI represents a distinct outcome path that can
occur when executing a single line of code.

**Key Properties:**

- Anchored to exactly one physical source line (one line number)
- MUST NOT span multiple lines
- Represents one specific outcome for that line
- Every executable line has at least one EI

**Relationship to traditional "branches":**

The term "branch" traditionally means a control flow decision point (if/else,
switch, etc.). In this specification, we use "Execution Item" to be more
precise: an EI is the atomic unit of execution, not just decision points. A
single `if` statement creates 2 EIs (true path, false path). A ternary
expression creates 2 EIs. A simple assignment creates 1 EI.

**EI enumeration rules:**

1. EI is always anchored to exactly one physical source line
2. An EI MUST NOT include, imply, or represent execution across multiple lines
3. Every physical source line has at least 1 EI
4. A single physical source line may have more than 1 EI if that line contains
   multiple mutually-exclusive outcomes
5. "Main flow" is not an EI. It is a sequence of EIs (one per executed line)
6. Control statements (if/elif/else, match/case, try/except/finally, for/while,
   with) each contribute one EI per distinct outcome when they appear on that
   line
7. EI IDs: one ID per (lineno, outcome). If only one outcome exists, there is
   exactly one EI for that line
8. If a line has no conditional or multi-outcome construct, it has exactly 1 EI
9. Any code sourced from the unit should be taken verbatim
   - Enclosed in single quotes
   - Any internal single quotes are doubled (two single quotes, not replaced
     with a double quote)
   - Single lines with line breaks are joined into a single line


**Examples (Python):**

```python
# Line 42: Simple assignment - 1 EI
x = calculate_value()

# Line 43: If statement - 2 EIs
if x > 0:  # EI-1: x > 0 is true → enter block
           # EI-2: x > 0 is false → skip block

# Line 47: Ternary expression - 2 EIs
result = value_a if condition else value_b
           # EI-1: condition is true → value_a assigned
           # EI-2: condition is false → value_b assigned

# Line 51: List comprehension with filter - 3 EIs
items = [x for x in data if x.valid]
           # EI-1: data is empty → no iteration, items = []
           # EI-2: data has items, all filtered out → items = []
           # EI-3: data has items, some pass filter → items populated
```

**Examples (Java):**

```java
// Line 42: Simple assignment - 1 EI
int x = calculateValue();

// Line 43: If statement - 2 EIs
if (x > 0) {  // EI-1: x > 0 is true → enter block
              // EI-2: x > 0 is false → skip block

// Line 47: Ternary expression - 2 EIs
String result = condition ? valueA : valueB;
              // EI-1: condition true → valueA assigned
              // EI-2: condition false → valueB assigned

// Line 51: Stream with filter - 3 EIs
List<Item> items = data.stream()
    .filter(Item::isValid)
    .collect(Collectors.toList());
// EI-1: data is empty → no elements, items = []
// EI-2: data has items, all filtered out → items = []
// EI-3: data has items, some pass filter → items populated
```

#### Distinct Outcome Path

A **distinct outcome path** is a unique way that execution can flow through a
line of code to produce a result. This is NOT about different input values,but
about different execution behaviors.

**Key Distinction - Outcome Paths vs Test Cases:**

- `x - 2` has **1 outcome path** (subtraction always executes the same way)
- Testing with inputs 5, 10, 100 creates **3 test cases**, not 3 outcome paths
- Outcome paths are about **control flow**, not **data variation**

A single outcome path might require dozens or hundreds of test cases to verify
correctness across different inputs. The ledger enumerates outcome paths; test
generation parameterizes across input variations.

**Example:**
```python
# This function has 6 outcome paths:
def validate(value: int) -> str:
    if value < 0:           # EI-1: value < 0 true
        return "negative"
                            # EI-2: value < 0 false
    if value == 0:          # EI-3: value == 0 true
        return "zero"
                            # EI-4: value == 0 false
    return "positive"       # EI-5: implicit else, returns "positive"
    
# But testing might use 100 different input values to verify
# correctness across the range. That's 100 test cases, not 100 EIs.
```

#### Integration Facts

**Integration facts** capture how a unit interacts with code outside its
boundaries. These facts enable integration test planning and flow analysis.

Three categories exist:

**Interunit Integration:**

Calls or data exchanges with other units within the same project/system.

Example:
```python
# In unit validation.py, calling unit parser.py
from parser import parse_config

def validate_config(raw: str) -> bool:
    config = parse_config(raw)  # ← Interunit integration
    return config.is_valid()
```

**External Library Integration:**

Calls or data exchanges with third-party libraries that are not part of the
project codebase and do not cross external system boundaries.

Example:
```python
import requests
from lxml import etree

def process_data(url: str) -> dict:
    response = requests.get(url)  # â† extlib (but also boundary: network)
    doc = etree.fromstring(response.text)  # â† extlib
    return parse_doc(doc)
```

**Boundary Integration:**

Calls or data exchanges that cross external system boundaries (network,
filesystem, database, etc.).

Example:
```python
import requests

def fetch_user(user_id: str) -> dict:
    response = requests.get(f"/api/users/{user_id}")  # ← Boundary
    return response.json()
```

**Classification During Ledger Generation:**

When identifying an integration point during Stage 3, classify it using this
decision process:

1. **Load project type inventory**: If a file listing project types is
   available in project files (typically named `project-inventory.txt` or
   `PROJECT_TYPES.txt`), load it into memory as a lookup set
   
2. **For each integration target**:
   - If a target matches an entry in the project type inventory → **interunit**
   - If a target crosses an external system boundary (filesystem, network,
     database, clock, randomness, env, etc.) → **boundary**
   - Otherwise → **extlib**

3. **Overlapping cases**: If an external library call also crosses a system
   boundary (e.g., `requests.get()`), classify as **boundary** since the
   boundary crossing is the primary concern for integration testing

**Project Type Inventory Format:**

If provided, the project type inventory file contains one fully qualified name
per line. Inventory of python files might look like:
```
project_name.module.ClassName
project_name.module.ClassName.method_name
project_name.module.function_name
```

**If no project type inventory is provided**, use import analysis to infer
interunit vs. extlib by checking if imports reference modules within the
project's package namespace. When uncertain about a designation, add a
note to the summary document (Document 3) describing the uncertainty.

**Critical Integration Fact Properties:**

- Each integration fact is the EI ID for that item, prefixed with "I"
  (e.g., EI ID = C000F001B0003, integration fact ID = IC000F001B0003)
- `executionPaths` field lists all EI ID sequences that lead to the integration
- The final EI in each execution path must be the integration's own EI ID
- If `executionPaths` is empty, the integration is unreachable (dead code) and
  will be reported in the Review (Document 3)
- Be sure to enumerate ALL `executionPaths` that can reach the integration
  point. For N independent conditionals before an integration, generate 2^N
  paths (Cartesian product of all conditional outcomes).
- Any code sourced from the unit should be taken verbatim
  - Enclosed in single quotes
  - Any internal single quotes/apostrophes are escaped according to the YAML
    rules: In YAML, you escape a single quote (') within a single-quoted string
    by using two single quotes consecutively ('').  
  - Single lines with line breaks are joined into a single line

### 2.2 Operation Metadata Decorators

**Purpose**: Operation metadata decorators are comment-based annotations that
mark code elements with metadata for consumption by downstream processes.
They act as lightweight markers that can influence code analysis, test
generation, flow enumeration, and other automated tooling without requiring
changes to the actual code structure.

**Why This Matters**: As codebases grow and automation becomes more
sophisticated, different tools need different information about code elements.
Rather than encoding this information in separate configuration files or
tool-specific formats, operation metadata decorators provide a standardized,
human-readable way to annotate code with machine-processable metadata directly
at the point of definition.

#### Syntax

**Format**:
```
<comment> :: <DecoratorName> | type=<value> [ | <field>=<value> ... ]
```

Where:
- `<comment>` is the language-specific comment syntax (`#` for Python, `//`
  for Java/C-family, or within `"""` docstrings / `/** */` Javadoc comments /
  `/* ... */` c/Java multiline comments)
- `::` is the decorator marker (space after `::` is required)
- `<DecoratorName>` is the decorator identifier
- `type=<value>` is a required field specifying the decorator subtype
- Additional field-value pairs may follow, separated by `|`

**Standard Fields**:

| Field Name | Required | Description                            |
|------------|----------|----------------------------------------|
| `type`     | Yes      | Subclassification within the decorator |
| `alias`    | No       | Alternate name for the callable        |
| `desc`     | No       | Human-readable description             |

**Extensibility**: Additional fields may be defined as needed. Tools should
preserve unknown fields in the `extra` map.

**Format Rules:**
- Space after `::` is required
- Case-sensitive decorator and field names
- Field values may contain spaces (no quotes needed)
- If field value contains `|`, enclose the entire value in single or double
  quotes: `desc="Converts to dict | includes metadata"`
- Multiple field-value pairs separated by `|`

**Location**: Operation metadata decorators may appear in:
- Single-line comment immediately preceding the callable definition
- Within the callable's docstring (Python) or Javadoc comment (Java)

If decorators appear in multiple comment locations for the same callable:
- Identical decorators → Capture once
- Same decorator with different kwargs → Capture the one with more complete
  information (more fields)
- Different decorators → Capture both
- Duplication → Note in Document 3 findings as a warning

**Examples:**

Python (single-line comment):
```python
# :: MechanicalOperation | type=serialization
def to_mapping(self) -> Mapping[str, Any]:
    """Convert to dictionary representation."""
    return {
        "overrides": self.overrides,
        "mode": self.mode.value,
    }
```

Python (in docstring):
```python
def to_mapping(self) -> Mapping[str, Any]:
    """Convert to dictionary representation.
    
    :: MechanicalOperation | type=serialization | alias=ToDict
    """
    return {
        "overrides": self.overrides,
        "mode": self.mode.value,
    }
```

Java (single-line comment):
```java
// :: MechanicalOperation | type=serialization
public Map<String, Object> toMapping() {
    // Convert to map representation
    ...
}
```

Java (in Javadoc):
```java
/**
 * Convert to map representation.
 * 
 * :: UtilityOperation | type=validation | alias=ValidDict
 * @param desc Human-readable description
 * @param map  Map to validate
 */
public static void validateMap(String desc, Map<String, Object> map) {
    // Validate map against schema
    ...
}
```

Multiple fields with quoted value:
```python
# :: MechanicalOperation | type=serialization | desc="Converts to dict | includes metadata"
def to_mapping(self) -> Mapping[str, Any]:
    ...
```

#### Capturing in Ledgers

**During ledger generation**, operation metadata decorators are captured in
the `decorators` field of the Entry or CallableSpec using the existing
Decorator shape.

**Decorator Shape**:
```yaml
decorators:
  - name: <DecoratorName>
    kwargs:
      type: <type_value>
      alias: <alias_value>     # if present
      desc: <desc_value>       # if present
      <custom>: <value>        # any additional fields
```

**Field Mapping**:
- Decorator name → `name` field
- All field-value pairs → `kwargs` object
- Unknown/custom fields → Preserved in `kwargs`

**Example in Document 2 (Ledger)**:
```yaml
- id: C001M003
  kind: callable
  name: to_mapping
  signature: 'to_mapping(self) -> Mapping[str, Any]'
  decorators:
    - name: MechanicalOperation
      kwargs:
        type: serialization
        alias: ToDict
  callable:
    # ... branches, params, etc.
```

**In Document 3 (Review)**, note decorator usage:
```yaml
findings:
  - severity: info
    category: assumption
    message: 'Unit uses 2 operation metadata decorators 
              (1 MechanicalOperation, 1 UtilityOperation)'
  - severity: warn
    category: anomaly
    message: 'Callable C001M003 has duplicate operation metadata decorators
              in both docstring and single-line comment'
```

#### Recognized Operation Metadata Decorators

Operation metadata decorators are organized by category based on their
intended use by downstream processes.

##### Category 1: Flow Analysis Control

These decorators control how operations are treated during integration flow
enumeration and traversal. Flow analysis tools use these decorators to
determine which operations should be traversed during flow enumeration and
which should be replaced with fixture placeholders.

###### 1.1 Flow Exclusion

Flow exclusion decorators mark operations that should not be traversed during
integration flow enumeration. When flow analysis encounters a decorated node,
it emits a fixture placeholder (e.g., `FIXTURE_C000F001`) instead of
traversing into the operation's integration points. This prevents exponential
flow explosion from mechanical or utility operations while preserving the
flow structure for test generation.

**1.1.1 `MechanicalOperation`**

**Decorator Name**: `MechanicalOperation`
**Purpose**: Identifies mechanical data transformation operations that should
be replaced with fixtures during flow enumeration

Marks operations that perform deterministic data transformations without
business logic. These operations are thoroughly tested in unit tests and add
little value when included in integration test flows.

**Standard type values:**
- `serialization` - Converts objects to serialized formats
- `deserialization` - Converts serialized formats to objects
- `conversion` - Converts between data representations
- `formatting` - Formats data for display or output

**Typical operations:**
- **Serialization**: `to_dict()`, `to_mapping()`, `to_json()`, `serialize()`
- **Deserialization**: `from_dict()`, `from_mapping()`, `parse()`,
  `deserialize()`
- **Format conversion**: `to_string()`, `to_xml()`, `to_protobuf()`
- **Type conversion**: `as_list()`, `as_tuple()`, `cast_to()`

**Characteristics:**
- Deterministic transformations
- No business rules or validation
- No side effects (pure functions preferred)
- Generally low branching complexity

**Example:**
```python
# :: MechanicalOperation | type=serialization | alias=ToDict
def to_mapping(self) -> Mapping[str, Any]:
    """Convert to dictionary representation."""
    return {"id": self.id, "name": self.name}
```

**1.1.2 `UtilityOperation`**

**Decorator Name**: `UtilityOperation`
**Purpose**: Identifies cross-cutting utility operations that should be
replaced with fixtures during flow enumeration

Marks operations that provide infrastructure services used by many features.
These operations are typically well-tested independently and create flow
explosion when traversed due to their widespread use.

**Standard type values:**
- `validation` - Validates data structure or format
- `logging` - Records events or state
- `caching` - Stores/retrieves cached data
- `configuration` - Reads/manages configuration settings

**Typical operations:**
- **Validation**: `validate_schema()`, `check_format()`, `assert_valid()`
- **Logging**: `log_event()`, `write_audit()`, `record_metric()`
- **Caching**: `cache_get()`, `cache_set()`, `memoize()`
- **Configuration**: `load_config()`, `get_setting()`, `read_env()`

**Characteristics:**
- Used by many different business flows
- Often infrastructure/framework code
- May have conditional logic but not business-specific
- Often boundary operations (logging, config, cache)

**Example:**
```python
# :: UtilityOperation | type=validation | alias=ValidDict
def validate_typed_dict(desc: str, mapping: dict, ...) -> None:
    """Validate dict against TypedDict schema."""
    ...
```

##### Category 2: Feature Architecture (Reserved)

This category is reserved for future decorators that mark architectural roles
in feature implementation. Examples might include entry point markers,
terminal operation markers, or orchestrator identifiers.

#### Downstream Process Behavior

**Flow Analysis and Test Generation**: Operation metadata decorators are
consumed by downstream flow enumeration and test generation processes. When
flow analysis encounters an integration node marked with a flow exclusion
decorator (Category 1), it emits a fixture placeholder (e.g.,
`FIXTURE_C000F001`) instead of traversing the node's edges. Test generation
tools then create appropriate mocks or fixtures for these placeholders.

**Detailed flow enumeration behavior**: See the Integration Test Contract
specification for complete details on how flow analysis consumes decorator
metadata and generates fixture placeholders.

**Example impact:**
- Without decorators: `to_mapping()` with 3 execution items creates 3-way
  branching at every call site, leading to exponential flow explosion
- With `:: MechanicalOperation | type=serialization`: Flow traversal emits
  `FIXTURE_C000F001` and skips traversal, treating it as a pass-through
- Observed reduction: 135MB → 11MB flow output, 18.6 → 6.5 average hops

### 2.3 Data Structures

#### Outcome Map

The **outcome map** is the intermediate structure produced during Stage 1
(Outcome Path Analysis). It maps line numbers to lists of distinct outcome
descriptions.

**Structure:**
```yaml
{
    "id": "C000F001",
    "name": "validate_typed_dict",
    "line_range": [13, 52],
    "outcome_map": {
        36: ["executes: allowed_env_keys assigned"],
        37: ["executes: bad_keys computed"],
        38: ["bad_keys truthy → raise ValueError", 
                "bad_keys falsy → continue to next check"],
        40: ["mapping empty → bad_vals = []",
                "all values valid → bad_vals = []",
                "some values invalid → bad_vals populated"],
        45: ["bad_vals truthy → raise ValueError",
                "bad_vals falsy → return None"],
        # ...
    }
}
```

**Properties:**

- Line numbers are absolute (from the source file)
- Non-executable lines (comments, blanks) are omitted
- Empty lists indicate lines with no branching (single outcome)
- List length at each line = number of EIs for that line
- Sum of all list lengths = total EI count for the callable

### 2.4 YAML Document Shapes

The Unit Ledger consists of three YAML documents. Below are the key shapes used
in these documents. For complete shape templates and all optional fields, see
the Unit Ledger Specification.

As a reminder for any data included in YAML, any code sourced from the unit
should be taken verbatim:
- Enclosed in single quotes
- Any internal single quotes are doubled (two single quotes, not replaced
  with a double quote)
- Single lines with line breaks are joined into a single line


#### EiSpec

Describes a single execution item.

**Required fields:**

- `condition`: string, copied from code (e.g., "if bad_keys")
- `id`: string, unique EI identifier (e.g., "C000F001E0001")
- `outcome`: string, observable result (e.g., "raises ValueError")

**Optional fields:**

- `precondition`: string, required state for this EI to be reachable
- `notes`: string, clarifications, or context
- `test`: `TestGuide`, hints for test generation
- `extra`: map, extension fields
- `inferred`: map, derived facts (if enabled)

**Example:**

```yaml
- id: C000F001E0001
  condition: 'if bad_keys'
  outcome: 'raises ValueError with "Invalid {desc} keys: {bad_keys}"'
  notes: 'bad_keys is computed as set difference on line 37'
```

#### IntegrationFact

Describes an integration point (interunit, extlib, or boundary).

**Required fields:**

- `executionPaths`: list[list[string]], EI ID sequences leading to this
  integration
- `id`: string, integration ID prefixed with "I" (e.g., "IC000F001E0003")
- `target`: string, fully qualified target identifier

**Optional fields:**

- `boundary`: `BoundarySummary`, if this crosses an external boundary
- `condition`: string, if integration is conditional
- `contract`: `ContractSummary`, target's call contract details
- `kind`: enum, one of: call, construct, import, dispatch, io, other
- `signature`: string, call signature if readily available
- `via`: string, intermediary helper if applicable
- `test`: `TestGuide`, mocking/stubbing hints
- `notes`: string
- `extra`: map
- `inferred`: map

**Example:**

For `interunit`:

```yaml
interunit:
  - id: IC003M002E0002
    target: validate_typed_dict
    kind: call
    signature: 'validate_typed_dict("marker_env overrides", overrides_map, EnvironmentOverrides, str)'
    executionPaths:
      - [C003M002E0002]  # Always called if execution reaches line
    notes: 'Validates overrides dict against TypedDict schema'
```

For `extlib`:

```yaml
extlib:
  - id: IC003M002E0005
    target: lxml.etree.fromstring
    kind: call
    signature: 'etree.fromstring(xml_text)'
    executionPaths:
      - [C003M002E0003, C003M002E0005]
    notes: 'Third-party XML parsing library'
```

#### BoundarySummary

Describes external system details for boundary integrations.

**Required fields:**

- `kind`: enum, one of: filesystem, database, network, subprocess, message_bus,
  clock, randomness, env, other

**Optional fields:**

- `endpoint`: string (host, URL, topic, queue, etc.)
- `operation`: string (query, read, write, publish, consume, etc.)
- `protocol`: string (http, grpc, sql, amqp, kafka, etc.)
- `resource`: string (table, bucket, path, collection, etc.)
- `system`: string (postgres, s3, kafka, service-name, etc.)
- `notes`: string
- `extra`: map

**Example:**

```yaml
boundary:
  - id: IC005F002E0007
    target: requests.get
    kind: call
    boundary:
      kind: network
      protocol: http
      system: external-api
      endpoint: 'https://api.example.com'
      operation: read
    executionPaths:
      - [C005F002E0001, C005F002E0003, C005F002E0007]
```

#### ContractSummary

Language-agnostic summary of an integration target's call contract.

**All fields optional (omit if unknown):**

- `interaction`: enum, one of: request_response, fire_and_forget, stream_out,
  stream_in, pubsub, async_job, other
- `signature`: string, human-readable signature
- `returnType`: `TypeRef`
- `params`: list[`ContractParam`]
- `raises`: list[`TypeRef`], exceptions that may be raised
- `notes`: string
- `extra`: map

**Example:**

```yaml
contract:
  interaction: request_response
  signature: 'parse_config(raw: str) -> Config'
  returnType:
    name: Config
    unit: config_parser
  params:
    - name: raw
      type:
        name: str
  raises:
    - name: ValueError
      notes: 'If raw is malformed JSON'
```

### 2.5 Terminology Notes

**"Branch" vs. "Execution Item":**

In the current YAML schema and many existing ledgers, the term "branch" is used
(BranchSpec, Branch ID, etc.). This specification uses "Execution Item" or "EI"
to be more precise about what we're enumerating. When reading existing schemas
or ledgers, treat "branch" and "execution item" as synonymous.

**Decorators and Annotations:**

Decorators (Python `@decorator`) and annotations (Java `@Annotation`) are
captured in the ledger's `decorators` field as **metadata about behavior**, but
they do not themselves create execution items. The behavior they add is
typically:

- Cross-cutting concerns (caching, retry, transactions)
- Framework hooks (test lifecycle, dependency injection)
- Compile-time transformations (dataclass generation)

These should be mocked/stubbed in unit tests, and may indicate boundary
crossings (e.g., `@transactional` implies database interaction).

**Observed vs Inferred:**

By default, ledgers contain **observed** facts only: what is literally present
in the source code (decorators, keywords, condition text, signatures). Inferred
facts (derived from language semantics) may be included under an `inferred` map
if explicitly enabled, but the default mode is observation-only.

## 3. Stage-by-Stage Procedure

This section defines the five sequential stages for generating a Unit Ledger.
Each stage must be completed before proceeding to the next. Built-in
verification gates ensure correctness at each step.

**Processing Model: Per-Callable**

The procedure processes one callable at a time, completing all stages for that
callable before moving to the next. This prevents context loss and makes errors
easier to isolate.

**Output Modes:**

- **Default mode**: Process all stages silently, output stage summaries, deliver
  final YAML
- **Debug mode**: Output intermediate artifacts after each stage, pause for
  human confirmation

### 3.1 Stage 1: Outcome Path Analysis

**Purpose:** Analyze each line of a callable to identify all distinct outcome
paths, producing an outcome map.

**Input:**

- Callable identifier (e.g., C000F001)
- Callable name
- Source code lines for the callable
- Absolute start and end line numbers

**Procedure:**

1. Initialize an outcome map as an empty dict
2. For each line from start_line to end_line (inclusive):
   - If the line is non-executable (blank, comment only), skip it
   - If a line is executable, determine outcome count using the decision tree
     (Section 3.2)
   - Create list of outcome descriptions for this line
   - Add entry to outcome map: `line_number: [outcome_descriptions]`
3. Build the callable analysis structure

**Decision Process for Each Line:**

For each executable line, ask:

1. **Does this line contain a conditional construct?**
   - If statement: 2 outcomes (condition true, condition false)
   - Elif statement: 2 outcomes (condition true, continue to next)
   - Else statement: 1 outcome (executes)
   - Ternary expression: 2 outcomes (condition true, condition false)
   - Match/case statement: N outcomes (one per case)
   - Try statement: 1 outcome (enters try block)
   - Except handler: 1 outcome (exception caught)
   - Loop (for/while): 2 outcomes (0 iterations, ≥1 iterations)

2. **Does this line contain a comprehension or stream operation?**
   - List/dict/set comprehension with filter: 3 outcomes (empty source, all
     filtered, some pass)
   - Generator expression with filter: 3 outcomes (empty source, all filtered,
     some pass)
   - Stream with filter (Java): 3 outcomes (empty stream, all filtered, some
     pass)

3. **Does this line contain operations (e.g., method/function calls, collection
   comprehensions, python property access with `@property`) as parameters to
   other operations?**
   - Each operation in the parameter position creates **at least one EI**. The 
     exact number depends on analyzing that operation's possible outcomes.
     - Operations in this unit: enumerate actual outcome paths
     - Operations that are integration points: apply reasonable analysis (assume 
       success/failure at a minimum)
     - Variables or literals as parameters: do not create additional EIs
     - Each operation that is an **explicitly confirmed** integration point must
       be captured in Stage 3
   **See the relevant language guide for detailed guidance and examples.**

4. **Otherwise:** 1 outcome (line executes)

**Output Structure:**

```yaml
{
    "id": "C000F001",
    "name": "validate_typed_dict",
    "line_range": [13, 52],
    "outcome_map": {
        36: ["executes: allowed_env_keys = set(validation_type.__annotations__.keys())"],
        37: ["executes: bad_keys = set(mapping.keys()) - allowed_env_keys"],
        38: ["bad_keys truthy → raises ValueError", "bad_keys falsy → continues to line 40"],
        40: ["mapping.items() empty → bad_vals = []", "mapping has items, all isinstance checks pass → bad_vals = []",
            "mapping has items, some isinstance checks fail → bad_vals populated"],
        45: ["bad_vals truthy → raises ValueError", "bad_vals falsy → continues to line 52"],
        48: ["isinstance(value_type, type) true → expected = value_type.__name__",
            "isinstance(value_type, type) false → expected = joined tuple names"],
        52: ["executes: returns None"]
    },
    "total_outcomes": 11
}
```

**Verification Gates:**

- [ ] Every line between start_line and end_line has been considered
- [ ] Non-executable lines (comments, blanks) are omitted
- [ ] Lines with conditional constructs have multiple outcomes
- [ ] Lines without conditionals have single outcome
- [ ] Total outcome count matches sum of all outcome list lengths

**Stage Summary Output:**
```
✓ Stage 1 complete: C000F001 (validate_typed_dict)
  Lines analyzed: 36-52 (17 lines total)
  Executable lines: 8
  Total outcome paths: 11
```

### 3.2 Stage 2: EI ID Assignment

**Purpose:** Assign sequential EI IDs to each outcome path identified in
Stage 1.

**Input:**
- Outcome map from Stage 1 for current callable

**Procedure:**

1. Initialize EI counter at 0001 for this callable
2. For each line in outcome_map (in ascending line number order):
   - For each outcome in the line's outcome list:
     - Construct EI ID: callable_id + "E" + zero_padded_counter
     - Create mapping: EI_ID → (line_number, outcome_description)
     - Increment counter
3. Build the EI mapping structure

**EI ID Format:**

```
C000F001E0001  ← Unit function, first EI
C001M003E0012  ← Class method, 12th EI
```

**Grammar:**
- Unit function: C000 + F### + E####
- Class method: C### + M### + E####
- Nested function: C000 + F### + N### + E####
- Leading zeros are mandatory

**Output Structure:**
```yaml
{
    "callable_id": "C000F001",
    "ei_mappings": [
        {
            "id": "C000F001E0001",
            "line": 36,
            "outcome": "executes: allowed_env_keys = set(validation_type.__annotations__.keys())"
        },
        {
            "id": "C000F001E0002",
            "line": 37,
            "outcome": "executes: bad_keys = set(mapping.keys()) - allowed_env_keys"
        },
        {
            "id": "C000F001E0003",
            "line": 38,
            "outcome": "bad_keys truthy → raises ValueError"
        },
        {
            "id": "C000F001E0004",
            "line": 38,
            "outcome": "bad_keys falsy → continues to line 40"
        },
        # ... remaining EIs
    ],
    "total_eis": 11
}
```

**Verification Gates:**

- [ ] EI IDs are sequential with no gaps (E0001, E0002, E0003, ...)
- [ ] Total EI count equals total outcome count from Stage 1
- [ ] Every outcome from Stage 1 has exactly one EI ID
- [ ] EI ID format follows grammar rules

**Stage Summary Output:**
```
✓ Stage 2 complete: C000F001 (validate_typed_dict)
  EI IDs assigned: E0001 through E0011 (11 total)
  ID format verified: ✓
```

### 3.3 Stage 3: Integration Fact Enumeration

**Purpose:** Identify all integration points (interunit calls, extlib calls, and
boundary crossings) within the callable, linking them to their EI IDs.

**Input:**
- Source code for current callable
- EI mappings from Stage 2

**Procedure:**

1. Scan callable for integration candidates:
   - Imports from other project units (interunit)
   - Calls to other project units (interunit)
   - Calls to external libraries (potential boundary)
   - File I/O operations (boundary: filesystem)
   - Network operations (boundary: network)
   - Database operations (boundary: database)
   - Environment variable access (boundary: env)
   - Time/randomness operations (boundary: clock/randomness)
   - Subprocess calls (boundary: subprocess)

2. For each integration candidate:
   - Determine the EI ID where this integration occurs
   - Determine if interunit or boundary
   - Trace execution paths: what EI sequence leads here?
   - Create IntegrationFact with ID = "I" + EI_ID
   - Populate executionPaths field
   - Populate boundary or contract details if available

3. Categorize into interunit and boundaries lists

**Execution Path Tracing:**

For each integration, list all minimal EI sequences that lead to it:

- If integration is unconditional in the callable: `[[EI_ID]]`
- If integration is behind one condition: `[[condition_EI, integration_EI]]`
- If multiple paths can reach it: `[[path1_EIs], [path2_EIs]]`
- Minimal still means **thorough** and **complete**:
  - Enumerate ALL execution paths that can reach the integration point. For N
    independent conditionals before an integration, generate 2^N paths
    (Cartesian product of all conditional outcomes).

**Example:**

```python
def fetch_and_validate(raw: str) -> Config:
    if not raw:                    # E0001: empty check true
        raise ValueError("empty")  # E0002: raises
                                   # E0003: empty check false
    config = parse_config(raw)     # E0004: calls parse_config ← INTEGRATION
    if not config.valid:           # E0005: valid check true
        return None                # E0006: returns None
                                   # E0007: valid check false
    return config                  # E0008: returns config
```

Integration fact for parse_config call:

```yaml
id: IE0004  # "I" + E0004
target: parse_config
executionPaths:
  - [E0003, E0004]  # Only reachable if E0003 (not empty) executes
```

**Output Structure:**

```yaml
{
    "callable_id": "C000F001",
    "integration": {
        "interunit": [
            {
                "id": "IC000F001E0004",
                "target": "validate_typed_dict",
                "kind": "call",
                "signature": "validate_typed_dict('marker_env overrides', overrides_map, EnvironmentOverrides, str)",
                "executionPaths": [
                    ["C000F001E0004"]
                ]
            }
        ],
        "boundaries": [
            {
                "id": "IC000F001E0007",
                "target": "requests.get",
                "kind": "call",
                "boundary": {
                    "kind": "network",
                    "protocol": "http",
                    "operation": "read"
                },
                "executionPaths": [
                    ["C000F001E0003", "C000F001E0007"]
                ]
            }
        ]
    }
}
```

**Verification Gates:**

- [ ] Every integration ID starts with "I"
- [ ] Every integration ID references a valid EI ID from Stage 2
- [ ] Every executionPaths entry ends with the integration's own EI ID
- [ ] No integration has empty executionPaths (would indicate dead code)
- [ ] Interunit integrations target project code only
- [ ] Boundary integrations have boundary kind specified

**Stage Summary Output:**
```
✓ Stage 3 complete: C000F001 (validate_typed_dict)
  Interunit integrations: 1
  Boundary integrations: 0
  All execution paths verified: ✓
```

### 3.4 Stage 4: Document Generation

**Purpose:** Generate the three-document YAML ledger from all collected data.

**Input:**
- All outcome maps from Stage 1 (all callables)
- All EI mappings from Stage 2 (all callables)
- All integration facts from Stage 3 (all callables)
- Unit metadata (name, language, etc.)

**Procedure:**

1. **Generate Document 1 (Derived IDs):**
   - Populate unit metadata
   - Populate assigned.entries list with all callable IDs
   - Populate assigned.branches list with all EI IDs
   - Each branch entry includes: id, address, summary
   - Any code sourced from the unit should be taken verbatim
     - Enclosed in single quotes
     - Any internal single quotes are doubled (two single quotes, not replaced
       with a double quote)
     - Single lines with line breaks are joined into a single line

2. **Generate Document 2 (Ledger):**
   - Create unit entry (id: C000)
   - For each callable:
     - Create Entry with callable metadata
     - Populate params from signature
     - Create CallableSpec
     - Populate branches list with EiSpec for each EI
     - Populate the integration section if integrations exist
   - Nest entries hierarchically (e.g., classes contain methods)
     - Entries **must** be in the `children` list of their parent
     - Classes are always nested under the unit entry
     - Methods are always nested under their class
     - Functions of a module are always nested under their parent unit

3. **Generate Document 3 (Review):**
   - Initially empty findings list
   - Add findings for any anomalies noted during generation
   - Add findings for any UNREACHABLE EIs
   - Add findings for integrations with empty executionPaths
   - Add any suggestions for spec improvements that would reduce difficulties

4. Combine three documents with `---` separators

**Output:**

Multi-document YAML file with three documents.

**Verification Gates:**

- [ ] Document 1 has all callable IDs
- [ ] Document 1 has all EI IDs
- [ ] Document 2 unit.id is "C000"
- [ ] Document 2 has entry for every callable
- [ ] Document 2 has EiSpec for every EI ID from Document 1
- [ ] Document 3 exists (even if findings list is empty)
- [ ] YAML syntax is valid

**Stage Summary Output:**
```
✓ Stage 4 complete: Document generation
  Document 1: 47 entries, 203 EIs
  Document 2: Unit ledger with 12 callables
  Document 3: 2 findings noted
```

### 3.5 Stage 5: Schema Validation

**Purpose:** Validate the generated YAML against the JSON schema.

**Input:**
- Generated YAML from Stage 4
- JSON schema file (if provided)

**Procedure:**

1. Load YAML as multi-document structure
2. If schema provided:
   - Load schema
   - Validate YAML against schema
   - Collect all validation errors
3. If validation fails:
   - Report errors with locations
   - Enter repair loop (max 5 iterations):
     - Fix reported errors
     - Re-validate
     - If passes, exit loop
     - If max iterations reached, report failure

**Verification Gates:**

- [ ] YAML parses successfully
- [ ] Schema validation passes (if schema provided)
- [ ] All required fields present
- [ ] All enum values are valid
- [ ] All ID formats match patterns

**Stage Summary Output:**
```
✓ Stage 5 complete: Schema validation
  Validation: PASSED
  Warnings: 0
  Errors: 0
```

**If validation fails:**
```
✗ Stage 5 failed: Schema validation
  Errors found: 3
  [1] at .unit.children[2].callable.branches[5].id
      message: String does not match pattern
      expected: ^C000F\\d{3}E\\d{4}$
  Attempting repair (iteration 1/5)...
```

### 3.6 Final Deliverable

After all stages complete for all callables:

**Output the complete three-document YAML ledger.**

**Final Summary:**

```
═══════════════════════════════════════════════════════════════════
Unit Ledger Generation Complete
═══════════════════════════════════════════════════════════════════
Unit: compatibility.py (Python)
Callables analyzed: 12
Total EIs: 203
Interunit integrations: 8
Extlib integrationms: 5
Boundary integrations: 3
Findings: 2 (see Document 3)
Validation: PASSED
═══════════════════════════════════════════════════════════════════
```

## 4. Verification Gates

This section consolidates all verification checkpoints from the procedure.
Gates are organized by stage and must be satisfied before proceeding.

### 4.1 Purpose of Verification Gates

Verification gates serve multiple purposes:

- **Error prevention**: Catch mistakes early before they cascade
- **Determinism**: Ensure consistent results across runs
- **Auditability**: Provide clear checkpoints for review
- **Quality assurance**: Verify completeness and correctness

**Gate Types:**

- **Mandatory gates**: Must pass to proceed (marked with ✓/✗)
- **Warning gates**: Should pass but don't block progress (marked with ⚠)

### 4.2 Stage 1 Gates: Outcome Path Analysis

**Pre-stage validation:**

- [ ] Callable has valid ID format
- [ ] Start line number ≤ end line number
- [ ] Source code is available for specified line range

**Post-stage validation:**

- [ ] Every line between start_line and end_line has been considered
- [ ] Non-executable lines (comments, blanks) are omitted from outcome_map
- [ ] Lines with conditional constructs have 2+ outcomes
- [ ] Lines without conditionals have exactly 1 outcome
- [ ] Every outcome description is non-empty string
- [ ] Total outcome count = sum of all outcome list lengths
- [ ] outcome_map keys are in ascending order

**Warning conditions:**

- ⚠ Callable has 0 executable lines (empty function body)
- ⚠ Callable has 50+ outcome paths (very complex function)
- ⚠ Single line has 5+ outcomes (complex expression)

### 4.3 Stage 2 Gates: EI ID Assignment

**Pre-stage validation:**

- [ ] Outcome map from Stage 1 exists
- [ ] Outcome map has at least 1 entry
- [ ] Callable ID is valid format

**Post-stage validation:**

- [ ] EI IDs are sequential with no gaps (E0001, E0002, E0003, ...)
- [ ] First EI ID ends with E0001
- [ ] Total EI count equals total outcome count from Stage 1
- [ ] Every outcome from Stage 1 has exactly one EI ID
- [ ] EI ID format follows grammar: `callable_id + "E" + ####`
- [ ] No duplicate EI IDs exist
- [ ] Every EI mapping has non-empty outcome description

**ID Format Validation:**

Unit function EI:

```
Pattern: ^C000F\d{3}(?:N\d{3})*E\d{4}$
Example: C000F001E0001
```

Class method EI:

```
Pattern: ^C(?!000)\d{3}(?:N\d{3})*M\d{3}(?:N\d{3})*E\d{4}$
Example: C001M002E0015
```

### 4.4 Stage 3 Gates: Integration Fact Enumeration

**Pre-stage validation:**

- [ ] EI mappings from Stage 2 exist
- [ ] Source code is available

**Post-stage validation:**

- [ ] Every integration ID starts with "I"
- [ ] Every integration ID = "I" + valid EI ID from Stage 2
- [ ] Every integration has non-empty target string
- [ ] Every integration has at least one execution path
- [ ] Every executionPaths entry is non-empty list
- [ ] Every executionPaths entry ends with integration's own EI ID
- [ ] All EI IDs in executionPaths exist in Stage 2 mappings
- [ ] Interunit integrations target project code (not external libs)
- [ ] Boundary integrations have boundary.kind specified
- [ ] No integration has empty executionPaths

**Warning conditions:**

- ⚠ Integration has no condition specified (might be always executed)
- ⚠ Integration has 10+ execution paths (complex reachability)
- ⚠ Boundary integration missing protocol or system details

**Critical validation for executionPaths:**

For each integration with ID `Ixxxx`:

```python
# The corresponding EI ID must be `xxxx`
integration_ei = integration_id[1:]  # Strip "I" prefix

# Every path must end with this EI ID
for path in executionPaths:
    assert path[-1] == integration_ei, "Path must end with integration EI"
    
# Every EI in every path must exist
all_eis_in_paths = set(flatten(executionPaths))
assert all_eis_in_paths.issubset(stage2_ei_ids), "Unknown EI in path"
```

### 4.5 Stage 4 Gates: Document Generation

**Pre-stage validation:**

- [ ] All Stage 1 outcome maps available
- [ ] All Stage 2 EI mappings available
- [ ] All Stage 3 integration facts available
- [ ] Unit metadata is complete (name, language)

**Document 1 validation:**

- [ ] docKind is "derived-ids"
- [ ] schemaVersion is present and valid format
- [ ] unit.unitId is "C000"
- [ ] unit.name is non-empty
- [ ] unit.language is non-empty
- [ ] assigned.entries list contains all callable IDs
- [ ] assigned.branches list contains all EI IDs
- [ ] Every branch entry has: id, address, summary
- [ ] No duplicate entry IDs
- [ ] No duplicate branch IDs

**Document 2 validation:**

- [ ] docKind is "ledger"
- [ ] schemaVersion is present and valid format
- [ ] unit.id is "C000"
- [ ] unit.kind is "unit"
- [ ] unit.name matches Document 1 unit.name
- [ ] Every callable has Entry with matching ID from Document 1
- [ ] Every EI has EiSpec with matching ID from Document 1
- [ ] Every Entry with kind="callable" has callable field
- [ ] Every CallableSpec has branches list
- [ ] Every CallableSpec has params list (may be empty)
- [ ] Integration sections only appear in CallableSpec
- [ ] No orphaned EI IDs (every EI belongs to a callable)

**Document 3 validation:**

- [ ] docKind is "ledger-generation-review"
- [ ] schemaVersion is present and its format is valid
- [ ] unit.name matches Documents 1 and 2
- [ ] unit.language matches Documents 1 and 2
- [ ] unit.callablesAnalyzed reflects the number of callables analyzed
- [ ] unit.totalExeItems reflects the number of EIs enumerated
- [ ] unit.interunitIntegrations reflects the number of interunit integrations enumerated
- [ ] unit.boundaryIntegrations reflects the number of boundary integrations enumerated
- [ ] findings list exists (can be empty)
- [ ] Every finding has: severity, category, message
- [ ] Severity is one of: info, warn, error

**Cross-document validation:**

- [ ] All EI IDs in Document 1 appear in Document 2
- [ ] All callable IDs in Document 1 appear in Document 2
- [ ] Unit name is consistent across all 3 documents
- [ ] No EI ID appears in multiple callables

**YAML structure validation:**

- [ ] Valid YAML syntax (parseable)
- [ ] Exactly 3 documents separated by `---`
- [ ] No placeholder text (???, TBD, TODO, ...)
- [ ] All required fields present
- [ ] All empty optionals omitted

### 4.6 Stage 5 Gates: Schema Validation

**Pre-stage validation:**

- [ ] Generated YAML from Stage 4 exists
- [ ] YAML is parseable

**Schema validation (if schema provided):**

- [ ] YAML validates against schema
- [ ] All enum values are valid
- [ ] All ID patterns match regex
- [ ] All required fields present per schema
- [ ] No additional properties where prohibited
- [ ] Array minItems/maxItems satisfied
- [ ] String minLength/maxLength satisfied

**Common schema violations to check:**

- EI ID format mismatch
- Invalid enum value (kind, severity, category, etc.)
- Missing required field
- Empty string where non-empty required
- Empty array where minItems: 1

**Repair loop constraints:**

- [ ] Maximum 5 repair iterations
- [ ] Each iteration fixes at least one error
- [ ] No new errors introduced during repair
- [ ] If max iterations reached without success, report failure

### 4.7 Final Validation Checklist

Before delivering the ledger, verify:

**Completeness:**
- [ ] Every callable in the unit has been analyzed
- [ ] Every executable line has at least one EI
- [ ] All integration points have been identified

**Correctness:**
- [ ] All EI IDs follow grammar rules
- [ ] All integration IDs = "I" + valid EI ID
- [ ] All executionPaths are valid and reachable
- [ ] All cross-references resolve (no dangling IDs)

**Consistency:**
- [ ] Unit name identical across all 3 documents
- [ ] Language identical across Documents 1 and 3
- [ ] schemaVersion identical across all 3 documents
- [ ] EI count in Document 1 = EI count in Document 2

**Quality:**
- [ ] No placeholder text anywhere
- [ ] All outcome descriptions are meaningful
- [ ] All integration targets are fully qualified where possible
- [ ] All boundary integrations have kind specified

### 4.8 Validation Failure Handling

**If any mandatory gate fails:**

1. **Stop immediately** – Do not proceed to next stage
2. **Report failure** with specific gate that failed
3. **Provide context** (line number, callable ID, specific issue)
4. **Suggest fix** if deterministic
5. **Log to Document 3** as finding with severity: error

**Example failure report:**

```
✗ Stage 2 Gate Failed: EI ID sequence has gaps
  Callable: C001M003 (parse_response)
  Expected: E0001, E0002, E0003, E0004
  Found: E0001, E0002, E0004
  Issue: E0003 is missing
  Fix: Review Stage 1 outcome map for line-to-EI assignment
```

**If a warning gate triggers:**

1. **Continue processing** – Warnings don't block progress
2. **Log to Document 3** as finding with severity: warn
3. **Include it in the stage summary**

**Example warning report:**

```
⚠ Stage 1 Warning: High complexity detected
  Callable: C005F002 (resolve_dependencies)
  Outcome count: 73
  Note: Function has high cyclomatic complexity
  Recommendation: Consider refactoring for testability
```

## 5. Examples

This section provides detailed, worked examples demonstrating the procedure.
Examples use both Python and Java to illustrate language-agnostic principles,
and include common patterns and tricky cases.

### 5.1 Simple Example: Python Function with Basic Conditionals

**Source Code:**

```python
def classify_value(x: int) -> str:
    """Classify an integer as negative, zero, or positive."""
    if x < 0:
        return "negative"
    if x == 0:
        return "zero"
    return "positive"
```

**Stage 1: Outcome Path Analysis**

Line-by-line analysis:

```yaml
{
    "id": "C000F001",
    "name": "classify_value",
    "line_range": [1, 7],
    "outcome_map": {
        2: ["x < 0 true → returns 'negative'", "x < 0 false → continues to line 4"],
        4: ["x == 0 true → returns 'zero'", "x == 0 false → continues to line 6"],
        6: ["executes: returns 'positive'"]
    },
    "total_outcomes": 5
}
```

**Stage 2: EI ID Assignment**

```yaml
{
    "callable_id": "C000F001",
    "ei_mappings": [
        {"id": "C000F001E0001", "line": 2, "outcome": "x < 0 true → returns 'negative'"},
        {"id": "C000F001E0002", "line": 2, "outcome": "x < 0 false → continues to line 4"},
        {"id": "C000F001E0003", "line": 4, "outcome": "x == 0 true → returns 'zero'"},
        {"id": "C000F001E0004", "line": 4, "outcome": "x == 0 false → continues to line 6"},
        {"id": "C000F001E0005", "line": 6, "outcome": "executes: returns 'positive'"}
    ],
    "total_eis": 5
}
```

**Stage 3: Integration Facts**

No integrations (pure function, no external calls).

**Resulting YAML (Document 2 excerpt):**

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
        condition: 'else (x >= 0)'
        outcome: 'continues to next check'
      - id: C000F001E0003
        condition: 'if x == 0'
        outcome: 'returns "zero"'
      - id: C000F001E0004
        condition: 'else (x > 0)'
        outcome: 'continues to final return'
      - id: C000F001E0005
        condition: 'final return'
        outcome: 'returns "positive"'
```

### 5.2 Ternary Expression: Python

**Source Code:**

```python
def format_name(first: str, last: str | None) -> str:
    return f"{first} {last}" if last else first
```

**Stage 1: Outcome Path Analysis**

```yaml
{
    "id": "C000F002",
    "name": "format_name",
    "line_range": [1, 2],
    "outcome_map": {
        2: ["last is truthy → returns formatted string with last", "last is falsy → returns first only"]
    },
    "total_outcomes": 2
}
```

**Stage 2: EI ID Assignment**

```yaml
{
    "callable_id": "C000F002",
    "ei_mappings": [
        {"id": "C000F002E0001", "line": 2, "outcome": "last is truthy → returns formatted string with last"},
        {"id": "C000F002E0002", "line": 2, "outcome": "last is falsy → returns first only"}
    ],
    "total_eis": 2
}
```

**Resulting YAML (Document 2 excerpt):**

```yaml
- id: C000F002
  kind: callable
  name: format_name
  signature: 'format_name(first: str, last: str | None) -> str'
  callable:
    params:
      - name: first
        type:
          name: str
      - name: last
        type:
          name: 'str | None'
    returnType:
      name: str
    branches:
      - id: C000F002E0001
        condition: 'if last'
        outcome: 'returns f"{first} {last}"'
      - id: C000F002E0002
        condition: 'else'
        outcome: 'returns first'
```

### 5.3 List Comprehension with Filter: Python

**Source Code:**

```python
def get_valid_items(data: list[Item]) -> list[Item]:
    return [item for item in data if item.is_valid()]
# You can observe 3 distinct outcomes here:
# EI-1: data is empty → items = []
# EI-2: data not empty, all filtered out → items = []
# EI-3: data not empty, some pass filter → items populated
```

**Stage 1: Outcome Path Analysis**

```yaml
{
    "id": "C000F003",
    "name": "get_valid_items",
    "line_range": [1, 2],
    "outcome_map": {
        2: ["data is empty → returns []", "data has items but all filtered out → returns []", "data has items and some pass filter → returns populated list"]
    },
    "total_outcomes": 3
}
```

**Stage 2: EI ID Assignment**

```yaml
{
    "callable_id": "C000F003",
    "ei_mappings": [
        {"id": "C000F003E0001", "line": 2, "outcome": "data is empty → returns []"},
        {"id": "C000F003E0002", "line": 2, "outcome": "data has items but all filtered out → returns []"},
        {"id": "C000F003E0003", "line": 2, "outcome": "data has items and some pass filter → returns populated list"}
    ],
    "total_eis": 3
}
```

**Resulting YAML (Document 2 excerpt):**

```yaml
- id: C000F003
  kind: callable
  name: get_valid_items
  signature: 'get_valid_items(data: list[Item]) -> list[Item]'
  callable:
    params:
      - name: data
        type:
          name: 'list[Item]'
    returnType:
      name: 'list[Item]'
    branches:
      - id: C000F003E0001
        condition: 'data is empty'
        outcome: 'returns []'
      - id: C000F003E0002
        condition: 'data not empty but all items fail is_valid()'
        outcome: 'returns []'
      - id: C000F003E0003
        condition: 'data not empty and some items pass is_valid()'
        outcome: 'returns list with valid items'
```

### 5.4 For Loops

**Source Code:**
```python
def process_items(collection: list[Item]) -> None:
    for item in collection:
        item.process()
```

**Stage 1: Outcome Path Analysis**
```yaml
{
    "id": "C000F001",
    "name": "process_items",
    "line_range": [1, 2],
    "outcome_map": {
        2: ["collection is empty → loop body never executes (0 iterations)",
            "collection has items → loop body executes (≥1 iterations)"]
    },
    "total_outcomes": 2
}
```

**Stage 2: EI ID Assignment**
```yaml
{
    "callable_id": "C000F001",
    "ei_mappings": [
        {"id": "C000F001E0001", "line": 2, "outcome": "collection is empty → loop body never executes (0 iterations)"},
        {"id": "C000F001E0002", "line": 2, "outcome": "collection has items → loop body executes (≥1 iterations)"}
    ],
    "total_eis": 2
}
```

**Resulting YAML (Document 2 excerpt):**
```yaml
- id: C000F001
  kind: callable
  name: process_items
  signature: 'process_items(collection: list[Item]) -> None'
  callable:
    params:
      - name: collection
        type:
          name: 'list[Item]'
    returnType:
      name: None
    branches:
      - id: C000F001E0001
        condition: 'for item in collection (0 iterations)'
        outcome: 'loop body never executes'
      - id: C000F001E0002
        condition: 'for item in collection (≥1 iterations)'
        outcome: 'loop body executes, item.process() called for each item'
```

### 5.5 While Loops

**Source Code:**
```python
def wait_until_ready(service: Service) -> None:
    while not service.is_ready():
        time.sleep(0.1)
```

**Stage 1: Outcome Path Analysis**
```yaml
{
    "id": "C000F002",
    "name": "wait_until_ready",
    "line_range": [1, 3],
    "outcome_map": {
        2: ["service.is_ready() initially true → loop body never executes (0 iterations)",
            "service.is_ready() initially false → loop body executes (≥1 iterations)"]
    },
    "total_outcomes": 2
}
```

**Stage 2: EI ID Assignment**
```yaml
{
    "callable_id": "C000F002",
    "ei_mappings": [
        {"id": "C000F002E0001", "line": 2, "outcome": "service.is_ready() initially true → loop body never executes (0 iterations)"},
        {"id": "C000F002E0002", "line": 2, "outcome": "service.is_ready() initially false → loop body executes (≥1 iterations)"}
    ],
    "total_eis": 2
}
```

**Resulting YAML (Document 2 excerpt):**
```yaml
- id: C000F002
  kind: callable
  name: wait_until_ready
  signature: 'wait_until_ready(service: Service) -> None'
  callable:
    params:
      - name: service
        type:
          name: Service
    returnType:
      name: None
    branches:
      - id: C000F002E0001
        condition: 'while not service.is_ready() (condition initially false)'
        outcome: 'loop body never executes'
      - id: C000F002E0002
        condition: 'while not service.is_ready() (condition initially true)'
        outcome: 'loop body executes, time.sleep(0.1) called until ready'
```
### 5.6 Integration Example: Python with External Call

**Source Code:**

```python
def fetch_user_data(user_id: str) -> dict:
    if not user_id:
        raise ValueError("user_id required")
    response = requests.get(f"https://api.example.com/users/{user_id}")
    if response.status_code != 200:
        raise RuntimeError(f"API error: {response.status_code}")
    return response.json()
```

**Stage 1: Outcome Path Analysis**

```yaml
{
    "id": "C000F004",
    "name": "fetch_user_data",
    "line_range": [1, 7],
    "outcome_map": {
        2: ["user_id falsy → raises ValueError", "user_id truthy → continues to line 4"],
        4: ["executes: calls requests.get"],
        5: ["status_code != 200 → raises RuntimeError", "status_code == 200 → continues to line 7"],
        7: ["executes: returns response.json()"]
    },
    "total_outcomes": 6
}
```

**Stage 2: EI ID Assignment**

```yaml
{
    "callable_id": "C000F004",
    "ei_mappings": [
        {"id": "C000F004E0001", "line": 2, "outcome": "user_id falsy → raises ValueError"},
        {"id": "C000F004E0002", "line": 2, "outcome": "user_id truthy → continues to line 4"},
        {"id": "C000F004E0003", "line": 4, "outcome": "executes: calls requests.get"},
        {"id": "C000F004E0004", "line": 5, "outcome": "status_code != 200 → raises RuntimeError"},
        {"id": "C000F004E0005", "line": 5, "outcome": "status_code == 200 → continues to line 7"},
        {"id": "C000F004E0006", "line": 7, "outcome": "executes: returns response.json()"}
    ],
    "total_eis": 6
}
```

**Stage 3: Integration Facts**

```yaml
{
    "callable_id": "C000F004",
    "integration": {
        "boundaries": [
            {
                "id": "IC000F004E0004",
                "target": "requests.get",
                "kind": "call",
                "signature": "requests.get(f'https://api.example.com/users/{user_id}')",
                "boundary": {
                    "kind": "network",
                    "protocol": "http",
                    "system": "api.example.com",
                    "endpoint": "https://api.example.com",
                    "operation": "read",
                    "resource": "/users/{user_id}"
                },
                "executionPaths": [
                    ["C000F004E0002", "C000F004E0003"]
                ]
            }
        ]
    }
}
```

**Resulting YAML (Document 2 excerpt):**

```yaml
- id: C000F004
  kind: callable
  name: fetch_user_data
  signature: 'fetch_user_data(user_id: str) -> dict'
  callable:
    params:
      - name: user_id
        type:
          name: str
    returnType:
      name: dict
    branches:
      - id: C000F004E0001
        condition: 'if not user_id'
        outcome: 'raises ValueError("user_id required")'
      - id: C000F004E0002
        condition: 'else'
        outcome: 'continues to API call'
      - id: C000F004E0003
        condition: 'raise statement'
        outcome: 'raises ValueError'
      - id: C000F004E0004
        condition: 'requests.get call'
        outcome: 'executes HTTP GET request'
      - id: C000F004E0005
        condition: 'if response.status_code != 200'
        outcome: 'raises RuntimeError'
      - id: C000F004E0006
        condition: 'else'
        outcome: 'continues to return'
      - id: C000F004E0007
        condition: 'return statement'
        outcome: 'returns response.json()'
    integration:
      boundaries:
        - id: IC000F004E0004
          target: requests.get
          kind: call
          signature: 'requests.get(f"https://api.example.com/users/{user_id}")'
          boundary:
            kind: network
            protocol: http
            system: api.example.com
            endpoint: 'https://api.example.com'
            operation: read
            resource: '/users/{user_id}'
          executionPaths:
            - [C000F004E0002, C000F004E0004]
```

### 5.7 Java Example: Stream Operations

**Source Code:**

```java
public List<String> filterAndTransform(List<Item> items) {
    return items.stream()
        .filter(item -> item.isValid())
        .map(Item::getName)
        .collect(Collectors.toList());
}
```

**Stage 1: Outcome Path Analysis**

```yaml
{
    "id": "C001M001",
    "name": "filterAndTransform",
    "line_range": [1, 5],
    "outcome_map": {
        2: ["items.stream() is empty → returns []", "items.stream() has elements but all filtered out → returns []", "items.stream() has elements and some pass filter → returns populated list"]
    },
    "total_outcomes": 3
}
```

**Stage 2: EI ID Assignment**

```yaml
{
    "callable_id": "C001M001",
    "ei_mappings": [
        {"id": "C001M001E0001", "line": 2, "outcome": "items.stream() is empty → returns []"},
        {"id": "C001M001E0002", "line": 2, "outcome": "items.stream() has elements but all filtered out → returns []"},
        {"id": "C001M001E0003", "line": 2, "outcome": "items.stream() has elements and some pass filter → returns populated list"}
    ],
    "total_eis": 3
}
```

**Resulting YAML (Document 2 excerpt):**

```yaml
- id: C001M001
  kind: callable
  name: filterAndTransform
  signature: 'public List<String> filterAndTransform(List<Item> items)'
  callable:
    params:
      - name: items
        type:
          name: 'List<Item>'
    returnType:
      name: 'List<String>'
    branches:
      - id: C001M001E0001
        condition: 'items is empty'
        outcome: 'returns empty List<String>'
      - id: C001M001E0002
        condition: 'items not empty but all fail isValid()'
        outcome: 'returns empty List<String>'
      - id: C001M001E0003
        condition: 'items not empty and some pass isValid()'
        outcome: 'returns List<String> with mapped names'
```

### 5.8 Tricky Case: Nested Ternary (Python)

**Source Code:**

```python
def categorize(age: int, student: bool) -> str:
    return "child" if age < 18 else ("student" if student else "adult")
```

**Stage 1: Outcome Path Analysis**

This line has a nested ternary, creating 3 distinct outcome paths:

```yaml
{
    "id": "C000F005",
    "name": "categorize",
    "line_range": [1, 2],
    "outcome_map": {
        2: ["age < 18 true → returns 'child'", "age >= 18 and student true → returns 'student'", "age >= 18 and student false → returns 'adult'"]
    },
    "total_outcomes": 3
}
```

**Stage 2: EI ID Assignment**

```yaml
{
    "callable_id": "C000F005",
    "ei_mappings": [
        {"id": "C000F005E0001", "line": 2, "outcome": "age < 18 true → returns 'child'"},
        {"id": "C000F005E0002", "line": 2, "outcome": "age >= 18 and student true → returns 'student'"},
        {"id": "C000F005E0003", "line": 2, "outcome": "age >= 18 and student false → returns 'adult'"}
    ],
    "total_eis": 3
}
```

**Resulting YAML (Document 2 excerpt):**

```yaml
- id: C000F005
  kind: callable
  name: categorize
  signature: 'categorize(age: int, student: bool) -> str'
  callable:
    params:
      - name: age
        type:
          name: int
      - name: student
        type:
          name: bool
    returnType:
      name: str
    branches:
      - id: C000F005E0001
        condition: 'age < 18'
        outcome: 'returns "child"'
      - id: C000F005E0002
        condition: 'age >= 18 and student'
        outcome: 'returns "student"'
      - id: C000F005E0003
        condition: 'age >= 18 and not student'
        outcome: 'returns "adult"'
```

### 5.9 Tricky Cases

#### 5.9.1 Try-Except with Multiple Handlers (Python)

**Source Code:**

```python
def safe_parse(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None
```

**Stage 1: Outcome Path Analysis**

```yaml
{
    "id": "C000F006",
    "name": "safe_parse",
    "line_range": [1, 8],
    "outcome_map": {
        3: ["json.loads succeeds → returns dict", "json.loads raises JSONDecodeError → enters handler line 4",
            "json.loads raises other Exception → enters handler line 6"],
        5: ["executes: returns None"],
        7: ["executes: logger.error call"],
        8: ["executes: returns None"]
    },
    "total_outcomes": 6
}
```

**Stage 2: EI ID Assignment**

```yaml
{
    "callable_id": "C000F006",
    "ei_mappings": [
        {"id": "C000F006E0001", "line": 3, "outcome": "json.loads succeeds → returns dict"},
        {"id": "C000F006E0002", "line": 3, "outcome": "json.loads raises JSONDecodeError → enters handler line 4"},
        {"id": "C000F006E0003", "line": 3, "outcome": "json.loads raises other Exception → enters handler line 6"},
        {"id": "C000F006E0004", "line": 5, "outcome": "executes: returns None"},
        {"id": "C000F006E0005", "line": 7, "outcome": "executes: logger.error call"},
        {"id": "C000F006E0006", "line": 8, "outcome": "executes: returns None"}
    ],
    "total_eis": 6
}
```

**Stage 3: Integration Facts**

```yaml
{
    "callable_id": "C000F006",
    "integration": {
        "interunit": [
            {
                "id": "IC000F006E0001",
                "target": "json.loads",
                "kind": "call",
                "signature": "json.loads(raw)",
                "executionPaths": [
                    ["C000F006E0001"]
                ]
            },
            {
                "id": "IC000F006E0005",
                "target": "logger.error",
                "kind": "call",
                "signature": "logger.error(f'Unexpected error: {e}')",
                "executionPaths": [
                    ["C000F006E0003", "C000F006E0005"]
                ]
            }
        ]
    }
}
```

**Resulting YAML (Document 2 excerpt):**

```yaml
- id: C000F006
  kind: callable
  name: safe_parse
  signature: 'safe_parse(raw: str) -> dict | None'
  callable:
    params:
      - name: raw
        type:
          name: str
    returnType:
      name: 'dict | None'
    branches:
      - id: C000F006E0001
        condition: 'json.loads succeeds'
        outcome: 'returns parsed dict'
      - id: C000F006E0002
        condition: 'json.loads raises JSONDecodeError'
        outcome: 'enters except JSONDecodeError handler'
      - id: C000F006E0003
        condition: 'json.loads raises other Exception'
        outcome: 'enters except Exception handler'
      - id: C000F006E0004
        condition: 'return None from JSONDecodeError handler'
        outcome: 'returns None'
      - id: C000F006E0005
        condition: 'logger.error call'
        outcome: 'logs error message'
      - id: C000F006E0006
        condition: 'return None from Exception handler'
        outcome: 'returns None'
    integration:
      interunit:
        - id: IC000F006E0001
          target: json.loads
          kind: call
          signature: 'json.loads(raw)'
          executionPaths:
            - [C000F006E0001]
        - id: IC000F006E0005
          target: logger.error
          kind: call
          signature: 'logger.error(f"Unexpected error: {e}")'
          executionPaths:
            - [C000F006E0003, C000F006E0005]0F006E0004, C000F006E0007]
```

#### 5.9.2 Sequential Independent Conditionals (Python)

**Source code:**

```python
def to_mapping(self) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    if self.interpreter is not None:
        result["interpreter"] = self.interpreter.to_mapping()
    if self.abi is not None:
        result["abi"] = self.abi.to_mapping()
    if self.platform is not None:
        result["platform"] = self.platform.to_mapping()
    return result
```

**Stage 1: Outcome Path Analysis**

Unlike functions with early returns, this has sequential independent
conditionals where both true and false paths continue to the next line.

```yaml
{
    "id": "C010M001",
    "name": "to_mapping",
    "line_range": [1, 9],
    "outcome_map": {
        2: ["executes: result = {}"],
        3: ["self.interpreter is not None true → continues to line 4",
            "self.interpreter is not None false → continues to line 5"],
        4: ["executes: result['interpreter'] = self.interpreter.to_mapping()"],
        5: ["self.abi is not None true → continues to line 6",
            "self.abi is not None false → continues to line 7"],
        6: ["executes: result['abi'] = self.abi.to_mapping()"],
        7: ["self.platform is not None true → continues to line 8",
            "self.platform is not None false → continues to line 9"],
        8: ["executes: result['platform'] = self.platform.to_mapping()"],
        9: ["executes: returns result"]
    },
    "total_outcomes": 11
}
```

**Stage 2: EI ID Assignment**

```yaml
{
    "callable_id": "C010M001",
    "ei_mappings": [
        {"id": "C010M001E0001", "line": 2, "outcome": "executes: result = {}"},
        {"id": "C010M001E0002", "line": 3, "outcome": "self.interpreter is not None true → continues to line 4"},
        {"id": "C010M001E0003", "line": 3, "outcome": "self.interpreter is not None false → continues to line 5"},
        {"id": "C010M001E0004", "line": 4, "outcome": "executes: result['interpreter'] = self.interpreter.to_mapping()"},
        {"id": "C010M001E0005", "line": 5, "outcome": "self.abi is not None true → continues to line 6"},
        {"id": "C010M001E0006", "line": 5, "outcome": "self.abi is not None false → continues to line 7"},
        {"id": "C010M001E0007", "line": 6, "outcome": "executes: result['abi'] = self.abi.to_mapping()"},
        {"id": "C010M001E0008", "line": 7, "outcome": "self.platform is not None true → continues to line 8"},
        {"id": "C010M001E0009", "line": 7, "outcome": "self.platform is not None false → continues to line 9"},
        {"id": "C010M001E0010", "line": 8, "outcome": "executes: result['platform'] = self.platform.to_mapping()"},
        {"id": "C010M001E0011", "line": 9, "outcome": "executes: returns result"}
    ],
    "total_eis": 11
}
```

**Stage 3: Integration Facts**

For the integration at line 8 (self.platform.to_mapping()), all execution paths
that reach it must be enumerated. Since there are 2 independent conditionals
before it (interpreter and abi), there are 2² = 4 possible paths:

```yaml
{
    "callable_id": "C010M001",
    "integration": {
        "interunit": [
            {
                "id": "IC010M001E0010",
                "target": "PlatformConfig.to_mapping",
                "kind": "call",
                "signature": "self.platform.to_mapping()",
                "executionPaths": [
                    ["C010M001E0001", "C010M001E0002", "C010M001E0004", "C010M001E0005", "C010M001E0007", "C010M001E0008", "C010M001E0010"],  # T, T, T
                    ["C010M001E0001", "C010M001E0002", "C010M001E0004", "C010M001E0006", "C010M001E0008", "C010M001E0010"],  # T, F, T
                    ["C010M001E0001", "C010M001E0003", "C010M001E0005", "C010M001E0007", "C010M001E0008", "C010M001E0010"],  # F, T, T
                    ["C010M001E0001", "C010M001E0003", "C010M001E0006", "C010M001E0008", "C010M001E0010"]   # F, F, T
                ],
                "condition": "self.platform is not None",
                "notes": "All combinations of prior conditionals enumerated"
            }
        ]
    }
}
```

**Resulting YAML (Document 2 excerpt):**

```yaml
- id: C010M001
  kind: callable
  name: to_mapping
  signature: 'to_mapping(self) -> Mapping[str, Any]'
  callable:
    params:
      - name: self
        type:
          name: PlatformContext
    returnType:
      name: 'Mapping[str, Any]'
    branches:
      - id: C010M001E0001
        condition: 'executes'
        outcome: 'result = {}'
      - id: C010M001E0002
        condition: 'if self.interpreter is not None'
        outcome: 'continues to line 4'
      - id: C010M001E0003
        condition: 'else'
        outcome: 'continues to line 5'
      - id: C010M001E0004
        condition: 'executes'
        outcome: 'result["interpreter"] = self.interpreter.to_mapping()'
      - id: C010M001E0005
        condition: 'if self.abi is not None'
        outcome: 'continues to line 6'
      - id: C010M001E0006
        condition: 'else'
        outcome: 'continues to line 7'
      - id: C010M001E0007
        condition: 'executes'
        outcome: 'result["abi"] = self.abi.to_mapping()'
      - id: C010M001E0008
        condition: 'if self.platform is not None'
        outcome: 'continues to line 8'
      - id: C010M001E0009
        condition: 'else'
        outcome: 'continues to line 9'
      - id: C010M001E0010
        condition: 'executes'
        outcome: 'result["platform"] = self.platform.to_mapping()'
      - id: C010M001E0011
        condition: 'executes'
        outcome: 'returns result'
    integration:
      interunit:
        - id: IC010M001E0010
          target: PlatformConfig.to_mapping
          kind: call
          signature: 'self.platform.to_mapping()'
          executionPaths:
            - [C010M001E0001, C010M001E0002, C010M001E0004, C010M001E0005, C010M001E0007, C010M001E0008, C010M001E0010]
            - [C010M001E0001, C010M001E0002, C010M001E0004, C010M001E0006, C010M001E0008, C010M001E0010]
            - [C010M001E0001, C010M001E0003, C010M001E0005, C010M001E0007, C010M001E0008, C010M001E0010]
            - [C010M001E0001, C010M001E0003, C010M001E0006, C010M001E0008, C010M001E0010]
          condition: 'self.platform is not None'
          notes: 'All combinations of prior conditionals enumerated'
```

### 5.10 Complete Small Unit Example

This example shows all three documents for a tiny unit.

**Source Code (example.py):**

```python
def double_if_positive(x: int) -> int:
    if x > 0:
        return x * 2
    return x
```

**Document 1: Derived IDs**

```yaml
docKind: derived-ids
schemaVersion: "1.0.0"
unit:
  name: example
  language: python
  unitId: C000
assigned:
  entries:
    - id: C000F001
      kind: callable
      name: double_if_positive
      address: 'example::double_if_positive@L1'
  branches:
    - id: C000F001E0001
      address: 'example::double_if_positive@if x > 0@L2'
      summary: 'x > 0 true → returns x * 2'
    - id: C000F001E0002
      address: 'example::double_if_positive@if x > 0@L2'
      summary: 'x > 0 false → continues to line 4'
    - id: C000F001E0003
      address: 'example::double_if_positive@return@L4'
      summary: 'returns x'
```

**Document 2: Ledger**

```yaml
docKind: ledger
schemaVersion: "1.0.0"
unit:
  id: C000
  kind: unit
  name: example
  children:
    - id: C000F001
      kind: callable
      name: double_if_positive
      signature: 'double_if_positive(x: int) -> int'
      callable:
        params:
          - name: x
            type:
              name: int
        returnType:
          name: int
        branches:
          - id: C000F001E0001
            condition: 'if x > 0'
            outcome: 'returns x * 2'
          - id: C000F001E0002
            condition: 'else'
            outcome: 'continues to line 4'
          - id: C000F001E0003
            condition: 'return x'
            outcome: 'returns original value'
```

**Document 3: Review**

```yaml
docKind: ledger-generation-review
schemaVersion: "1.0.0"
unit:
  name: example
  language: python
  callablesAnalyzed: 7
  totalExeItems: 70
  interunitIntegrations: 6
  extlibIntegrations: 2
  boundaryIntegrations: 4
findings:
  - severity: info
    category: assumption
    message: 'No notable findings during ledger generation'
```
