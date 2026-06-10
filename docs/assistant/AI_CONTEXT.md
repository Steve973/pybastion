# AI_CONTEXT.md

## Purpose

PyBastion analyzes Python projects and produces structured artifacts that make executable behavior, integration seams,
and feature-level flows explicit for human review and AI-assisted test generation.

The goal is not to generate perfect finished tests automatically. The goal is to create accurate, inspectable analysis
artifacts that let a developer or coding agent reason about unit behavior, integration paths, feature flows, and test
obligations without rediscovering the whole project from raw source every time.

## Source Documents

Primary design documents:

- `docs/unit/branch_ids.md`
- `docs/unit/unit-analysis-pipeline.md`
- `docs/integration/integration-analysis-pipeline.md`
- `docs/integration/project-analysis-markers.md`

Important filename note:

- The branch ID file is `branch_ids.md`, not `branch-ids.md`.

## Core Architecture

PyBastion has two major analysis layers:

1. Unit analysis
2. Integration analysis

Unit analysis is source-facing. It analyzes individual Python source units and produces unit-scoped inventories.

Integration analysis is project-facing. It consumes unit inventories and produces project-level integration analysis
artifacts, integration test specifications, split spec files, and feature-flow analysis outputs.

Integration analysis should not casually rederive source-level facts from raw source. The unit inventory is the
authoritative input for integration graph construction and specification generation.

## Unit Analysis Contract

The unit pipeline analyzes Python source units and produces structured unit inventories.

A unit inventory describes:

- units
- classes
- functions
- methods
- nested classes
- nested functions
- unit-scoped bindings
- callable-owned execution items
- integration candidates
- callable-local paths to integration points
- supporting type, binding, and resolution context
- uncertainty or unresolved analysis findings

The inventory is not a traditional control-flow graph. It is a structured, unit-scoped artifact that preserves enough
source structure and executable behavior for downstream analysis, graph construction, and test generation.

## Unit Analysis Stages

Current conceptual stages:

1. Preliminary unit analysis
2. Execution item analysis
3. Inventory generation

The implementation may combine or reorganize stages internally, but output artifacts must preserve the semantics of
these conceptual stages.

### Stage 1: Preliminary Unit Analysis

Stage 1 identifies durable unit structure.

It records:

- unit metadata
- classes
- functions
- methods
- nested entries
- unit-scoped bindings
- deterministic IDs
- ownership
- ordering
- source paths
- fully qualified names
- structural metadata

Stage 1 output goes under:

`dist/pybastion/inspect/`

The important Stage 1 artifact is the unit index.

### Stage 2: Execution Item Analysis

Stage 2 enumerates execution items.

Execution Items, or EIs, are callable-owned execution facts. They represent distinct executable outcomes inside a
function or method.

Examples of EIs include:

- callable entry
- statement execution
- function or method invocation
- condition true path
- condition false path
- loop zero-iteration path
- loop one-or-more-iterations path
- return
- raise
- break
- continue
- synthetic implicit return

An EI is not the same thing as a test case.

An EI should represent distinct executable behavior or outcome structure. Concrete test inputs, equivalence classes, and
boundary values are test-generation concerns, not separate EIs by themselves.

Stage 2 output goes under:

`dist/pybastion/eis/`

### Stage 3: Inventory Generation

Stage 3 generates the final unit inventory.

It organizes and normalizes structural facts and EI facts into a stable, inspectable, machine-consumable unit artifact.

Stage 3 should mostly organize facts that already exist. It should not quietly invent semantic facts. If analysis is
uncertain, the uncertainty should remain visible.

Stage 3 output goes under:

`dist/pybastion/inventory/`

## Core Modeling Rules

### Observed Facts vs Derived Facts

Preserve the distinction between observed source facts and derived analysis facts.

Observed facts include:

- source structure
- names
- signatures
- decorators
- source spans
- expressions
- syntactic operation targets
- unit-scoped bindings

Derived facts include:

- resolved targets
- inferred receiver types
- integration classification
- execution paths
- feasibility results
- hierarchy metadata
- summary counts

A quiet guess is worse than an honest gap.

### Inventory, Not Ledger

Use the current term `inventory`.

Older design language may refer to a `ledger`, but the current active artifact is the unit inventory.

### Callable Ownership

Execution items belong to callables.

A unit may own entries and bindings, but executable behavior is owned by the callable where it occurs.

Rules:

- a unit owns entries and bindings
- a class owns methods and nested entries
- a callable owns its execution items
- a callable owns integration candidates that occur inside it
- unit-scoped bindings do not own execution items
- bindings may be referenced by callable-owned execution items

Example:

`_SHA256_RE` at module scope is a binding.

`_SHA256_RE.match(value)` inside a function is callable-owned executable behavior and may be represented by an EI in
that function.

### Bindings Are Not Execution Owners

Unit-scoped bindings are useful for name, type, value, receiver, alias, or integration resolution.

Bindings are not callable entries.

Bindings do not receive `E####` execution item IDs.

Bindings do not participate in uncalled callable analysis.

Binding IDs must never be used as callable IDs or EI IDs.

## Flattened Flow Model

PyBastion uses a flattened execution flow model.

Loops are not modeled as unbounded runtime cycles in the inventory.

Loops are collapsed into representative finite paths such as:

- zero iterations
- one or more iterations
- disruptive paths, when present

The point is to preserve useful execution structure for reachability, integration path analysis, and test generation
without pretending to simulate arbitrary runtime repetition.

Important rule:

Downstream integration graph construction must preserve the flattened unit-analysis model. It must not restore source
loops into cyclic runtime-style CFGs unless a future graph mode explicitly chooses to do that.

A loop containing an integration call means the inventory should show representative paths that can or cannot reach that
integration EI. It does not mean the integration graph should model every possible iteration count.

## Successor Resolution

Each EI outcome should describe where execution continues when known.

When the successor EI is known, use a concrete successor reference.

When the successor cannot be known immediately, preserve a target hint.

Target hints may include:

- line number of the next statement
- source span context
- conditional polarity
- structural context
- line numbers or EI IDs to skip
- expression context

Uncertainty should remain visible until resolved.

Do not invent confident successors when evidence is incomplete.

## Implicit Returns

If normal execution can fall through the end of a callable, include a synthetic implicit return EI.

This gives fallthrough execution a real terminal landing point.

A callable that can fall through should not appear to have no terminal flow.

## Branch IDs and Deterministic Numbering

Branch and execution item IDs are a core contract.

Sloppy IDs weaken the value of the inventory.

IDs must be deterministic, fully qualified, and stable under unrelated edits.

### Abbreviations

- `Uxxxxxxxxxx` = unit ID, SHA-256 first 10 characters of fully qualified file name
- `Cxxx` = class ID, starting at `C001`
- `Fxxx` = unit-level function ID, starting at `F001`
- `Mxxx` = method ID, starting at `M001`
- `Exxxx` = execution item ID, starting at `E0001`
- `Bxxx` = unit-scoped binding ID, starting at `B001`

### Fully Qualified IDs Only

Do not use unscoped IDs.

Correct examples:

- `U1234567890_F001_E0001`
- `U1234567890_C001_M001_E0001`
- `U1234567890_C001.C002_M001.F001_E0001`

Incorrect examples:

- `F001_E0001`
- `M001_E0001`
- `B001_E0001`

### Unit IDs

A unit ID is:

`U` + first 10 characters of SHA-256 of the fully qualified file name

All IDs inside the unit are prefixed with the unit ID.

### Class IDs

Classes are numbered starting at `C001`.

They are assigned in file order, top to bottom.

Nested classes also receive class IDs.

### Function IDs

Unit-level functions are numbered starting at `F001` within the unit.

They are assigned in file order, top to bottom.

### Method IDs

Methods are numbered starting at `M001` within each class.

They are assigned in order within the class definition, top to bottom.

### Execution Item IDs

Execution items are numbered starting at `E0001` within each function or method.

They are assigned in order of appearance in the code.

### Binding IDs

Unit-scoped bindings are numbered starting at `B001` within the unit.

Binding ID format:

`<unit_id>_<binding_id>`

Example:

`U1234567890_B001`

Bindings are not callables and do not own EIs.

### Nested IDs

Use `.` to append a nested item to its parent designator.

Examples:

- nested class under class: `Uxxxxxxxxxx_C001.C002`
- nested function under unit function: `Uxxxxxxxxxx_F001.F002`
- nested function under method: `Uxxxxxxxxxx_C001_M001.F001`
- EI under nested function: `Uxxxxxxxxxx_C001_M001.F001_E0001`

Use nested IDs only for named nested defs or classes that are referenced as objects or meaningfully testable in
isolation.

If a nested def is purely an implementation detail and is not directly targeted, treat its control flow as branches of
the parent and do not assign a nested designator.

### ID Composition Order

Callable/EI IDs are composed in this order:

1. unit or class scope
2. nested class scope, if needed
3. callable
4. nested callable, if needed
5. execution item

Unit-level callable EI:

`<unit_id>_<function_id><.nested_function_id>_<execution_item_id>`

Class method EI:

`<unit_id>_<class_id><.nested_class_id>_<method_id><.nested_function_id>_<execution_item_id>`

Leading zeros are mandatory.

Use `_` between ID segments.

Use `.` for nesting.

## Integration Analysis Contract

The integration pipeline consumes unit inventories and produces integration test specifications.

It builds a project-level view of how one unit reaches another.

It answers questions such as:

- which source units call which target units
- which execution items participate in those calls
- which callable-local paths can reach each integration point
- which seams should become integration test specifications
- which smaller spec files should be handed to a developer or coding agent

The integration pipeline should work from structural and behavioral facts emitted by unit analysis.

It should preserve known facts, derive project-level relationships, and keep uncertainty visible when resolution is
incomplete.

## Integration Analysis Stages

Current conceptual stages:

1. Build the EI call graph
2. Generate integration test specifications
3. Split specifications
4. Feature flow analysis

The graph checker may optionally run after Stage 1.

### Stage 1: Build the EI Call Graph

Stage 1 builds a project-level graph from unit inventories.

The graph is centered on execution items, not only callables.

The graph should preserve:

- source unit
- source callable
- source execution item
- integration target
- target unit, when resolved
- target callable, when resolved
- paths that reach the integration point
- metadata needed by later stages

Default output:

`dist/pybastion/integration-output/stage1-ei-cfg.pkl`

The pickle format is an implementation choice, not necessarily a long-term public artifact contract.

### Optional Graph Checker

The graph checker validates that the Stage 1 graph lines up with source unit inventories.

It should detect problems such as:

- graph nodes that do not correspond to inventory facts
- missing callable references
- missing EI references
- broken source-to-target relationships
- unexpected graph structure
- inventory facts not represented correctly in the graph

Default output:

`dist/pybastion/integration-output/inventory-graph-report.yaml`

The graph checker is diagnostic. It should report uncertainty or mismatches visibly rather than silently repairing them.

### Stage 2: Generate Integration Test Specifications

Stage 2 consumes the EI call graph and generates integration test specifications.

A specification is not a finished test file.

A specification should give enough context to decide:

- fixture setup
- mocks or fakes
- action
- assertions
- source unit
- target unit or callable
- source EI
- path or paths to the seam
- behavior to exercise
- remaining uncertainty

Default output:

`dist/pybastion/integration-output/stage2-integration-test-specs.yaml`

### Stage 3: Split Specifications

Stage 3 splits integration specs into smaller files.

Split output is grouped by source unit and target unit pair.

Default output directory:

`dist/pybastion/integration-output/specs/`

The split is practical. Smaller files are easier to inspect and safer to hand to an agent for focused test generation.

### Stage 4: Feature Flow Analysis

Stage 4 turns feature-flow markers into testable integration flow cases.

It uses marker information, EI graph traversal, branch-aware state, convergence points, and feature end markers to
produce distinct feature-flow test obligations.

Feature-flow tracing is not a shortest-path problem.

A feature can contain multiple marked branch points. Each branch point can expand the number of testable feature-flow
cases.

Feature convergence allows shared downstream traversal but does not erase distinct case identity.

## Project Analysis Markers

Project analysis markers are comment-based metadata used to control integration flow tracing, feature flow
identification, and call graph interpretation.

Markers are captured during unit analysis along with decorators.

Their usage primarily pertains to integration analysis.

General marker syntax:

`:: MarkerName | field=value | field=value`

Rules:

- marker name is case-sensitive
- fields are pipe-delimited
- field values use `key=value`
- whitespace around delimiters is ignored
- quote marks around values are stripped during parsing
- markers may appear in supported language comment styles

## Marker Placement

Markers may appear:

- immediately before a function or method signature
- immediately before the statement they apply to
- inside method-level docstrings when they apply to the whole method

Feature-flow statement markers apply to the next statement, not to the surrounding callable as a whole.

## Operation Markers

Operation markers apply at function or method scope.

They denote mechanical or utility operations that should be excluded from integration flow tracing or represented with
mocks or fixtures.

### MechanicalOperation

Use for pure data transformations without business decisions.

Examples:

- serialization
- deserialization
- validation without business rules
- formatting
- conversion
- normalization
- data transformation
- presentation
- construction

### UtilityOperation

Use for infrastructure or plumbing operations without business logic.

Examples:

- logging
- caching
- config
- observability
- audit
- data structure operations
- registry operations

Do not mark business logic as mechanical or utility.

Business logic includes:

- domain decisions
- orchestration
- meaningful state changes
- rules that should be traced in integration tests

## Reachability Markers

Reachability markers explain how callables may be invoked even when they do not appear to have normal direct callers in
the project graph.

They do not exclude a callable from flow tracing by default.

### ExternalApiMethod

Marks a callable as externally reachable through an API boundary.

Use for public entry points invoked by clients, CLIs, services, plugins, or other external systems.

### FrameworkCallback

Marks a callable as invoked by framework lifecycle, hook, event, route, callback, or extension-point behavior.

### CalledThroughAbstraction

Marks a callable as reached through an abstraction, interface, contract, registry, strategy, plugin mechanism, or
dynamic dispatch path rather than through direct source-level calls.

## Feature Flow Markers

Feature-flow markers are statement-scope markers used to guide feature-level integration tracing over the EI graph.

They identify:

- feature starts
- waypoints
- branch points
- convergence points
- feature ends
- conditional feature ends

All markers in a feature flow are correlated by the `name` field.

Branches within a feature flow are correlated by the `branch` field.

### FeatureStart

Marks the statement where a feature flow begins.

Rules:

- each feature flow must have exactly one `FeatureStart`
- must have at least one corresponding `FeatureEnd` or `FeatureEndConditional`
- applies to the immediately following statement

### FeatureTrace

Marks a waypoint along a feature flow path.

Use to:

- identify intermediate steps
- disambiguate complex flows
- mark where a named branch continues

### FeatureBranch

Marks a control statement where a feature flow splits into named paths.

A control statement may have multiple feature-relevant outcomes.

Each `FeatureBranch` marker describes one named branch mapping for the marked control statement.

Branch lifecycle:

1. `FeatureBranch` names a feature branch at a control statement
2. `control_polarity`, when present, maps that branch to a modeled control outcome
3. the branch may continue through `FeatureTrace` markers with the same branch name
4. the branch must eventually reach a feature end, conditional end, or convergence point

Feature branches multiply testable feature-flow cases.

### FeatureConverge

Marks a point where separate feature branches can resume shared traversal.

Convergence does not reduce the number of test cases.

Convergence identifies where branches share a downstream segment while preserving the case identity that reached the
convergence point.

### FeatureEnd

Marks a normal feature completion point.

### FeatureEndConditional

Marks a conditional feature completion point.

A conditional end may terminate one branch case while other cases continue to a normal feature end.

## Feature Flow Case Rules

Feature-flow tracing should preserve branch-aware case identity.

Important rules:

- `FeatureStart` begins the feature root `main` case
- each branch group multiplies active cases that reach it
- nested branch groups extend only the current branch lineage
- nested branch groups do not multiply unrelated branches that do not reach them
- convergence joins execution location but preserves case identity
- converged branches share downstream traversal
- feature-flow paths are assembled from reusable path segments
- each segment records a local EI path and absolute branch lineage
- final feature-flow cases are assembled from main, branch, convergence, shared-tail, and end segments

Example:

`main -> branch group: a | b -> converge -> branch group: c | d -> converge -> end`

Produces:

- `main -> a -> c -> end`
- `main -> a -> d -> end`
- `main -> b -> c -> end`
- `main -> b -> d -> end`

## Artifact Locations

Unit analysis outputs:

- `dist/pybastion/inspect/`
- `dist/pybastion/eis/`
- `dist/pybastion/inventory/`
- `dist/pybastion/logs/`

Integration analysis outputs:

- `dist/pybastion/integration-output/stage1-ei-cfg.pkl`
- `dist/pybastion/integration-output/inventory-graph-report.yaml`
- `dist/pybastion/integration-output/stage2-integration-test-specs.yaml`
- `dist/pybastion/integration-output/specs/`

## Validation Expectations

Artifacts should preserve checkable invariants.

Examples:

- every graph node should correspond to known inventory information
- every source EI should belong to its containing callable
- every source callable should belong to its containing unit
- every resolved target unit should exist in the inventory set
- every integration spec should refer to a valid source unit
- every split spec should preserve the source and target relationship
- graph checker findings should be visible
- unresolved or uncertain targets should remain visible

## What PyBastion Does Not Do

PyBastion does not guarantee full test coverage by itself.

It does not replace:

- developer review
- fixture design
- assertion design
- domain knowledge
- runtime validation
- coverage measurement
- mutation testing
- normal test maintenance

PyBastion provides structured analysis and test guidance. Real tests still need meaningful setup, inputs, assertions,
and correctness criteria.

## Assistant Operating Rules

When helping with PyBastion coding tasks:

1. Treat this document as stable context, not active task state.
2. Ask for or use a separate `AI_STATE.md` for current bugs, hypotheses, failing tests, and current implementation
   state.
3. Prefer current pasted files over memory or prior conversation.
4. If the user says a file version is current, ignore earlier versions.
5. Do not reason from stale implementation details.
6. Do not invent schema changes.
7. Do not silently change ID grammar.
8. Do not confuse bindings with callable-owned execution items.
9. Do not treat integration analysis as raw-source rediscovery.
10. Do not restore flattened loop modeling into unbounded runtime CFG behavior.
11. Preserve uncertainty instead of making quiet guesses.
12. Prefer targeted fixes over broad rewrites.
13. Before coding, restate the current task, relevant invariant, affected files, and intended change when the task is
    complex.
14. When giving code changes, provide full replacement blocks or full files unless the user asks for a diff.
15. When the conversation gets noisy, produce a compact handoff checkpoint instead of continuing from polluted context.

## Things Not To Assume

Do not assume:

- nested source structure implies nested execution IDs unless the branch ID rules require it
- every source statement maps to exactly one EI
- every branch is a test case
- every syntactic call is a useful integration test obligation
- all unresolved calls should be guessed
- bindings own executable behavior
- integration specs are finished tests
- feature convergence collapses test case identity
- feature tracing is a shortest-path problem
- loop bodies should create unbounded graph cycles
- project markers apply to an entire callable when they are statement-scope markers
