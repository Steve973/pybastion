# Unit Analysis Pipeline

The PyBastion unit analysis pipeline analyzes Python source units and produces
structured unit inventories for testing and downstream analysis.

A unit inventory describes the structure and executable behavior of a unit. It
captures classes, functions, methods, unit-scoped bindings, callable-owned
execution items, integration candidates, and the execution paths that reach
those integration points.

The purpose of the inventory is not merely to summarize source code. It is to
make executable behavior and integration surfaces explicit. Coverage tools can
tell you what tests executed. The inventory gives you a structured checklist of
what behavior exists and which integration seams tests should consider.

This matters especially for agent-assisted test generation. Instead of asking
an agent to infer everything from raw source text, PyBastion gives the agent a
structured model of the unit: what exists, where execution can go, what calls
cross meaningful boundaries, and what uncertainty remains.

The pipeline is implementation-oriented, but the conceptual boundaries are still
important. The implementation may combine or reorganize stages internally, but
the produced artifacts should preserve the semantics described here.

## Table of Contents

- [Overview](#overview)
- [Core Principles](#core-principles)
  - [Observed Facts and Derived Facts](#observed-facts-and-derived-facts)
  - [Inventory, Not Ledger](#inventory-not-ledger)
  - [Callable Ownership](#callable-ownership)
- [Stage 1: Preliminary Unit Analysis](#stage-1-preliminary-unit-analysis)
  - [Entries](#entries)
  - [Bindings](#bindings)
  - [Deterministic Ordering](#deterministic-ordering)
  - [Stage 1 Output](#stage-1-output)
- [Stage 2: Execution Item Analysis](#stage-2-execution-item-analysis)
  - [What Counts as an Execution Item](#what-counts-as-an-execution-item)
  - [Flattened Flow Representation](#flattened-flow-representation)
  - [Collapsed Loops and the Call Graph](#collapsed-loops-and-the-call-graph)
  - [Successor Resolution](#successor-resolution)
  - [Implicit Returns](#implicit-returns)
  - [Outcome Paths vs. Test Cases](#outcome-paths-vs-test-cases)
  - [Stage 2 Output](#stage-2-output)
- [Stage 3: Inventory Generation](#stage-3-inventory-generation)
  - [Integration Candidate Analysis](#integration-candidate-analysis)
  - [Integration Categories](#integration-categories)
  - [Paths to Integration Points](#paths-to-integration-points)
  - [Feasibility Filtering](#feasibility-filtering)
  - [Inventory Structure](#inventory-structure)
  - [Stage 3 Output](#stage-3-output)
- [Relationship to Integration Analysis](#relationship-to-integration-analysis)
- [Validation Expectations](#validation-expectations)
- [What This Pipeline Does Not Do](#what-this-pipeline-does-not-do)

## Overview

The unit pipeline is the source-facing side of PyBastion.

It scans Python source units, identifies their durable structural elements,
enumerates callable-local execution behavior, and packages the result as
unit-scoped inventory artifacts.

The current pipeline is organized around three main stages:

1. Inspect units.
2. Enumerate execution items.
3. Generate callable and inventory artifacts.

The readiness scanner is an optional preflight step. It is not one of the
three inventory stages. It looks for source patterns that may reduce analysis
quality, such as unresolved receiver types, dynamic dispatch, broad `Any`
annotations, and opaque branch conditions.

The normal output goes under:

```text
dist/pybastion/
  inspect/
  eis/
  inventory/
  logs/
```

The inventory artifacts are consumed by the integration pipeline and can also be
read directly by developers or coding agents.

## Core Principles

### Observed Facts and Derived Facts

The inventory should preserve a clear distinction between facts observed
directly from source and facts derived through analysis.

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

Derived facts are useful, but they should not hide the source facts that support
them. If analysis depends on inference, partial evidence, or external knowledge,
that uncertainty should remain visible in the inventory or related findings.

A quiet guess is worse than an honest gap.

### Inventory, Not Ledger

Older versions of the design used the term "ledger" for the generated unit
artifact. The current model uses "inventory."

That is not just a naming cleanup. The current inventory is the active unit
artifact. It is a unit-scoped representation of structure, callable behavior,
execution items, integration candidates, and supporting context.

The inventory is not a control flow graph. It preserves enough structural and
behavioral information that downstream tooling can derive graphs, inspect
integration seams, and generate tests without reconstructing the whole unit from
source.

### Callable Ownership

Execution items belong to callables.

This is one of the most important modeling rules in the unit pipeline. A unit
may contain many structural items and many bindings, but executable behavior is
owned by the callable in which it occurs.

That means:

- a unit owns entries and bindings
- a class owns methods and nested entries
- a callable owns its execution items
- a callable owns the integration candidates that occur inside it
- unit-scoped bindings do not own execution items

A binding can be used by a callable, but the use is executable behavior in the
callable. The binding itself is not a callable and is not an execution owner.

For example:

```python
_SHA256_RE: re.Pattern[str] = re.compile(...)
```

is a unit-scoped binding.

A later use inside a callable:

```python
_SHA256_RE.match(value)
```

is executable behavior. That use may be represented by an execution item owned
by the callable where the call occurs.

This distinction prevents module-level names from being confused with
callable-local execution flow.

## Stage 1: Preliminary Unit Analysis

Stage 1 scans the source tree and identifies the durable structure of each
unit.

This includes items such as:

- units
- classes
- functions
- methods
- nested classes
- nested functions
- unit-scoped bindings
- field metadata, when available

The goal is not to understand every execution path yet. The goal is to build
the structural context that later stages need.

Stage 1 assigns deterministic IDs and records enough metadata for later stages
to resolve ownership, hierarchy, source location, and ordering.

### Entries

Entries are structural items that may appear in the inventory hierarchy.

Examples include:

- classes
- functions
- methods
- nested classes
- nested functions

Entries can own other entries. For example, a class can own methods, and a
function can own nested functions.

Callable entries eventually receive callable-owned analysis details, including
execution items and integration candidates.

### Bindings

Bindings are names bound at unit scope.

Examples include:

- module-level assignments
- annotated assignments
- constants
- aliases
- TypeVars
- compiled regexes
- maps of callable values

Bindings are useful for name resolution, type resolution, receiver resolution,
alias resolution, and integration analysis. They are not structural callable
entries, and they do not own execution items.

This matters because a unit-scoped binding can be relevant to execution without
being executable flow by itself.

### Deterministic Ordering

Ordinal assignment is a semantic choice, not just bookkeeping.

The unit index should preserve deterministic sibling ordering. Downstream
stages may rely on that ordering when reconstructing hierarchy, comparing
artifacts, generating stable IDs, or producing deterministic output.

The same principle applies to ID generation. ID rules are part of the analysis
contract. Changing them changes downstream artifacts.

### Stage 1 Output

Stage 1 produces inspection artifacts under:

```text
dist/pybastion/inspect/
```

The important output is the unit index. It records unit-level structural facts,
including entries, bindings, IDs, ownership, ordering, source paths, fully
qualified names, and related metadata.

A later stage should not have to rediscover the basic unit hierarchy from raw
source. It should be able to consume the unit index as the structural base for
deeper analysis.

## Stage 2: Execution Item Analysis

Stage 2 enumerates execution items.

Execution Items, or EIs, are callable-owned execution facts. They represent
distinct executable outcomes inside a function or method.

Execution items are not necessarily the same size as source statements. A
single source statement can contain multiple execution items. EIs are also lower
level than the usual idea of a branch. Branching constructs produce EIs, but
not all EIs are branches.

The goal is to describe callable-local execution behavior in a form that is
specific enough for path analysis and test generation, but not so literal that
it becomes a runtime interpreter.

### What Counts as an Execution Item

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

An EI should represent a distinct executable outcome. It should not represent
every possible input value.

For example, this expression has one executable operation:

```python
x - 2
```

A test suite may still need several values for `x`, but those values are test
case choices. They are not separate execution items.

### Flattened Flow Representation

The EI artifact does not represent loops as open-ended runtime cycles.

Instead, loops are flattened into a small set of representative execution paths.
Generally, this means modeling cases such as:

- zero iterations
- one or more iterations
- disruptive paths, when present

This flattening is intentional. The inventory is not trying to simulate runtime
execution forever. It preserves the useful callable-local flow structure needed
for reachability, integration path analysis, and downstream test generation.

A construct like `continue` may be represented as a disruptive or continuation
outcome in the flattened structure. The important point is that it participates
in the representative loop flow without forcing the analysis to model an
unbounded cycle.

### Collapsed Loops and the Call Graph

This is the part that is easy to get wrong.

The unit inventory is not supposed to emit a traditional cyclic loop structure
and then expect later stages to reason over arbitrary runtime repetition. Loops
are collapsed into representative paths before downstream graph analysis uses
the inventory.

That means the later call graph should not treat a loop as an unbounded set of
runtime edges. It should treat the loop as the finite, analyzed set of EI paths
that the unit inventory exposes.

A loop with an integration call inside it does not mean the graph needs a
literal cycle for every possible iteration. It means the inventory needs to
show the representative way execution reaches that integration point:

- the zero-iteration path may not reach the integration
- the one-or-more path may reach the integration
- disruptive outcomes may alter the path
- the integration EI remains callable-owned
- paths to that EI remain finite and inspectable

The call graph built later from these facts should preserve those collapsed
relationships. It should not "restore" the source loop into a runtime-style CFG
unless that is a deliberate future graph mode.

This is a practical modeling choice. It keeps the artifacts usable for test
generation. A test generator usually needs to know that a loop can skip, enter,
call something, continue, break, return, or raise. It usually does not need an
infinite graph that models every possible iteration count.

### Successor Resolution

Each EI outcome should describe where execution continues when that is known.

When the target EI is known, the outcome should carry the concrete successor
reference. When the target EI cannot be known immediately, the outcome should
carry a target hint that can be resolved later.

Target hints may include:

- the line number of the next statement
- source span context
- conditional polarity
- structural context
- line numbers or EI IDs to skip
- expression context

Successor resolution may need deferred disambiguation for conditionals, loops,
exception handling, and structurally indirect flow.

The important rule is that uncertainty should remain visible until resolved.
The model should not invent a confident successor when the evidence is not
available yet.

### Implicit Returns

Implicit returns are represented structurally.

If normal execution can fall through the end of a callable, the model should
include a synthetic implicit return EI. This gives the callable a real terminal
landing point even when the source does not contain an explicit `return`.

That makes downstream analysis cleaner. A callable that can fall through should
not look like it has no terminal flow.

### Outcome Paths vs. Test Cases

Execution paths are not test cases.

An execution path describes a route through EIs. A test case is a concrete
choice of inputs, fixtures, setup, action, and assertions.

One execution path may require multiple tests. Several input values may be
needed to verify correctness across important equivalence classes, boundary
values, or domain-specific cases.

The inventory identifies executable behavior and paths. Test generation still
decides which concrete values and assertions are meaningful.

### Stage 2 Output

Stage 2 writes EI artifacts under:

```text
dist/pybastion/eis/
```

These artifacts are unit-scoped and callable-organized. They describe the EIs
owned by each callable and the outcome information needed to understand
callable-local flow.

Stage 2 is the primary owner of callable-local control flow semantics. Later
stages rely on the accuracy of this output.

## Stage 3: Inventory Generation

Stage 3 generates the unit inventory.

Earlier stages produce structural facts and EI facts. Stage 3 organizes those
facts into a hierarchical, unit-scoped artifact that is stable, inspectable,
human-readable, and machine-consumable.

Inventory generation should mostly organize and normalize facts that already
exist. It should not casually invent new semantic facts. Supplemental enrichment
is acceptable when it supports representation, review, or resolution, but it
should not hide missing upstream analysis.

### Integration Candidate Analysis

Units contain operations that reach outside the immediate callable context.
Those operations are integration candidates.

An integration candidate may be:

- a call to another callable in the same project
- a call to another unit in the same project
- a call to the Python standard library
- a call to a third-party library
- an operation that crosses an external boundary

External boundaries include things like:

- filesystem
- database
- network
- subprocess
- message bus
- environment
- clock
- randomness

Finding a call expression is not enough by itself. A call expression is an
operational site, but the analyzer still needs contextual evidence to classify
what kind of operation the site represents.

The analysis may use evidence such as:

- imports
- project callable inventories
- annotated parameter types
- inferred local variable types
- known class field types
- known return types
- unit-scoped binding types
- source expression context
- known standard library or external library facts

Integration determination should be evidence-based rather than only syntactic.

### Integration Categories

Integration candidates may be classified into categories such as:

- **interunit**: interaction with another unit or callable in the same project
- **stdlib**: interaction with the Python standard library
- **extlib**: interaction with a third-party library
- **boundary**: interaction with an external system boundary

Classification is project-relative. A resolved target may be treated
differently depending on whether it belongs to the same unit, another project
unit, the standard library, an external library, or an external system boundary.

Boundary classifications usually matter most for testing. For example, a
third-party HTTP client call is an external library call, but it is also a
network boundary. The boundary crossing is usually the more important testing
concern.

Same-unit operations should be distinguished from integrations as early and as
reliably as possible. Same-unit calls may matter to downstream consumers, but
they are not the same thing as crossing a unit or external boundary.

### Paths to Integration Points

For each integration candidate, the inventory records the callable-local
execution paths that can reach it.

An execution path is a list of EI IDs. If multiple conditional routes can reach
the same integration point, the integration point may have multiple execution
paths.

The final EI in each path should be the EI associated with the integration
candidate.

This makes the integration point more useful for test generation. A downstream
consumer can see not only that the integration exists, but how execution reaches
it.

### Feasibility Filtering

Some syntactic paths are impossible because their conditions conflict.

Where possible, paths to integration points should be filtered with feasibility
analysis so impossible syntactic paths do not become test obligations.

The goal is practical test guidance. The inventory should avoid presenting
unreachable paths as if they are meaningful cases to test.

When feasibility cannot be determined, uncertainty should remain visible.

### Inventory Structure

The inventory should represent the unit clearly and accurately.

At a high level, the inventory contains:

- unit metadata
- unit-scoped bindings
- structural entries
- class and callable hierarchy
- signature information
- decorator and marker metadata
- execution items
- integration candidates
- execution paths to integration points
- type hierarchy metadata
- contract method metadata, when available
- summary information

Example high-level shape:

```yaml
unit: keys
fully_qualified_name: package.module.keys
unit_id: U1234567890
filepath: src/package/module/keys.py
language: python
bindings: []
type_hierarchy: {}
contract_methods: {}
entries: []
summary: {}
```

The `entries` tree contains structural items in the unit. Callable entries
contain their execution items and integration candidates. The `bindings`
collection contains unit-scoped names used for resolution and traceability.

EIs and integration candidates are contained by callables. They are not
first-class child entries in the unit hierarchy.

### Stage 3 Output

Stage 3 writes inventory artifacts under:

```text
dist/pybastion/inventory/
```

The generated inventory should preserve analysis results faithfully while making
them easier to inspect, reason about, validate, and consume in downstream
tooling.

At minimum, the generated inventory should provide:

- consistent identifiers for analyzed artifacts
- a hierarchical representation of the analyzed unit
- structural containment relationships
- unit-scoped bindings
- callable-owned execution items
- callable-owned integration candidates
- execution paths to integration points
- summary information about the analyzed unit
- visible uncertainty or findings when analysis is incomplete

## Relationship to Integration Analysis

The integration pipeline consumes the unit inventories.

Unit analysis is responsible for source-level decomposition. It identifies the
unit structure, callable structure, execution items, and unit-local integration
facts. Integration analysis should not redo that work from raw source.

Instead, integration analysis consumes the inventories and derives project-level
relationships from them.

In practical terms:

1. Unit analysis produces inventories.
2. Integration analysis builds a project-level EI call graph.
3. Integration analysis generates seam-oriented test specifications.
4. Integration analysis splits those specifications into focused files for
   review or agent-assisted test generation.

This separation matters. If unit analysis emits confused or incomplete facts,
integration analysis will either expose the gaps or produce weak specs. It
should not hide the problem by guessing.

## Validation Expectations

Because the pipeline is automated, each phase should preserve checkable
invariants.

Examples include:

- IDs are unique and deterministic.
- Entry IDs, binding IDs, and EI IDs follow the documented grammar.
- Unit-scoped bindings do not appear as callable entries.
- Unit-scoped bindings do not own EIs.
- Every EI belongs to exactly one callable.
- Every non-contract executable callable has a reachable terminal EI.
- Every integration candidate references an EI owned by its containing callable.
- Every execution path to an integration point ends at that integration point's
  EI.
- Execution paths only reference EIs that exist in the containing callable.
- Deferred successor hints are either resolved or remain visible as unresolved
  analysis facts.
- Inventory summary counts match the emitted inventory content.
- Ambiguity is surfaced as findings or uncertainty, not hidden.

Validation failures should be surfaced as errors or findings rather than hidden
or silently repaired.

## What This Pipeline Does Not Do

The unit pipeline does not generate finished tests by itself.

It also does not guarantee full coverage by itself. It produces the structured
artifact that tells a developer or agent what executable behavior and
integration surfaces exist. Test generation still has to choose useful inputs,
fixtures, mocks, fakes, assertions, and coverage strategy.

The unit pipeline also does not try to be a runtime interpreter. It models
representative callable-local execution behavior. Loops are collapsed into
finite, useful paths. Integration points are classified from available evidence.
Uncertainty is preserved when the evidence is incomplete.

That is the point. The inventory is not supposed to be magic. It is supposed to
make the important structure explicit enough that the next step has something
solid to work with.