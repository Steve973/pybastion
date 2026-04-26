# Pybastion Unit Analysis

The Pybastion Unit Analysis module analyzes compilation units and produces
structured unit inventories for testing and downstream analysis.

A unit inventory describes the structure and executable behavior of a unit. It
captures classes, functions, methods, unit-scoped bindings, callable-owned
execution items, integration points, and the execution paths that reach those
integration points.

The purpose of the inventory is not merely to summarize source code. It is to
make test obligations and integration surfaces explicit. Coverage tools can
prove what tests executed, but the inventory provides the structured checklist
of executable behavior and integration points that tests should consider.

The following sections describe the conceptual analysis phases. The sections are
not intended to prescribe the exact implementation structure. The implementation
may compose, combine, or organize the work in whatever way is most effective, as
long as the resulting artifacts preserve the same semantics.

---

## Observed Facts and Derived Facts

The unit inventory should preserve a clear distinction between facts observed
directly from source and facts derived through analysis. Observed facts include
source structure, names, signatures, decorators, source spans, expressions, and
syntactic operation targets. Derived facts include resolved targets, inferred
types, integration classification, execution paths, feasibility results, and
hierarchy metadata.

Derived facts are useful, but they should not obscure the observed source facts
that support them. When analysis depends on inference, partial evidence, or
external knowledge sources, that uncertainty should remain visible in the
inventory or related review output rather than being hidden behind a confident
classification.

---

## Preliminary Unit Analysis

To facilitate deeper analysis, the pipeline first scans the unit for its basic
structure. During this preliminary analysis, the durable items in the unit are
identified and assigned deterministic IDs.

This includes identifying items such as:

- classes
- functions
- methods
- nested definitions
- unit-scoped bindings
- field metadata, when available

This phase provides the structural context required by the deeper analysis that
follows.

### Entries and Bindings

The unit index separates structural entries from unit-scoped bindings.

**Entries** are structural items that may be represented in the inventory
hierarchy. Examples include classes, functions, methods, nested classes, and
nested functions.

**Bindings** are names bound at unit scope. They are useful for name, type,
value, receiver, alias, or integration resolution, but they are not callable
control-flow owners.

Examples of unit-scoped bindings include:

- module-level assignments
- annotated assignments
- TypeVars
- constants
- aliases
- compiled regexes
- maps of callable values

A binding does not own execution items. A later use of that binding inside a
callable may be represented by an execution item owned by that callable.

For example:

```python
_SHA256_RE: re.Pattern[str] = re.compile(...)
```

is a unit-scoped binding.

A later use inside a callable:

```python
_SHA256_RE.match(value)
```

is an executable operation and may be represented by an execution item in that
callable.

### Preliminary Unit Analysis Implementation Notes

- The generation of IDs is important, and it is very specific. The callable and
  binding ID generation rules must be understood before changing this phase.

- `owner_id` and `parent_id` currently represent closely related structural
  relationships. If their meanings diverge in the future, that distinction
  should be documented because downstream analysis may depend on it.

- Ordinal assignment is a semantic choice, not just bookkeeping. The index
  preserves deterministic sibling ordering through ordinal fields and child
  ordering. Downstream stages may rely on that ordering when reconstructing
  hierarchy or comparing artifacts.

- File-to-unit FQN derivation is a policy choice. In the current Python
  implementation, `__init__.py` handling affects whether package-level code is
  represented as a unit. Any change to that policy changes the unit inventory
  surface.

- A content hash may be calculated and recorded for each unit. This can be used
  by downstream tooling to detect stale artifacts or changed source content.

### Preliminary Unit Analysis Outcome

This phase produces structured information about the unit.

Each unit is represented with metadata such as:

- unit ID
- fully qualified name
- source path
- language
- source hash, when enabled
- structural entries
- unit-scoped bindings

This information establishes the stable structure of the unit before execution
item and integration point analysis begins.

---

## Execution Item Analysis

Execution Items, or EIs, are callable-owned execution facts. They represent
distinct executable outcomes inside a function or method.

Execution items are not necessarily at the scale of individual source
statements. A statement can include multiple execution items. EIs are also lower
level than the usual definition of a branch. Branching constructs produce EIs,
but not all EIs are branches in the traditional sense.

Examples of EIs include:

- function start
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

The result of EI analysis is a unit-scoped artifact that lists callable-owned
EIs and represents callable-local control flow.

### Flattened Flow Representation

Unlike the source code, the EI artifact does not represent loops as open-ended
runtime cycles. Instead, loops are flattened into the smallest useful set of
representative paths.

Generally, loop analysis represents cases such as:

- zero iterations
- one or more iterations
- disruptive paths, when present

A construct like `continue` can function as a loop continuation or loop exit
outcome in the flattened representation.

This flattening is intentional. The inventory is not trying to be a runtime
interpreter. It preserves enough callable-local flow structure to support test
generation, integration point reachability, and downstream tooling that may
derive larger graphs from unit inventories.

### Execution Item Analysis Implementation Notes

- EIs are decomposed within their parent callable context. If the next EI is
  already known, it should be included in the generated result.

- When a statement contains a body, the statement and its body are decomposed in
  a way that preserves the contextual relationship between the statement, the
  body, and all constituent EIs.

- Next EI ID resolution may be immediate or deferred. When the next EI is known,
  it can be recorded directly. When it is not known yet, the outcome should
  carry a target hint for later disambiguation.

- The `target_ei` field is the concrete flow fact when known. When it cannot be
  known yet, `target_hint` metadata exists to support follow-on resolution.

- Successor resolution may need deferred disambiguation for conditionals, loops,
  exception handling, and structurally indirect flow.

- Correctness depends on callable-scoped source slicing. If the statement
  window for a callable is wrong, EI IDs, spans, and successor resolution will
  also be wrong.

- Implicit returns are represented structurally. If a callable can fall through,
  the model should include a synthetic implicit return EI.

- Branch outcomes may be modeled through multiple outcome structures. The
  meaningful flow data lives in:

  - `statement_outcome`
  - `conditional_targets`
  - `disruptive_outcomes`

- This phase is the primary owner of callable-local control flow semantics.
  Later phases rely on accurate modeling of implicit returns, explicit returns,
  disruptive flow, and successor resolution.

### Execution Item Analysis Outcome

The artifact produced by this phase contains EI information for the unit.

The unit is broken down into callables, and each callable receives its
callable-owned EIs. Each EI contains metadata describing the executable outcome
and how execution continues.

An important part of an EI outcome is the knowledge of where execution flow
continues after the EI. Ideally, this includes a resolved `target_ei` whenever
possible. When it is not yet possible to determine the next EI, the outcome can
include `target_hint` information to help resolve the next EI later.

Target hint information may include:

- the line number of the next statement
- line numbers or known EI IDs to skip
- conditional polarity
- structural context
- source expression context

A callable flow may also end after an EI, with or without an explicit return
statement. When normal execution can fall through the end of a callable, a
synthetic EI can represent the implicit return landing point.

### Outcome Paths vs. Test Cases

Execution items describe distinct executable outcomes, not all input values that
should be tested.

For example, `x - 2` has one executable outcome even though tests may use many
different values for `x`. Likewise, a single EI path may require multiple test
cases to verify correctness across important input ranges.

The inventory identifies what execution behavior exists. Test generation still
decides which concrete values, fixtures, and assertions are needed to verify
that behavior.

---

## Next EI ID Resolution Analysis

This phase does not have to be separated from EI analysis, but it is described
separately here for clarity.

After initial EI enumeration, the next step is to resolve any successor
relationships that could not be resolved immediately during decomposition.

All EI entries that contain target hints are candidates for follow-on EI ID
resolution. Since all execution items now have unique IDs, target hints can be
processed to resolve concrete successor EI IDs.

### Next EI ID Resolution Analysis Outcome

The result of this step is that callable-local flow relationships are resolved
as concrete EI references wherever possible.

This step can happen before the EI artifact is written, or it can be performed
as a follow-on normalization step before inventory generation.

---

## Integration Point Analysis

Units contain operations that reach outside the immediate callable context.
These operations are integration points.

An integration point may be:

- a call to another callable in the same project
- a call to another unit in the project
- a call to the Python standard library
- a call to a third-party library
- an operation that crosses an external boundary, such as filesystem, network,
  database, environment, clock, randomness, subprocess, or message bus

Integration points are important for understanding the behavior of the unit and
for knowing which calls may need to be substituted with a fixture, mock, fake,
stub, or other test double.

The challenge is not merely finding call expressions. A call expression is an
operational site, but the analyzer still needs contextual evidence to classify
what kind of operation that site represents.

### Integration Categories

Integration points may be classified into categories such as:

- **interunit**: interaction with another unit or callable in the same project
- **stdlib**: interaction with the language standard library
- **extlib**: interaction with a third-party library
- **boundary**: interaction with an external system boundary

Boundary integrations take priority when classifications overlap. For example,
a third-party HTTP client call is both an external library call and a network
boundary interaction, but the boundary crossing is usually the more important
testing concern.

### Integration Point Analysis Implementation Notes

- Integration point determination is evidence-based rather than only syntactic.
  The presence of a call expression is enough to identify an operational site,
  but not sufficient by itself to classify that site as an integration point.

- The analysis begins with AST call discovery and derives a target expression
  from the callable object being invoked.

- The target expression is resolved using available context, such as imports,
  project callable inventories, annotated parameter types, inferred local
  variable types, known class field types, known return types, and unit-scoped
  binding types.

- Same-unit operations should be distinguished from integration points as early
  and reliably as possible. Same-unit calls may still matter to downstream
  consumers, but they are not the same thing as crossing a unit or external
  boundary.

- Classification is project-relative. A resolved target may be treated
  differently depending on whether it belongs to the same unit, another project
  unit, the Python standard library, an external library, or an external system
  boundary.

- Builtins and builtin methods may be handled through explicit knowledge-based
  checks. Standard library classification is broader than builtin detection and
  may require its own knowledge source.

- Integration facts are associated with the containing callable and EI context,
  not only with the raw call expression. This supports association with
  execution items and path enumeration.

### Optional Integration Enrichment

When available, integration candidates may include boundary or contract metadata.
Boundary metadata describes the external system interaction represented by the
integration point, such as filesystem access, network calls, database operations,
subprocess execution, message bus interaction, environment access, clock access,
or randomness. This information helps test generation determine what kind of
fixture, mock, fake, or controlled substitute may be needed.

Contract metadata describes the interaction contract of the target, such as its
signature, parameters, return type, raised exceptions, or interaction style. This
information helps downstream tooling understand how the integration point is
called and what kind of behavior a test double may need to provide.

These fields are enrichment. They should be included when supported by source
evidence, project inventories, type information, or trusted knowledge sources.
When the evidence is incomplete, the inventory should omit the field or preserve
the uncertainty rather than presenting a guessed classification as fact.

### Paths to Integration Points

For each integration point, the inventory records the callable-local execution
paths that can reach it.

An execution path is a list of EI IDs. If multiple conditional routes can reach
the same integration point, the integration point may have multiple execution
paths.

The final EI in each execution path should be the EI associated with the
integration point.

These paths describe how execution can reach an integration point from the
containing callable's entry path.

### Feasibility Filtering

Some syntactic paths are impossible because their conditions conflict.

Where possible, paths to integration points are filtered with SMT feasibility
analysis. This avoids turning impossible syntactic paths into test obligations.

The goal is to provide concise, accurate, and fine-grained information for
writing tests without introducing paths or cases that cannot actually be
traversed.

### Integration Point Analysis Outcome

The result of integration point analysis is a refined understanding of the
integration points inside the unit.

For each integration point, the inventory can capture:

- the EI where the integration occurs
- the observed source expression or operation target
- the resolved target, when available
- the classification
- signature information, when available
- execution paths to the integration point
- path feasibility information, when available

This remains unit-scoped analysis. Broader project-level tooling can consume
these facts to build larger views across units.

---

## Inventory Generation

Earlier phases may produce analysis facts in relatively flat forms. Inventory
generation organizes those facts into a hierarchical unit-scoped artifact that
is stable, inspectable, human-readable, and machine-consumable.

The inventory should represent the unit as clearly and accurately as possible.
It should show how the unit contains classes and callables, how callables own
their execution items and integration facts, and how unit-scoped bindings
support resolution without becoming callable entries themselves.

The inventory is not a control flow graph. It is a representation of the unit
that preserves enough structural and behavioral information for downstream
consumers to derive control flow, inspect integrations, generate tests, and
perform further analysis without reconstructing the entire unit from source.

EIs and integration facts are hierarchically contained by callables. They are
represented inside callable analysis data rather than as first-class child
entries. Unit-scoped bindings remain at unit scope. They are emitted alongside
the entry hierarchy because they are available as unit-level resolution context,
but they are not callable entries and they are not execution item owners.

### Inventory Generation Implementation Notes

- Inventory generation should primarily organize, normalize, and package
  previously analyzed facts rather than invent new semantic facts.

- The transformation should preserve a clear structural distinction between
  entries, bindings, and callable-owned analysis details.

- Entries form the structural hierarchy. They represent classes, functions,
  methods, nested classes, and nested functions.

- Bindings represent unit-scoped names used for resolution and traceability.
  They are emitted at unit scope, not inside callable entries.

- EIs and integration candidates belong to the callable in which they occur.

- Identifier assignments, hierarchical inventory content, integration paths,
  and summary information must remain consistent with one another.

- Inventory generation may perform limited supplemental enrichment, such as
  loading project type information, consulting source-derived type information,
  or incorporating quality metrics. Such enrichment should support
  representation and review, not replace missing upstream analysis.

- Ambiguity should remain visible. When integration categorization is
  incomplete, uncertain, or dependent on unavailable external knowledge, the
  inventory or related review output should make that uncertainty clear.

### Inventory Contents

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

The `entries` tree contains the structural items in the unit. Callable entries
contain their execution items and integration candidates. The `bindings`
collection contains unit-scoped names used for resolution and traceability.

### Inventory Generation Outcome

The outcome of inventory generation is a unit-scoped artifact that organizes the
results of prior analysis into a stable, structured, and inspectable form.

At a minimum, the generated inventory provides:

- consistent identifiers for analyzed artifacts
- a hierarchical representation of the analyzed unit
- structural containment relationships between units, classes, and callables
- unit-scoped bindings
- execution items associated with the callables in which they occur
- integration facts and relevant contextual information
- execution paths to integration points
- summary information about the analyzed unit

The generated inventory should preserve analysis results faithfully while
improving their usability. It should make the analyzed unit easier to inspect,
reason about, validate, and consume in downstream tooling.

---

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
- Every execution path to an integration point ends at that integration point's EI.
- Execution paths only reference EIs that exist in the containing callable.
- Deferred successor hints are either resolved or remain visible as unresolved analysis facts.
- Inventory summary counts match the emitted inventory content.

Validation failures should be surfaced as errors or findings rather than hidden
or silently repaired.
