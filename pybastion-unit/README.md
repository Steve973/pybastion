# Pybastion Unit Analysis

The Pybastion Unit Analysis module provides python unit analysis to facilitate
unit testing. With a comprehensive unit ledger that enumerates all execution
items of a python compilation unit, you can ensure complete coverage of the
unit, and utilize generative AI to automate the generation of unit tests.

The following sections describe the conceptual analysis phases. Note that the
sections are not prescribing the specific way to structure or separate the
implementation. There is room to compose or organize the implementation in any
way that is most effective or efficient.

---

## Preliminary Unit/Callable Analysis

To facilitate deeper analysis, it is helpful to scan the unit for its basic
structure. During this preliminary analysis, the higher level structure of the
unit is identified. This includes identifying items like classes and functions.
This feeds into the process of the deeper analysis that follows and provides
context for the deeper analysis.

### Preliminary Unit/Callable Analysis Implementation Notes

- The generation of IDs is important, and it is very specific. For that reason,
the `Callable ID Generation` document must be read and understood before
proceeding with the analysis.

- Currently, `owner_id` and `parent_id` are the same. It is TBD if this will
change in the future and be useful information for downstream analysis.

- Ordinal assignment is a real semantic choice, not just bookkeeping. The script
maintains `ordinal_within_parent` via per owner counters and also builds
`child_ids` in visitation order. That means the inspection artifact is
preserving deterministic sibling ordering, which downstream stages may rely on.

- File to module FQN derivation has an intentional policy choice: `__init__.py`
is excluded from traversal entirely, and path derivation would also collapse
`__init__` to the package name if it were processed. That is worth mentioning 
because it affects whether package level code is represented as a unit at all.

- Last, but not least, is the fact that a content hash is calculated and recorded
for each unit. This might not be useful for downstream analysis, so this will
need to be reviewed and potentially removed.

### Preliminary Unit/Callable Analysis Outcome

This phase of the analysis produces information about the structure of the unit.
This is a representation that establishes the relationships between the items in
the unit and what they contain. Each class and method/function is represented by
its fully qualified name, and a unique identifier, along with other metadata
that contributes to later analysis.

This information can store information for every unit within a source tree. The
source root path is represented, and each unit is clearly identified.

---

## Execution Item Analysis

Execution Items (EIs) are the smallest unit of execution. They result in an
outcome, so anything that has a distinct outcome should be viewed as an EI.
Execution items are not necessarily at the scale of individual statements,
since a statement can include multiple execution items. EIs are also at a
lower level than the usual definition of a branch, since branching instructions
are EIs, but not all EIs are branching instructions.

The result of EI analysis should be a unit-scoped document that lists all EIs
in the unit that represents the control flow of the unit. Unlike the source
code, this document does not represent loops in the traditional sense. Instead,
loops are flattened into the smallest number of possible paths. Generally,
there is one success path, and any number of possible failure paths. A loop is
broken down as if it were invoked with no items, and as if it were invoked with
one item. As such, constructs like `continue` function as a loop exit condition
in the document.

And, though it is not strictly EI-specific, decorator information is captured in
this phase. The decorator name and parameters are included in the EI metadata.

### EI Analysis Implementation Notes

- EIs are decomposed within their parent context. This means that, if the next
EI is available, it is passed in with the statement so it can be included in the
generated result. It also means that, when a statement includes a body, the
statement and its body are decomposed in a way that preserves the contextual
relationship between the statement and the body, and all constituent EIs that
have been decomposed.

- Next EI ID resolution is a two-phase process for some EIs. When the next IE
ID is known, it is immediately included in the generated result. When the next
EI ID is not yet known, it is deferred until the EI set is fully resolved.

- The `next_ei` field is the real flow fact when known. When not known yet, the
outcome should carry a target_hint for later disambiguation.

- Successor resolution may need deferred disambiguation for conditionals, loops,
and structurally indirect flow. That is why `target_hint` metadata exists.

- Correctness depends on callable scoped source slicing. If the statement window
for a callable is wrong, EI IDs, spans, and successor resolution will be wrong.

- Implicit returns are represented structurally. If a callable can fall through,
the model should be able to land on a synthetic implicit return EI.

- Branch outcomes may be modeled in any of three types. The meaningful flow data
lives in statement_outcome, conditional_targets, and disruptive_outcomes, as the
context dictates. An EI entry might need to include multiple outcome types.

- This phase is the primary owner of control flow determination and semantics.
Later phases will rely on accurate modeling of control flow information, like
implicit and explicit returns, and `next_ei` resolution.

### Execution Item Analysis Outcome

The artifact if this analysis is a per-unit file that contains EI information
for the unit. The unit is broken down into functions (that may or may not be
methods), the metadata of each function, and its branches. The branches are
decomposed into entries that are identified by their EI ID. These entries
contain the metadata of the EI, and its outcomes.

An important part of an EI outcome is the knowledge of where the execution
flow continues after an EI. Ideally, this includes the next EI ID, whenever
possible. When it is not yet possible to determine the next EI by its ID, then
it can include "target hint" information to help to disambiguate later
(follow-on) resolution of the next EI ID. This may include information
including:

- the line number of the next statement
- line numbers (or known EI IDs) to skip
- conditional polarity

It is also possible that the callable flow might end after an EI, with or
without a return statement. This is where a synthetic EI can be created to
represent a "landing point" for an implicit return.

---

## Next EI ID Resolution Analysis

This phase does not have to be separated from EI analysis, but it is described
here for elaboration and completeness.

After the initial EI analysis, the next step is to resolve the next EI ID for
each EI that could not be immediately resolved during EI enumeration.

All EI entries from the EI analysis phase that contain a target hint are the
candidates for EI ID resolution. Since all execution items now have their own
unique IDs, the target hints are processed to resolve the next EI ID.

### Next EI ID Resolution Analysis Outcome

This step can be run before the previous output is written, or it can be run
as a precursor to the next step. Either way, the result is that all EI entries
contain a resolved identifier for the next EI.

---

## Integration Points Analysis

Units contain calls to other units. The other units might be within the project,
from the Python standard library, or from external libraries. These integration
points may also be boundary conditions, where they interact with the outside
world (e.g., the console, a database, the file system, etc.). Integration points
are important for understanding the behavior of the unit and for knowing which
calls need to be substituted with a fixture, like a mock, fake, stub, etc.

The challenge, here, is in how to determine the specific type of integration point
that is represented by the EI. Since we have an index of callables from the
`Preliminary Unit/Callable Analysis` phase, it is quite straightforward to
determine *if* something is an integration point, but determining the *type* of
integration point is another matter, entirely. To help with this, we can use a
knowledge base that assists in determining the type of integration point by
listing things like standard library functions, well-known third-party
libraries information, and other helpful information.

Once the integration points have been identified, we need information about how
they are invoked. Specifically, we need to gather information about the
execution path to an integration point when calling into the unit from outside.
This is a list of EI IDs that represent the path from a callable entrypoint into
the unit, and to the integration point. Sometimes, the paths are conditional, so
these execution paths are represented as a list of EI ID sequences/paths. This
list of lists represents all possible paths to the integration point.

To ensure that paths to integration points are feasible and practical, the
analysis utilizes SMT solvers to determine if the paths are possible. When the
idea is to provide concise, accurate, and fine-grained information for writing
unit tests, the information must be useful and practical, without introducing
paths or cases that cannot be tested because they cannot be traversed.

### Integration Points Analysis Implementation Notes

- Integration point determination is evidence-based rather than only syntactic.
The presence of a call expression is enough to identify an operational site, but
not sufficient by itself to classify that site as an integration. Determination,
therefore, involves both discovery and contextual resolution.

- The current implementation begins with AST call discovery and derives a target
expression from the callable object being invoked. It then applies available
contextual knowledge, such as imports, annotated parameter types, and annotated
local variable types, to resolve that target more precisely.

- Same unit operations should be distinguished from integrations as early and as
reliably as possible. In the current implementation, this includes checks for
locally defined symbols, `self` or `cls` bound calls, and resolved targets that
map back to inventory entries within the same unit.

- Integration classification is project-relative. A resolved target may be
treated differently depending on whether it is determined to belong to the same
unit, another project unit, the Python standard library, or an external library.
This classification depends on the quality of available inventories and
knowledge sources, and may therefore be partial when resolution evidence is
incomplete.

- Builtins and builtin methods are currently treated through explicit
knowledge-based checks. Standard library classification, however, is a broader
concern than builtin detection alone and may require a distinct knowledge
source.

- Integration facts are associated with the containing statement context, and
not only with the call expression itself. This supports later association with
execution items and path enumeration.

### Integration Points Analysis Outcome

The result of integration points analysis is a refined understanding of all
integration points in the unit. The information about all EIs is becoming very
clear and detailed. The refined information can be stored in per-unit files, or
this information can be generated as part of a larger analysis phase.

## Hierarchical Restructuring Analysis

So far, the analysis phases have generated fairly flat documents. However, the
unit ledgers should represent the units that they describe as clearly and as
accurately as possible. We can retain hierarchical context by structuring the
data to reflect the hierarchy. This shows how the units contain callables, and
how callables contain their own callables and EIs. The data, and even the EI
IDs, imply the hierarchy pretty clearly. In this phase, we create a document
that models the hierarchy concretely. Here, all items are aggregated into their
parent containers.

### Hierarchical Restructuring Analysis Implementation Notes

- EIs and integration facts are hierarchically contained by callables. In the
analysis, they are represented inside callable specs rather than as first-class
child entries.

### Hierarchical Restructuring Analysis Outcome

The hierarchical restructuring analysis outcome is a document that models the
hierarchy of units, callables, and EIs. This document provides a clear and
accurate representation of the unit's structure, making it easier to understand
and navigate the unit's components. The hierarchical structure is crucial for
organizing and visualizing the relationships between the described items and
for generating the unit ledgers.

---

## Unit Ledger Generation

Unit ledgers consist of three documents:

1. Derived IDs
2. The Ledger
3. Review / Findings

*The derived IDs* document includes a listing of entries and branches. Its
purpose is to provide stable identifiers and references for analyzed artifacts
so that other parts of the ledger can refer to them consistently and
unambiguously. This document helps preserve traceability between structural
entries, execution items, branches, and any other analysis artifacts that need
durable addressing.

*The ledger document* is a representation of the analyzed unit. It captures the
unit's structure in a hierarchical and navigable form, including contained
classes, callables, execution items, and integration facts. Its purpose is to
present the analyzed unit in a way that is both human-readable and
machine-consumable, while preserving the relationships that matter for later
reasoning, inspection, and automation.

*The review / findings document* is a summary of the unit analysis. It can
include observations about completeness, ambiguity, unresolved or partially
resolved artifacts, classification findings, and any other analysis notes that
are useful for understanding the quality or current state of the generated
ledger. This document is intended to support review rather than to redefine the
analyzed facts themselves.

The unit ledger is not a control flow graph (CFG), but it is a representation of
the unit that can be parsed into a CFG in a way that is fairly straightforward.
It is intended to preserve enough structural and behavioral information that
downstream consumers can derive control flow, inspect integrations, and perform
further analysis without having to reconstruct the entire unit from source.

### Unit Ledger Generation Implementation Notes

- Ledger generation should primarily be a transformation of previously analyzed
artifacts rather than a phase that invents new semantic facts. Its
responsibility is to organize, normalize, and package established unit analysis
into the three ledger documents while preserving cross-document consistency.

- In practice, ledger generation may still perform limited supplemental 
enrichment, such as loading project type inventories, consulting source-derived
type information, or incorporating quality metrics. When such enrichment is
performed, it should be treated as support for representation and review, not as
a substitute for missing upstream analysis.

- The transformation should preserve a clear structural distinction between
entries that are represented as hierarchical ledger nodes and callable-owned
analysis details such as execution items and integration facts.

- The three generated documents should be treated as different views over the
same underlying analyzed unit. Identifier assignments, hierarchical ledger
content, and review findings must remain consistent with one another and should
not be derived from conflicting sources or divergent classification logic.

- Ledger generation should also make ambiguity visible. When integration
categorization is incomplete, uncertain, or dependent on external knowledge that
is not available, that uncertainty should be reflected in the review output,
rather than hidden.

### Unit Ledger Generation Outcome

The outcome of unit ledger generation is a unit-scoped artifact that organizes
the results of prior analysis into a stable, structured, and inspectable form.
Any work performed during this stage must not resolve new control flow,
reclassify integrations, or alter upstream semantics.

At a minimum, the generated ledger provides:

- consistent identifiers for derived analysis artifacts
- a hierarchical representation of the analyzed unit
- structural containment relationships between units, classes, and callables
- execution items associated with the callables in which they occur
- integration facts and their relevant contextual information
- a review-oriented summary of findings, ambiguity, or quality-related concerns

The generated ledger should preserve analysis results faithfully while improving
their usability. It should make the analyzed unit easier to inspect, reason
about, validate, and consume in downstream tooling.
