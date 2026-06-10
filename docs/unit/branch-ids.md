## Branch IDs and deterministic numbering

This section is non-negotiable. If IDs are sloppy, the ledger loses most of its
power.

The whole point of this enumeration is that an AI (or a human) can walk a list
and write tests for every branch, which tends to drive you toward 100% branch
coverage. A coverage tool is still the proof, but the ledger is what makes the
work predictable, reliable, and fast.

### Branch entry format

Even though the YAML format stores branches as objects, think of every branch
entry as this mini spec:

* `<branch_id>: <exact condition or pattern> -> <observable outcome>`

Each entry describes a location in the code (by unique ID), the branch condition
or pattern (copied close to source), and the expected outcome in a test.

### Abbreviations

Use the following abbreviations:

* `Uxxxxxxxxxx` = unit ID (SHA-256[:10] of the fully qualified name)
* `Cxxx` = class ID (starts at `C001`)
* `Fxxx` = unit level function ID (def at file scope)
* `Mxxx` = method ID (def inside a class)
* `Exxxx` = execution item ID
* `Bxxx` = unit scoped binding ID

### Fully qualified IDs only

ID format is always fully qualified (scoped). Do not use unscoped IDs like
`F001_E0001`.

* Use `UABCDEF1234_F001_E0001, UABCDEF1234_F001_E0002, ..., UABCDEF1234_F003_E0001, ...`
  for unit-level functions.
* Use `UABCDEF1234_C001_M001_E0001, UABCDEF1234_C001_M001_E0002, ..., UABCDEF1234_C002_M003_E0001, ...`
  for class methods.

### Notable Exception: Binding Enumeration

A binding ID identifies a name bound at unit scope that is useful for name,
type, value, receiver, alias, or integration resolution. A binding is not an
execution item and does not own execution item IDs. Therefore, this is not a
branch ID.

Binding enumeration is covered here because bindings are deterministically
numbered during the unit indexing phase alongside classes, callables, and other
unit-scoped artifacts. Later analysis phases may reference those binding IDs for
resolution, traceability, and inventory consistency.

### Numbering scheme per ledger

#### Units (File, Module, etc.)

* The unit is always `Uxxxxxxxxxx` where `xxxxxxxxxx` is the first 10 characters
  SHA-256 of the fully qualified file name.
* The unit ID prefixes are used as the prefix for all IDs within the unit.
* This ensures all unit level function branch IDs are fully namespaced and
  unique.

#### Classes

* Classes are numbered starting at `C001`.
* Assigned in file order, top to bottom.
* Required for all classes; without them, method IDs would be ambiguous.

#### Unit level functions

* Unit functions are numbered starting at `F001` within the unit (`Uxxxxxxxxxx`).
* Assigned in file order, top to bottom.
* Required for all unit level functions.

#### Methods

* Methods are numbered starting at `M001` within each class (`Cxxx`).
* Assigned in the order within the class definition, top to bottom.
* Required for all class methods.

#### Execution Items

* Execution items are numbered starting at `E0001` within each function or
  method.
* Assigned in order of appearance in the code.

#### Unit Scoped Bindings

* Unit-scoped bindings are numbered starting at `B001` within the unit
  (`Uxxxxxxxxxx`).
* Assigned in file order, top to bottom.
* Unit-scoped bindings represent names bound at compilation-unit scope that are
  not themselves callable control-flow owners.
* Examples include Python module-level assignments, annotated assignments,
  TypeVars, aliases, constants, compiled regexes, and maps of callable values.
* Unit-scoped bindings are represented under the unit's `bindings` collection,
  not under `entries`.
* Unit-scoped bindings do not receive `E####` execution item IDs.
* Unit-scoped bindings do not participate in uncalled callable analysis.
* Uses of a binding may appear in callable-owned execution items. For example,
  `_SHA256_RE` is a binding, while `_SHA256_RE.match(value)` inside a function
  is an operation represented by an EI in that function.

#### Nesting

Use a `.` to append a nested item to its parent designator.

**Classes**:

* For class `C001`, the first nested class is designated `Uxxxxxxxxxx_C001.C002`.
  The number is incremented by 1 for every class, including nested classes.

**Unit level functions**:

* For unit function `Uxxxxxxxxxx_F001`, the first nested def is designated as
  `Uxxxxxxxxxx_F001.F002`.

**Methods**:

* For method `Uxxxxxxxxxx_C001_M001`, the first nested def is designated as
  `Uxxxxxxxxxx_C001_M001.F001`.

**Execution Items**:

* Nesting does not apply to execution items, but the branch ID is still appended
  to the end (for example `Uxxxxxxxxxx_C001_M001.F001_E0001`).

When to use nested IDs:

* Use this nesting scheme only for named nested defs (def or class) that are
  referenced as objects or meaningfully testable in isolation.
* If the nested def is purely an implementation detail and not directly
  targeted, treat its control flow as branches of the parent and do not assign
  a nested designator.

#### Segment Delimiters

* Use `_` to separate segments within an ID.
* Do not use `_` at the beginning or end of an ID.
* Use `.` to separate a parent ID from a nested ID.

#### Leading zeros

* Leading zeros are used to maintain a consistent length.
* Leading zeros are mandatory.

#### Composition order and grammar for unit-scoped bindings

Binding IDs are composed separately from callable/EI IDs:

* Unit binding: `<unit_id>_<binding_id>`

Examples:

* First unit scoped binding: `U1234567890_B001`.
* Second unit scoped binding: `U1234567890_B002`.

Binding IDs must never be used as callable IDs or EI IDs.

#### Composition order and grammar for callables/EIs

IDs must be composed in this order:

1. Scope (`Uxxxxxxxxxx` for the unit, or `Uxxxxxxxxxx_Cxxx` for classes)
2. Nested scope (`Cxxx` as needed for classes, appended immediately after its
   parent scope: `Cxxx.Cxxx<+1>`)
3. Callable (`Fxxx` for unit functions, or `Mxxx` for class methods)
4. Nested callable (as needed, appended immediately after its
   parent callable)
5. Execution item (`Exxxx`)

Grammar, where a preceding question mark indicates optional:

* Unit level callable EI:
  `<unit_id>_<function_id><.<?nested_function_id>>_<execution_item_id>`
* Class level method EI:
  `<unit_id>_<class_id><.<?nested_class_id>>_<method_id><.<?nested_function_id>>_<execution_item_id>`

Examples:

* First execution item of first unit function:
  `U1234567890_F001_E0001`
* First execution item of first nested def inside first unit function:
  `U1234567890_F001.F002_E0001`
* First execution item of first method in first class:
  `U1234567890_C001_M001_E0001`
* First execution item of first nested def inside first method of first nested class:
  `U1234567890_C001.C002_M001.F001_E0001`

### Takeaways

This detailed process ensures that:

* all branch and binding IDs are unique.
* IDs deterministically indicate the relevant location or binding in the code.
* edits to one function or method do not force renumbering across the entire
  artifact.

Apply this scheme consistently across the entire process.