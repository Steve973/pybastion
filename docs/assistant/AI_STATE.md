# AI_STATE.md — PyBastion Integration Pipeline Current Work

## Current objective

Finish the reorganized PyBastion integration analysis pipeline and prepare it to generate useful integration-test
specification artifacts from inventory, graph, seam, and feature-flow analysis.

This file should describe only the current implementation state and remaining work. Broader architecture belongs in
`AI_CONTEXT.md`.

---

## Current pipeline stages

The integration pipeline is conceptually ordered as:

```text
Stage 1: Build EI call graph
Stage 2: Trace feature-flow cases
Stage 3: Generate integration test specs
Stage 4: Split integration test specs
```

The important dependency order is:

```text
unit inventories
  -> Stage 1 graph
  -> Stage 2 feature-flow cases
  -> Stage 3 integration specs
  -> Stage 4 split spec files
```

Feature-flow tracing should happen before final integration spec generation because feature-flow-scoped specs need
completed feature-flow case paths.

---

## Current output files

Normal pipeline output should stay small.
Current intended normal output is:

```text
stage1-ei-cfg.pkl
stage2-feature-flow-cases.yaml
stage3 integration spec output
specs/
```

Stage 2 should not always emit all diagnostic/support files.
Stage 2 support outputs are optional, and mostly for debugging:

```text
stage2-feature-marker-inventory.yaml
stage2-feature-branch-points.yaml
stage2-feature-converge-points.yaml
```

These should only be emitted when both conditions are true:

```text
--emit-all-output is present on the command line
and
the corresponding output path is explicitly provided
```

Do not make Stage 2 invent extra output paths by itself.

---

## Stage 2 current contract

Stage 2 primary output is feature-flow cases.
The main Stage 2 output must be:

```text
--output <stage2-feature-flow-cases.yaml>
```

The marker inventory, branch points, and converge points are support/debug artifacts only.
They must not be treated as the main Stage 2 product.
The Stage 2 script has been adjusted so optional diagnostic outputs require:

```text
--emit-all-output
```

plus explicit output paths.

The main driver also accepts `--emit-all-output`. Currently, that flag only applies to Stage 2 feature-flow tracing.

---

## Fixed defects that should not be reopened

The prior flattened-loop feature-flow defect is fixed.

Do not restore loop backedges.

Do not reintroduce runtime loop cycling into the flattened model.

Terminal representative loop-disruption cases such as `continue` and `break` should complete as terminal/conditional
representative cases. They must not flow into downstream unrelated branches.

Expected feature-flow tracing status for the synthetic fixture remains:

```text
completed_cases: 40
unresolved_cases: 0
```

Bad branch expansion patterns should remain absent:

```text
skip_value::has_content
skip_value::empty_content
```

---

## Remaining Work – The Immediate Next Step

* Stage 3 should generate integration test specification artifacts after feature-flow tracing is complete.

* Stage 3 should preserve two distinct integration-test specification outputs:

  * seam-scoped specs (complete)
  * feature-flow-scoped specs

* These outputs are conceptually separate and have different testing goals.

### Seam specs vs. feature-flow specs

Seam specs aim to provide ample information to test seam boundaries without mocking either unit on either side of the
seam. Seams are important, and testing them shows that each edge works as expected. 

Feature flow specs aim to provide ample information to test entire features from end to end:

- with as little mocking as possible
- to ensure that the feature works as intended
- across the normal, conditional, and error cases that can be encountered

Feature-flow specs should cover the completed feature cases discovered by Stage 2, including normal cases, conditional
cases, and error/disruption cases that the feature can encounter.

Feature-flow specs answer:

```text
What completed feature flow exists, what path does it take, what outcome should it produce, and what integration-level
test should verify that the feature works as intended?
```

At a high, conceptual level, we can summarize the distinction as:
- **seam specs**: used to write tests to verify that adjacent units can work together
- **feature flow specs**: used to write tests to verify the described and intended system functionality

**Note** that the integration seam spec work has been completed and is included here for reference and contrast to the
feature-flow specs. Do not change, enhance, or otherwise modify the existing seam spec generation.

### Stage 3 feature-flow spec generation

Feature-flow-scoped specs should be created from completed feature-flow cases in:

```text
stage2-feature-flow-cases.yaml
```

Use the feature-flow case paths that Stage 2 already produced.

Do not recompute any feature-flow paths by any means or technique; if you suspect that something is wrong or missing,
report it and ask how to proceed.

For each completed feature-flow case, preserve:

```text
feature name
case ID
case branch path
active branch path
outcome
end kind
end marker / terminal marker
segment IDs
assembled EI path / path evidence
integration-relevant EIs and operations on the path
inventory-backed fixture requirements or constraints
```

The feature-flow spec should be centered on the feature case, not on seam specs.

Seam specs may be used as reference material if useful, but feature-flow specs should not be modeled as wrappers around
seam specs or as references to seam specs.

The shared substrate is the inventory, graph, EIs, and completed feature-flow case paths.

Output files remain distinct per integration spec type:

```text
integration-seam-test-specs.yaml
integration-feature-test-specs.yaml
```

---

## Stick strictly to the task at hand

- Do not broaden behavior while finishing this pass

- Do not add speculative enrichment.

- Do not invent test fixtures, inputs, assertions, or expected values that are not directly supported by inventory/spec
  artifacts.

- Do not change loop, try, with, match, or branch semantics unless a specific failing artifact proves a separate defect,
  and you have discussed and designed this with the meatsuit software dude.

- Only change what is strictly necessary to get feature integration specs accurate and correct.

- Only do the work that has been communicated to, and approved by, His Bipedal Excellence, AKA the human.
