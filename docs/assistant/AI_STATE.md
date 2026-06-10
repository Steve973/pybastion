# AI_STATE.md — PyBastion Feature Flow Debug Bookmark

## Current status

The known PyBastion feature-flow tracing breakage for the synthetic feature-flow fixture is fixed.

The previous remaining issue was:

```text
fixture_mixed_control_feature::main::try_success::loop_entered::skip_value
```

That case was correctly no longer flowing into downstream branches, but Stage 4 was still reporting it unresolved instead of completing it as a terminal/conditional representative feature-flow case.

That Stage 4 issue has now been fixed.

Current regenerated feature-flow result:

```yaml
completed_cases: 40
unresolved_cases: 0
```

---

## Non-negotiable model invariants

These remain active and must not be violated.

### Loops are flattened

PyBastion does not model runtime loop cycling in feature-flow artifacts.

A loop may produce finite representative cases such as:

```text
loop_skipped
loop_entered
continue_requested / skip_value
break_requested
normal consume/use path
```

It must not restore runtime loop behavior such as:

```text
continue -> loop iterator/control
body completion -> loop iterator/control
break -> loop iterator/control
```

A graph is only an artifact. It must not override the documented flattened model.

### Feature tracing is branch-state traversal, not generic graph shortest path

Stage 4 must preserve case identity through branch selection and convergence.

Convergence resumes traversal location, but it does not erase branch lineage.

Branch groups multiply only the active cases that reach them.

### Inventory is authoritative

Stage 4 should consume the inventory and graph as modeled artifacts. It must not rediscover raw source structure or reinterpret flattened control as runtime control.

---

## What was fixed earlier

The decomposer loop-backedge bug was already fixed in `control.py`.

The bad previous behavior was that loop routes reintroduced runtime-style cycling:

```text
continue -> loop decision / iterator
body_next_iteration -> loop decision / iterator
```

That caused Stage 4 to produce invalid cases such as:

```text
main::try_success::loop_entered::skip_value::has_content
main::try_success::loop_entered::skip_value::empty_content
```

The decomposer fix removed those invalid loop backedges.

Post-regeneration checks showed:

```text
body_next_iteration routes: 0
derived EI loop_continue backedges: 0
loop_continue routes resolve to terminal placeholders, not loop control
```

The graph now preserves the flattened-loop model for this defect.

---

## Final Stage 4 fix now applied

The final remaining issue was in Stage 4 case finalization.

When an active case landed on a terminal loop-disruption representative EI, Stage 4 attempted to find a normal feature end. Since no normal end path existed, it emitted:

```text
unresolved: main::try_success::loop_entered::skip_value
```

The correct behavior is to complete that representative case at the terminal placeholder already present in the graph.

The applied Stage 4 fix recognizes terminal representative graph edges where:

```python
edge_type == "derived_control_route_execution_item_terminal"
resolved_target_kind == "terminal_placeholder"
route_kind in {"loop_continue", "loop_break"}
exit_kind in {"continue", "break"}
```

When such an edge exists from the current EI, Stage 4 completes the case as:

```text
end_kind: feature_end_conditional
outcome_kind: conditional
```

This is intentionally narrow.

It does not:

```text
restore loop cycling
make continue flow to downstream branches
treat all terminal placeholders as valid feature ends
change schemas
change graph construction
change decomposer behavior
change try/with behavior
```

---

## Current validated result

After the Stage 4 fix and artifact regeneration:

```yaml
completed_cases: 40
unresolved_cases: 0
```

The newly completed case is:

```text
fixture_mixed_control_feature::main::try_success::loop_entered::skip_value::end::UA48866E495_F022::control_terminal::for_436_route_continue_440_0
```

The case now terminates at:

```text
UA48866E495_F022::control_terminal::for_436_route_continue_440_0
```

Bad downstream expansions remain absent:

```text
skip_value::has_content
skip_value::empty_content
continue_requested::has_content
continue_requested::empty_content
break_requested::has_content
break_requested::empty_content
```

Note: branch names such as `continue_requested` and `break_requested` may still appear in legitimate completed fixture cases. That is expected. The invalid pattern is only when terminal disruption branches continue into unrelated downstream branches.

---

## Current fixture case-shape summary

The regenerated synthetic fixture produced this case distribution:

```text
fixture_if_branch_converge                    2
fixture_if_branch_conditional_end             2
fixture_match_branch_converge                 4
fixture_for_loop_entry_converge               2
fixture_while_loop_entry_converge             2
fixture_for_loop_disruption_branches          4
fixture_while_loop_disruption_branches        4
fixture_nested_if_branch_converge             3
fixture_nested_loop_converge                  3
fixture_try_except_else_finally_trace         3
fixture_try_with_inner_if_branch              3
fixture_with_trace_and_content_branch         2
fixture_mixed_control_feature                 6
```

The mixed-control feature now has the expected shape:

```text
main::os_error
main::try_success::loop_entered::skip_value
main::try_success::loop_entered::use_value::empty_content
main::try_success::loop_entered::use_value::has_content
main::try_success::loop_skipped::empty_content
main::try_success::loop_skipped::has_content
```

---

## What is safe to say

For the supplied synthetic feature-flow fixture, graph, inventory, and Stage 4 code:

```text
The known flattened-loop disruption bug is fixed.
The Stage 4 unresolved skip_value case is fixed.
The old bad downstream expansion is absent.
The regenerated synthetic fixture now hits the expected 40 completed / 0 unresolved result.
The case artifact is structurally consistent.
```

Do not claim the entire feature-flow engine is globally perfect.

This fix validates the known bug and this fixture. Broader confidence requires running the wider fixture suite and adding golden expected case-ID regression tests.

---

## Recommended regression tests

Add or update a golden regression test for the synthetic feature-flow fixture.

Minimum assertions:

```text
completed_cases == 40
unresolved_cases == 0
```

Stronger assertions:

```text
actual_case_ids == expected_case_ids
```

Also assert absence of invalid downstream terminal-disruption expansions:

```text
skip_value::has_content
skip_value::empty_content
continue_requested::has_content
continue_requested::empty_content
break_requested::has_content
break_requested::empty_content
```

And assert presence of the completed terminal representative case:

```text
fixture_mixed_control_feature::main::try_success::loop_entered::skip_value::end::UA48866E495_F022::control_terminal::for_436_route_continue_440_0
```

---

## What not to do next

Do not continue changing Stage 4 unless a new fixture or regression proves a separate defect.

Do not restore loop backedges.

Do not add broad terminal-placeholder completion behavior.

Do not make `continue` or `break` flow to downstream feature branches.

Do not change graph-builder behavior for this issue.

Do not change decomposer behavior for this issue.

Do not change schema for this issue.

The next sensible step is to commit the narrow Stage 4 fix and the regenerated/updated regression expectations.
