DEBUG_TRACE_CALLABLES = {
    "project_resolution_engine.internal.resolvelib.ProjectResolutionProvider._materialize_requirements",
    "project_resolution_engine.internal.resolvelib.ProjectResolutionProvider._compute_bad_set",
    "project_resolution_engine.internal.resolvelib.ProjectResolutionProvider._build_uri_candidates",
    "project_resolution_engine.internal.resolvelib.ProjectResolutionProvider._sort_candidates",
    "project_resolution_engine.internal.resolvelib.ProjectResolutionProvider._load_pep691",
}

DEBUG_TRACE_SYMBOLS = {
    "_services",
    "requirements",
    "incompatibilities",
    "candidates",
}

DEBUG_TRACE_TARGETS = {
    "requirements.get",
    "incompatibilities.get",
    "candidates.append",
    "candidates.sort",
    "self._services.index_metadata.resolve",
}


def debug_should_trace_target(target: str) -> bool:
    return target in DEBUG_TRACE_TARGETS


def debug_print_resolution_state(label: str, **values: object) -> None:
    print(f"[chain-debug] {label}")
    for key, value in values.items():
        print(f"  {key} = {value!r}")


def debug_trace_known_types(
        callable_fqn: str,
        known_types: dict[str, str] | None,
        local_types: dict[str, str] | None,
        merged_known_types: dict[str, str] | None,
) -> None:
    if callable_fqn not in DEBUG_TRACE_CALLABLES:
        return

    known_types = known_types or {}
    local_types = local_types or {}
    merged_known_types = merged_known_types or {}

    print(f"\n=== DEBUG TYPE CONTEXT: {callable_fqn} ===")
    for symbol in sorted(DEBUG_TRACE_SYMBOLS):
        print(
            f"  {symbol}: "
            f"known={known_types.get(symbol)!r} "
            f"local={local_types.get(symbol)!r} "
            f"merged={merged_known_types.get(symbol)!r}"
        )
