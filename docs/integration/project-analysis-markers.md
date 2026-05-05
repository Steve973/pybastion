# Project Analysis Markers Specification

## Overview

This specification defines comment-based markers for marking functions and
methods with metadata that controls integration flow tracing, feature flow
identification, and call graph interpretation. These markers serve three primary
purposes:

1. **Operation Markers** – Denote mechanical and utility operations to be
   excluded from flow tracing or represented with mocks/fixtures
2. **Feature Flow Markers** – Define boundaries, waypoints, branching, and
   convergence points for end-to-end feature testing
3. **Reachability Markers** – Identify callables that are invoked externally,
   by frameworks, or through abstraction/dispatch mechanisms that may not appear
   as ordinary direct calls in the project call graph

All markers use a pipe-delimited syntax embedded in comments immediately
preceding the item that they pertain to. Some markers appear immediately above
the function or method definition, and others appear immediately before the
statement that they apply to.

Note that project analysis markers are captured during the unit analysis phase,
along with any other decorator. Their usage, however, pertains to the
integration analysis phase.

## Marker Syntax

### General Format

```
:: MarkerName | field=value | field=value
```

**Rules:**
- Marker name is case-sensitive
- Fields are pipe-delimited (`|`)
- Field values use `key=value` format
- Whitespace around delimiters is ignored
- Quote marks around values are stripped during parsing
- Can appear in any comment style supported by the language

### Placement

Place markers in comments **immediately before** the function or method
signature:

```python
# :: MechanicalOperation | type=serialization
def to_dict(self) -> dict:
    return {"name": self.name}
```

Immediately preceding the statement that they apply to:

```python
def find_waldo(self) -> str:
    # :: FeatureStart | name=waldo_search | comment="Where's Waldo?"
    location: str = location_analysis("waldo")
    return location
```

Or inside method-level docstrings when they apply to the entire method:

```python
def validate_schema(data: dict) -> bool:
    """
    :: MechanicalOperation | type=validation
    
    Validate data against schema.
    """
    return schema.validate(data)
```

## Operation Markers (Function and Method Scope)

Operation markers denote functions that should be excluded from integration
flow tracing. These are mechanical transformations or utility operations that
do not contain business logic. All operation markers are applied to functions
and methods.

### MechanicalOperation

Marks pure data transformations that do not make business decisions.

**Format:**
```
:: MechanicalOperation | type=TYPE | alias=NAME | comment="DESCRIPTION"
```

**Fields:**

| Field   | Required | Description                                 |
|---------|----------|---------------------------------------------|
| type    | Yes      | Operation category from the list below      |
| alias   | No       | Alternative name for configuration matching |
| comment | No       | Human-readable description                  |

**Valid Types:**

| Type            | Description                                           |
|-----------------|-------------------------------------------------------|
| serialization   | Convert objects to external format (JSON, dict, etc.) |
| deserialization | Convert external format to objects                    |
| validation      | Schema or format checking without business rules      |
| formatting      | String formatting, output rendering                   |
| conversion      | Type conversion, coercion                             |
| normalization   | Data normalization without business logic             |
| data_transform  | Pure data reshaping without decisions                 |
| presentation    | UI or display formatting                              |
| construction    | Object construction or initialization                 |

**Example:**
```python
# :: MechanicalOperation | type=serialization | comment="Convert to JSON"
def to_mapping(self) -> dict[str, Any]:
    """Convert package to dictionary."""
    return {"name": self.name, "version": str(self.version)}
```

### UtilityOperation

Marks infrastructure or plumbing operations that do not contain business
logic.

**Format:**
```
:: UtilityOperation | type=TYPE | alias=NAME | comment="DESCRIPTION"
```

**Fields:**

| Field   | Required | Description                                 |
|---------|----------|---------------------------------------------|
| type    | Yes      | Operation category from the list below      |
| alias   | No       | Alternative name for configuration matching |
| comment | No       | Human-readable description                  |

**Valid Types:**

| Type           | Description                                          |
|----------------|------------------------------------------------------|
| logging        | Log statements, debug output                         |
| caching        | Cache get/set operations                             |
| config         | Configuration reading                                |
| observability  | Metrics, tracing, monitoring                         |
| audit          | Audit trail logging                                  |
| data_structure | Generic collection operations (flatten, sort, group) |
| registry       | Registry lookups or registration                     |

**Example:**
```python
# :: UtilityOperation | type=logging
def log_transaction(txn_id: str) -> None:
    logger.info(f"Transaction {txn_id} processed")
```

### When to Use Operation Markers

**Mark as MechanicalOperation:**
- Pure data transformations (same input always produces same output)
- Schema or format validation without business rules
- Serialization and deserialization
- Type conversion
- String formatting for display

**Mark as UtilityOperation:**
- Logging, metrics, tracing
- Caching operations
- Configuration reading
- Infrastructure plumbing

**Do NOT Mark (Business Logic):**
- Decision-making based on business rules
- Orchestration or coordination between units
- State changes with business significance
- Domain logic
- Anything that would be valuable to trace in integration tests

**Example – What to Mark vs Not Mark:**
```python
# MARK: Pure serialization
# :: MechanicalOperation | type=serialization
def to_dict(self) -> dict:
    return {"name": self.name, "price": self.price}

# MARK: Schema validation only
# :: MechanicalOperation | type=validation
def validate_format(data: dict) -> bool:
    return isinstance(data.get("name"), str)

# DO NOT MARK: Includes business logic
def validate_pricing_rules(product: Product) -> bool:
    """Validates business rules for pricing."""
    if product.category == "premium" and product.price < 100:
        return False  # Business rule: premium must be >= $100
    return True

# MARK: Just logging
# :: UtilityOperation | type=logging
def log_transaction(txn_id: str) -> None:
    logger.info(f"Transaction {txn_id} processed")

# DO NOT MARK: Core business logic
def process_payment(amount: float, method: str) -> PaymentResult:
    """Process payment - core business logic."""
    log_transaction(payment.id)  # Flow stops at this call (marked)
    result = validate_payment(amount, method)  # Flow continues
    return finalize_transaction(result)  # Flow continues
```

## Reachability Markers (Function and Method Scope)

Reachability markers describe callables that may be executed even when they do
not appear to have a normal direct caller in the project call graph. These
markers are used by diagnostics and test planning to avoid misclassifying
framework entry points, externally invoked API methods, and abstraction-based
dispatch targets as unused or suspicious. All reachability markers are applied
to functions and methods.

Reachability markers do not exclude a callable from flow tracing by default.
They explain how the callable is reached.

### ExternalApiMethod

Marks a callable as externally reachable through an API boundary. This is useful
for public entry points that are invoked by clients, CLIs, services, plugins, or
other systems outside the analyzed project graph.

**Format:**

```text
:: ExternalApiMethod | comment = Human readable description
```

**Fields:**

| Field    | Required | Description                |
|----------|----------|----------------------------|
| comment  | No       | Human-readable description |

**Example:**

```python
# :: ExternalApiMethod | comment = CLI entry point for dependency resolution
def resolve_command(args: argparse.Namespace) -> int:
    return run_resolution(args)
```

### FrameworkCallback

Marks a callable as invoked by a framework lifecycle, hook, event system, or
registration mechanism. This is useful when the project does not directly call
the method, but a framework will call it at runtime.

**Format:**

```text
:: FrameworkCallback | comment = Human readable description
```

**Fields:**

| Field     | Required | Description                                                     |
|-----------|----------|-----------------------------------------------------------------|
| hook      | No       | Lifecycle hook, event name, route, callback, or extension point |

**Example:**

```python
# :: FrameworkCallback | comment = Called by the resolution provider
def temp_project_dir(tmp_path: Path) -> Path:
    return tmp_path / "project"
```

### CalledThroughAbstraction

Marks a callable as expected to be reached through an abstraction, interface,
contract, registry, strategy, plugin mechanism, or dynamic dispatch path rather
than through a direct source-level call.

**Format:**

```text
:: CalledThroughAbstraction | comment = Human readable description
```

**Fields:**

| Field       | Required | Description                |
|-------------|----------|----------------------------|
| comment     | No       | Human-readable description |

**Example:**

```python
# :: CalledThroughAbstraction | comment = Called by the repository manager
def resolve_from_index(request: ResolutionRequest) -> ResolutionResult:
    return index_resolver.resolve(request)
```

## Feature Flow Markers (Statement Scope)

Feature-flow markers are analysis markers used to guide feature-level
integration tracing over the project EI graph. They identify feature-relevant
EIs, force branch-path selection where shortest-path tracing would choose the
wrong route, mark branch convergence, and identify completion paths that should
become feature integration test obligations. When features span multiple units
and contain multiple integration points, feature-flow markers become critical
for accurately tracing the execution paths that comprise the feature.

Feature flow markers apply to statements. Place the marker comment immediately
before the statement whose execution item should carry the marker.

### Overview of the Feature Flow Model

A feature flow consists of:
- **One entry point** (FeatureStart) – where the feature begins
- **Zero or more waypoints** (FeatureTrace) – intermediate steps
- **Zero or more branch points** (FeatureBranch) – where execution splits
- **Zero or more convergence points** (FeatureConverge) – where branches merge
- **One or more exit points** (FeatureEnd, FeatureEndConditional) – where the
  feature completes

All markers in a feature flow are correlated by the `name` field. Branches
within a feature flow are correlated by the `branch` field.

One or more markers apply to the next code statement:

```python
def resolve(params: ResolutionParams) -> ResolutionResult:
    # :: FeatureStart | name=full_resolution
    configs = _normalize_strategy_configs(params.strategy_configs)

    # :: FeatureBranch | name=full_resolution | branch=main | control_polarity=true
    # :: FeatureBranch | name=full_resolution | branch=resolve_no_env | control_polarity=false
    for env in params.target_environments:
        result = resolve_environment(env)

    # :: FeatureEnd | name=full_resolution
    return _format_result(result)
```

The marker does not apply to the surrounding callable as a whole. It applies to
the execution item generated for the marked statement.

### FeatureStart

Marks the statement where a feature flow begins.

**Format:**
```text
:: FeatureStart | name=FEATURE_NAME | variants=VARIANT_LIST | comment="DESC"
```

**Fields:**

| Field    | Required | Description                                         |
|----------|----------|-----------------------------------------------------|
| name     | Yes      | Unique identifier for this feature flow             |
| variants | No       | Comma-separated list of test configuration variants |
| comment  | No       | Human-readable description                          |

**Constraints:**
- Each feature flow must have exactly one FeatureStart
- Must have at least one corresponding FeatureEnd or FeatureEndConditional
  with the same `name`
- Applies to the statement immediately following the marker

**Example:**
```python
def resolve(params: ResolutionParams) -> ResolutionResult:
    """Resolve package dependencies."""

    # :: FeatureStart | name=dependency_resolution
    configs = _normalize_strategy_configs(params.strategy_configs)

    repo = open_repository(params.repo_id, params.repo_config)
    services = load_services(repo, configs)
    result = rl_resolve(services, params.environments, roots)

    # :: FeatureEnd | name=dependency_resolution
    return _format_result(result)
```

**Example with test variants:**
```python
def resolve(services: ResolutionServices, env: Env, roots: list) -> Result:
    """
    Main resolution entry point.

    Test variants configure different resolver strategies:
    - index_resolver: Use PyPI index-based resolution
    - builder_resolver: Build from sdist when needed
    - hybrid_fallback: Try index first, fall back to builder
    """

    # :: FeatureStart | name=resolution | variants=index_resolver,builder_resolver,hybrid_fallback
    provider = ProjectResolutionProvider(services=services, env=env)

    reporter = ProjectResolutionReporter()
    resolver = Resolver(provider, reporter)
    result = resolver.resolve(roots)

    # :: FeatureEnd | name=resolution
    return result
```

### FeatureTrace

Marks a waypoint along a feature flow path. Used to track intermediate steps
or disambiguate complex flows.

**Format:**
```text
:: FeatureTrace | name=FEATURE_NAME | branch=BRANCH | comment="DESC"
```

**Fields:**

| Field   | Required | Description                                    |
|---------|----------|------------------------------------------------|
| name    | Yes      | Feature flow identifier (matches FeatureStart) |
| branch  | No       | Branch identifier if on a named branch         |
| comment | No       | Human-readable description                     |

**When to use:**
- To mark intermediate steps in a complex feature flow
- To disambiguate paths when automatic tracing is unclear
- To mark where a named branch continues after branching

**Example – Simple waypoint:**
```python
def resolve(params: ResolutionParams) -> ResolutionResult:
    # :: FeatureStart | name=dependency_resolution
    configs = _normalize_strategy_configs(params.strategy_configs)

    # :: FeatureTrace | name=dependency_resolution
    repo = open_repository(params.repo_id, params.repo_config)

    services = load_services(repo, configs)
    result = rl_resolve(services, params.environments, roots)

    # :: FeatureEnd | name=dependency_resolution
    return _format_result(result)
```

**Example – Branch waypoint:**
```python
def get_artifact(key: str) -> Artifact:
    # :: FeatureBranch | name=artifact_retrieval | branch=cache_hit | control_polarity=true
    # :: FeatureBranch | name=artifact_retrieval | branch=cache_miss | control_polarity=false
    if cached := _check_cache(key):
        # :: FeatureTrace | name=artifact_retrieval | branch=cache_hit
        return cached

    # :: FeatureTrace | name=artifact_retrieval | branch=cache_miss
    artifact = remote_repository.fetch(key)

    _update_cache(key, artifact)
    return artifact
```

**Example – Multiple feature flows:**
```python
def resolve(params: ResolutionParams) -> ResolutionResult:
    # :: FeatureTrace | name=dependency_resolution
    # :: FeatureTrace | name=package_installation
    repo = open_repository(params.repo_id, params.repo_config)

    ...
```

### FeatureBranch

Marks a control statement where a feature flow splits into one or more named
paths.

A control statement may have multiple feature-relevant outcomes. Each
FeatureBranch marker describes one named branch mapping for the marked control
statement. Multiple FeatureBranch markers may appear on the same control
statement.

**Format:**
```text
:: FeatureBranch | name=FEATURE_NAME | branch=BRANCH | control_polarity=true|false | comment="DESC"
```

**Fields:**

| Field                  | Required | Description                                     |
|------------------------|----------|-------------------------------------------------|
| `name`                 | Yes      | Feature flow identifier (matches FeatureStart)  |
| `branch` or `branches` | Yes      | Branch name, or comma-separated branch names    |
| `control_polarity`     | No       | Selects the modeled control outcome for this branch |
| `comment`              | No       | Human-readable description                      |

**Branch lifecycle:**
1. FeatureBranch names a feature branch at a control statement
2. `control_polarity`, when present, maps that branch to a modeled control
   outcome
3. Each branch may continue via FeatureTrace markers with the same `branch`
4. Each branch must eventually:
   - Reach a FeatureEnd or FeatureEndConditional for that branch, OR
   - Participate in a FeatureConverge that retires the branch

**Control polarity:**

`control_polarity` maps a named feature branch to a modeled control outcome.

| Statement           | `control_polarity=true`                  | `control_polarity=false`                         |
|---------------------|------------------------------------------|--------------------------------------------------|
| `if`                | Condition satisfied                      | Condition not satisfied                          |
| `for` / `async for` | Loop body entered / another iteration    | Zero iterations / no more iterations             |
| `while`             | Condition initially true / body entered  | Condition initially false / body skipped         |
| `match`             | A case matched                           | No case matched / fallthrough, when modeled      |

If multiple modeled outcomes have the same polarity, `control_polarity` alone
may not uniquely identify one outgoing path. In that case, the tracer may keep
all matching outcomes, or an additional selector may be needed.

**Example – Loop with main and no-input branches:**
```python
def resolve(params: ResolutionParams) -> ResolutionResult:
    # :: FeatureStart | name=full_resolution
    results = []

    # :: FeatureBranch | name=full_resolution | branch=main | control_polarity=true
    # :: FeatureBranch | name=full_resolution | branch=resolve_no_env | control_polarity=false
    for env in params.target_environments:
        result = resolve_environment(env)
        results.append(result)

    # :: FeatureBranch | name=full_resolution | branch=main | control_polarity=true
    # :: FeatureBranch | name=full_resolution | branch=resolve_no_result | control_polarity=false
    if results:
        # :: FeatureEnd | name=full_resolution | branch=main
        return ResolutionResult.success(results)

    # :: FeatureEndConditional | name=full_resolution | branch=resolve_no_result | on_condition=no_resolved_environments
    return ResolutionResult.empty()
```

**Example – Conditional success and failure branches:**
```python
def process_order(order: Order) -> OrderResult:
    # :: FeatureStart | name=order_processing
    validation = validate_order(order)

    # :: FeatureBranch | name=order_processing | branch=main | control_polarity=true
    # :: FeatureBranch | name=order_processing | branch=validation_failed | control_polarity=false
    if validation.ok:
        charge = charge_payment(order.payment)

        # :: FeatureEnd | name=order_processing | branch=main
        return OrderResult.success(charge)

    # :: FeatureEndConditional | name=order_processing | branch=validation_failed | on_condition=validation_failed
    return OrderResult.failure(validation.errors)
```

**Example – Match branches:**
```python
def resolve_source(source: SourceConfig) -> Artifact:
    # :: FeatureStart | name=source_resolution
    source_type = source.kind

    # :: FeatureBranch | name=source_resolution | branch=pypi | control_polarity=true
    # :: FeatureBranch | name=source_resolution | branch=local | control_polarity=true
    # :: FeatureBranch | name=source_resolution | branch=unsupported_source | control_polarity=false
    match source_type:
        case "pypi":
            artifact = fetch_from_pypi(source)
        case "local":
            artifact = load_from_local_path(source)
        case _:
            # :: FeatureEndConditional | name=source_resolution | branch=unsupported_source | on_condition=unsupported_source
            return Artifact.error("Unsupported source")

    # :: FeatureEnd | name=source_resolution
    return artifact
```

### FeatureConverge

Marks a point where multiple branches merge back together.

**Format:**
```text
:: FeatureConverge | name=FEATURE_NAME | branches=BRANCH_LIST
:: | into=PARENT_BRANCH | comment="DESC"
```

**Fields:**

| Field    | Required | Description                                        |
|----------|----------|----------------------------------------------------|
| name     | Yes      | Feature flow identifier (matches FeatureStart)     |
| branches | Yes      | Comma-separated list of branch names that converge |
| into     | No       | Parent branch name to converge into                |
| comment  | No       | Human-readable description                         |

**Convergence rules:**
- If `into` is absent: branches converge back to main (the primary flow)
- If `into` is present: branches converge back to the named parent branch
- All branch names in `branches` are retired after convergence
- The parent flow (main or named branch) resumes after convergence
- Applies to the statement immediately following the marker

**Example – Converge to main:**
```python
def validate_input(data: dict) -> ProcessedResult:
    # :: FeatureStart | name=validation
    simple_case = _is_simple_case(data)

    # :: FeatureBranch | name=validation | branch=fast_path | control_polarity=true
    # :: FeatureBranch | name=validation | branch=slow_path | control_polarity=false
    if simple_case:
        result = _fast_validate(data)
    else:
        result = _slow_validate(data)

    # :: FeatureConverge | name=validation | branches=fast_path,slow_path
    processed = _process_result(result)

    # :: FeatureEnd | name=validation
    return processed
```

**Example – Nested branching (converge to parent branch):**
```python
def resolve_package(name: str) -> Package:
    # :: FeatureStart | name=resolution
    cached = _check_local_cache(name)

    # :: FeatureBranch | name=resolution | branch=local | control_polarity=true
    # :: FeatureBranch | name=resolution | branch=remote | control_polarity=false
    if cached:
        package = _resolve_local(name)
    else:
        source = _select_remote_source(name)

        # :: FeatureBranch | name=resolution | branch=pypi | control_polarity=true
        # :: FeatureBranch | name=resolution | branch=conda | control_polarity=true
        match source:
            case "pypi":
                package = _fetch_pypi(name)
            case "conda":
                package = _fetch_conda(name)

        # :: FeatureConverge | name=resolution | branches=pypi,conda | into=remote
        package = _cache_remote_package(package)

    # :: FeatureConverge | name=resolution | branches=local,remote
    validated = _validate_package(package)

    # :: FeatureEnd | name=resolution
    return validated
```

### FeatureEnd

Marks a definitive exit point where a feature flow completes.

**Format:**
```text
:: FeatureEnd | name=FEATURE_NAME | branch=BRANCH | comment="DESC"
```

**Fields:**

| Field   | Required | Description                                    |
|---------|----------|------------------------------------------------|
| name    | Yes      | Feature flow identifier (matches FeatureStart) |
| branch  | No       | Branch identifier if ending a named branch     |
| comment | No       | Human-readable description                     |

**Constraints:**
- Each feature flow must have at least one FeatureEnd or FeatureEndConditional
- If multiple ends exist, they represent different completion paths
- If on a named branch, include `branch`
- Applies to the statement immediately following the marker

**Example – Single end:**
```python
def resolve(params: ResolutionParams) -> ResolutionResult:
    # :: FeatureStart | name=dependency_resolution
    result = rl_resolve(params)

    # :: FeatureEnd | name=dependency_resolution
    return _format_result(result)
```

**Example – Multiple ends for different paths:**
```python
def validate_config(config: dict) -> ValidationResult:
    # :: FeatureStart | name=validation_pipeline
    schema_result = _check_schema(config)

    # :: FeatureBranch | name=validation_pipeline | branch=main | control_polarity=true
    # :: FeatureBranch | name=validation_pipeline | branch=schema_invalid | control_polarity=false
    if schema_result.valid:
        business_result = _validate_business_rules(config)

        # :: FeatureBranch | name=validation_pipeline | branch=main | control_polarity=true
        # :: FeatureBranch | name=validation_pipeline | branch=business_invalid | control_polarity=false
        if business_result.valid:
            # :: FeatureEnd | name=validation_pipeline | branch=main
            return ValidationResult.success()

        # :: FeatureEnd | name=validation_pipeline | branch=business_invalid
        return ValidationResult.failure(business_result.errors)

    # :: FeatureEnd | name=validation_pipeline | branch=schema_invalid
    return ValidationResult.failure(schema_result.errors)
```

**Example - Branch-specific end:**
```python
def process_payment(method: str, amount: float) -> PaymentResult:
    # :: FeatureStart | name=payment
    payment_method = method

    # :: FeatureBranch | name=payment | branch=credit_card | control_polarity=true
    # :: FeatureBranch | name=payment | branch=paypal | control_polarity=false
    if payment_method == "credit_card":
        result = _process_credit_card(amount)

        # :: FeatureEnd | name=payment | branch=credit_card
        return PaymentResult.from_credit(result)

    result = _process_paypal(amount)

    # :: FeatureEnd | name=payment | branch=paypal
    return PaymentResult.from_paypal(result)
```

### FeatureEndConditional

Marks a conditional exit point where a feature flow may complete based on
runtime conditions.

**Format:**
```text
:: FeatureEndConditional | name=FEATURE_NAME | on_condition=CONDITION | branch=BRANCH | comment="DESC"
```

**Fields:**

| Field        | Required | Description                                     |
|--------------|----------|-------------------------------------------------|
| name         | Yes      | Feature flow identifier (matches FeatureStart)  |
| on_condition | Yes      | Description of the condition that triggers exit |
| branch       | No       | Branch identifier if ending a named branch      |
| comment      | No       | Human-readable description                      |

**When to use:**
- Early termination based on runtime conditions
- Error paths that exit the feature
- Optional exit points in conditional logic
- Dead-end paths where the feature cannot proceed
- Empty-input or no-work paths

**Example:**
```python
def process_data(data: dict) -> ProcessedData:
    # :: FeatureStart | name=data_processing
    valid = _validate_input(data)

    # :: FeatureBranch | name=data_processing | branch=main | control_polarity=true
    # :: FeatureBranch | name=data_processing | branch=invalid_input | control_polarity=false
    if valid:
        result = _transform_data(data)

        # :: FeatureEnd | name=data_processing | branch=main
        return _finalize_processing(result)

    # :: FeatureEndConditional | name=data_processing | branch=invalid_input | on_condition=invalid_input
    return ProcessedData.error("Invalid input")
```

## Notes for Implementation

- Feature flow markers are statement-applicable.
- Markers are collected from comments immediately preceding the marked
  statement.
- The marker applies to the EI generated for the marked statement.
- Multiple feature flow markers may be attached to the same statement.
- Multiple FeatureBranch markers may be attached to the same control statement.
- FeatureBranch uses `branch` to name a single feature branch.
- FeatureBranch may use `branches` where a comma-separated list is useful, but
  one marker per named branch is preferred when mapping branches to control
  outcomes.
- `control_polarity` is meaningful only when the marked statement has modeled
  control outcomes.
- Feature flow tracing should use the project-level EI graph and the marker
  metadata attached to EI nodes.
- Feature flow tracing should preserve meaningful branch paths, including early
  exits and dead-end branches.
- Feasibility analysis may be used to filter impossible path combinations.
- Do not reconstruct source control flow during feature tracing. Use the EI graph
  and the control outcomes emitted by unit analysis.

## Co-occurrence Rules

### Operation Markers

MechanicalOperation and UtilityOperation have no co-occurrence constraints.
They apply only to the function they decorate.

### Reachability Markers

Reachability markers have no strict co-occurrence constraints.

A callable may have more than one reachability marker if it is legitimately
reachable through multiple mechanisms. For example, a method may be both an
external API method and a framework callback.

Reachability markers may co-occur with feature flow markers. They may also
co-occur with operation markers, although that should be used carefully because
operation markers may affect flow tracing while reachability markers affect
diagnostic interpretation.

### Feature Flow Markers

Feature flow markers have codebase-wide co-occurrence rules based on the
`name` field:

**FeatureStart constraints:**
- Must appear exactly once per feature flow
- Requires at least one of:
  - FeatureEnd with matching `name`, OR
  - FeatureEndConditional with matching `name`
- May have multiple FeatureEnd or FeatureEndConditional markers
- May have zero or more FeatureTrace, FeatureBranch, FeatureConverge

**FeatureBranch constraints:**
- Each declared branch name must eventually:
  - Appear in a FeatureConverge `branches` field, OR
  - Reach a FeatureEnd or FeatureEndConditional with that `branch_name`

**FeatureConverge constraints:**
- All branch names in `branches` must have been declared by a FeatureBranch
- If `into` is present, it must reference a currently active branch
- After convergence, the listed branch names are retired

## Extraction During Inventory Generation

When analyzing a callable during inventory generation:

1. **Collect comments** immediately preceding the function or method signature
   - Look backwards from the function line
   - Include: single-line comments (`//`, `#`), multi-line comments
     (`/* */`), docstrings (`"""`, `'''`), doc comments (`///`, `/**`)
   - Stop at blank lines or code

2. **Search for marker patterns**:
   - `:: MechanicalOperation`
   - `:: UtilityOperation`
   - `:: FeatureStart`
   - `:: FeatureTrace`
   - `:: FeatureBranch`
   - `:: FeatureConverge`
   - `:: FeatureEnd`
   - `:: FeatureEndConditional`

3. **Extract matching lines**

4. **Parse fields**:
   - Split on `|` delimiter
   - The first segment is the marker name
   - Remaining segments are `key=value` pairs
   - Strip whitespace around delimiters
   - Remove surrounding quotes from values

5. **Add to Inventory** as a decorator array in the callable entry

## Inventory Output Format

Markers appear in the inventory as an array under the `markers` field:

**Operation marker:**
```yaml
- id: U1234567890_M001
  kind: callable
  name: to_mapping
  signature: 'to_mapping(self) -> dict[str, Any]'
  decorators:
    - name: MechanicalOperation
      kwargs:
        type: serialization
        alias: ''
        comment: 'For JSON/YAML serialization'
  callable:
    branches: [...]
```

**Reachability marker:**
```yaml
- id: U1234567890_F001
  kind: function
  name: resolve_command
  signature_info:
    decorators:
      - name: ExternalApiMethod
        kwargs:
          boundary: cli
          comment: CLI entry point for dependency resolution
```

**Feature flow markers:**
```yaml
- id: C001_M001
  kind: callable
  name: resolve
  signature: 'resolve(params: ResolutionParams) -> ResolutionResult'
  decorators:
    - name: FeatureStart
      kwargs:
        name: dependency_resolution
        variants: ''
        comment: 'Main resolution orchestration'
  callable:
    branches: [...]
```

**Multiple markers on the same statement:**
```yaml
- id: F005
  kind: callable
  name: open_repository
  signature: 'open_repository(repo_id: str, config: dict) -> Repository'
  decorators:
    - name: FeatureTrace
      kwargs:
        name: dependency_resolution
        branch_name: ''
        comment: ''
    - name: FeatureTrace
      kwargs:
        name: package_installation
        branch_name: ''
        comment: ''
  callable:
    branches: [...]
```

**Branching markers:**
```yaml
- id: C002_M005
  kind: callable
  name: get_artifact
  signature: 'get_artifact(key: str) -> Artifact'
  decorators:
    - name: FeatureBranch
      kwargs:
        name: artifact_retrieval
        branches: cache_hit,cache_miss
        comment: 'Split based on cache status'
  callable:
    branches: [...]
```

**Convergence markers:**
```yaml
- id: C003_M008
  kind: callable
  name: _process_result
  signature: '_process_result(result: ValidationResult) -> ProcessedResult'
  decorators:
    - name: FeatureConverge
      kwargs:
        name: validation
        branches: fast_path,slow_path
        into: ''
        comment: 'Paths converge back to main'
  callable:
    branches: [...]
```

## Language-Specific Examples

### Python – Regular Comment
```python
# :: MechanicalOperation | type=serialization | comment="For JSON/YAML"
def to_mapping(self) -> dict[str, Any]:
    """Convert package to dictionary."""
    return {"name": self.name, "version": str(self.version)}
```

### Python – Inside Docstring
```python
def validate_schema(data: dict) -> bool:
    """
    :: MechanicalOperation | type=validation
    
    Validate data against schema.
    """
    return MetadataSchema().validate(data)
```

### Python – Multiple Markers
```python
def open_repository(repo_id: str, config: dict) -> Repository:
    """Open repository - shared by multiple features."""

    # :: FeatureTrace | name=dependency_resolution
    # :: FeatureTrace | name=package_installation
    return RepositoryFactory.create(repo_id, config)
```

### Java – Single Line Comment
```java
// :: MechanicalOperation | type=serialization
public Map<String, Object> toMapping() {
    return Map.of("name", this.name);
}
```

### Java – Inside Javadoc
```java
/**
 * :: UtilityOperation | type=logging | alias=audit_security_event
 * 
 * Log security event for compliance.
 * @param event Security event
 */
public void logSecurityEvent(SecurityEvent event) {
    auditLogger.critical(event.toMap());
}
```

### Java - Multi-line Comment
```java
/*
 * :: MechanicalOperation | type=validation | alias=validate_security_policy
 */
public boolean validatePolicy(Map<String, Object> policy) {
    return checkPolicy(policy);
}
```

### C++
```cpp
// :: MechanicalOperation | type=serialization
std::map<std::string, std::string> toMapping() {
    return {{"name", this->name}};
}
```

### Rust
```rust
/// :: UtilityOperation | type=caching
/// Get cached value if available
pub fn get_cached(&self, key: &str) -> Option<String> {
    self.cache.get(key)
}
```

## Configuration

### Flow Exclusion Overrides

By default, all functions marked with MechanicalOperation or UtilityOperation
are excluded from integration flow tracing (replaced with mocks or fixtures).

To include specific operations in flows, add overrides to
`integration_config.toml`:

```toml
[flow_exclusion_overrides]
# Include by type
include_mechanical_types = ["validation"]
include_utility_types = ["audit"]

# Include by name (uses alias if present, otherwise function name)
include_mechanical_names = [
    "validate_security_policy",
    "audit_security_event"
]
```

## Notes for Implementation

- **Comment parsing**: Handle all comment styles for the target language
- **Whitespace tolerance**: Strip whitespace around delimiters and field values
- **Quote handling**: Remove surrounding quotes (`"` and `'`) from values
- **Optional fields**: All fields except those marked required may be absent
- **Case-sensitive names**: Marker names must match exactly
- **Multiple markers**: A function may have multiple markers specified
- **Default behavior**: Functions without markers are treated as business
  logic and included in flow tracing
- **Field validation**: Validate required fields and strict field values
  against the marker schema
- **Co-occurrence validation**: Validate codebase-wide rules for feature flow
  markers after all files are processed