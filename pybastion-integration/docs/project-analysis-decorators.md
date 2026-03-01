# Project Analysis Decorators Specification

## Overview

This specification defines comment-based decorators for marking functions and
methods with metadata that controls integration flow tracing and feature flow
identification. These decorators serve two primary purposes:

1. **Operation Decorators** - Mark mechanical and utility operations to be
   excluded from flow tracing (replaced with mocks/fixtures)
2. **Feature Flow Decorators** - Define boundaries, waypoints, branching, and
   convergence points for end-to-end feature testing

All decorators use a pipe-delimited syntax embedded in comments immediately
preceding the function or method definition.

## Decorator Syntax

### General Format

```
:: DecoratorName | field=value | field=value
```

**Rules:**
- Decorator name is case-sensitive
- Fields are pipe-delimited (`|`)
- Field values use `key=value` format
- Whitespace around delimiters is ignored
- Quote marks around values are stripped during parsing
- Can appear in any comment style supported by the language

### Placement

Place decorators in comments **immediately before** the function or method
signature:

```python
# :: MechanicalOperation | type=serialization
def to_dict(self) -> dict:
    return {"name": self.name}
```

Or inside docstrings:

```python
def validate_schema(data: dict) -> bool:
    """
    :: MechanicalOperation | type=validation
    
    Validate data against schema.
    """
    return schema.validate(data)
```

## Operation Decorators

Operation decorators mark functions that should be excluded from integration
flow tracing. These are mechanical transformations or utility operations that
do not contain business logic.

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

### When to Use Operation Decorators

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

## Feature Flow Decorators

Feature flow decorators define the boundaries and structure of end-to-end
feature flows that span multiple integration points. They enable automated
tracing of execution paths from feature entry to exit, including branching
and convergence.

### Overview of Feature Flow Model

A feature flow consists of:
- **One entry point** (FeatureStart) – where the feature begins
- **Zero or more waypoints** (FeatureTrace) – intermediate steps
- **Zero or more branch points** (FeatureBranch) – where execution splits
- **Zero or more convergence points** (FeatureConverge) – where branches merge
- **One or more exit points** (FeatureEnd, FeatureEndConditional) – where the
  feature completes

All decorators in a feature flow are correlated by the `name` field. Branches
within a feature flow are correlated by the `branch_name` field.

### FeatureStart

Marks the entry point where a feature flow begins.

**Format:**
```
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

**Example:**
```python
# :: FeatureStart | name=dependency_resolution
def resolve(params: ResolutionParams) -> ResolutionResult:
    """Resolve package dependencies."""
    configs = _normalize_strategy_configs(params.strategy_configs)
    repo = open_repository(params.repo_id, params.repo_config)
    services = load_services(repo, configs)
    result = rl_resolve(services, params.environments, roots)
    return _format_result(result)
```

**Example with test variants:**
```python
# :: FeatureStart | name=resolution
# :: | variants=index_resolver,builder_resolver,hybrid_fallback
def resolve(services: ResolutionServices, env: Env, roots: list) -> Result:
    """
    Main resolution entry point.
    
    Test variants configure different resolver strategies:
    - index_resolver: Use PyPI index-based resolution
    - builder_resolver: Build from sdist when needed
    - hybrid_fallback: Try index first, fall back to builder
    """
    provider = ProjectResolutionProvider(services=services, env=env)
    reporter = ProjectResolutionReporter()
    resolver = Resolver(provider, reporter)
    return resolver.resolve(roots)
```

### FeatureTrace

Marks a waypoint along a feature flow path. Used to track intermediate steps
or disambiguate complex flows.

**Format:**
```
:: FeatureTrace | name=FEATURE_NAME | branch_name=BRANCH | comment="DESC"
```

**Fields:**

| Field       | Required | Description                                    |
|-------------|----------|------------------------------------------------|
| name        | Yes      | Feature flow identifier (matches FeatureStart) |
| branch_name | No       | Branch identifier if on a named branch         |
| comment     | No       | Human-readable description                     |

**When to use:**
- To mark intermediate steps in a complex feature flow
- To disambiguate paths when automatic tracing is unclear
- To mark where a named branch continues after branching

**Example – Simple waypoint:**
```python
# :: FeatureTrace | name=dependency_resolution
def open_repository(repo_id: str, config: dict) -> Repository:
    """Open repository for dependency resolution."""
    return RepositoryFactory.create(repo_id, config)
```

**Example – Branch waypoint:**
```python
# :: FeatureTrace | name=resolution | branch_name=remote_fetch
def fetch_from_pypi(package_name: str) -> Artifact:
    """Fetch package from PyPI - remote fetch branch."""
    return pypi_client.fetch(package_name)
```

**Example – Multiple feature flows:**
```python
# :: FeatureTrace | name=dependency_resolution
# :: FeatureTrace | name=package_installation
def open_repository(repo_id: str, config: dict) -> Repository:
    """
    Open repository - used by multiple features.
    
    Waypoint for both dependency_resolution and package_installation flows.
    """
    return RepositoryFactory.create(repo_id, config)
```

### FeatureBranch

Marks a decision point where a feature flow splits into multiple named paths.

**Format:**
```
:: FeatureBranch | name=FEATURE_NAME | branches=BRANCH_LIST | comment="DESC"
```

**Fields:**

| Field    | Required | Description                                    |
|----------|----------|------------------------------------------------|
| name     | Yes      | Feature flow identifier (matches FeatureStart) |
| branches | Yes      | Comma-separated list of branch names           |
| comment  | No       | Human-readable description                     |

**Branch lifecycle:**
1. FeatureBranch declares branch names (e.g., `path_a,path_b`)
2. Each branch continues via FeatureTrace with `branch_name` set
3. Each branch must eventually:
   - Reach a FeatureEnd or FeatureEndConditional with that `branch_name`, OR
   - Participate in a FeatureConverge that retires the branch

**Example:**
```python
# :: FeatureBranch | name=artifact_retrieval | branches=cache_hit,cache_miss
def get_artifact(key: str) -> Artifact:
    """Retrieve artifact with cache fallback."""
    cached = _check_cache(key)
    if cached:
        return cached  # Goes to cache_hit branch
    return _fetch_from_remote(key)  # Goes to cache_miss branch

# :: FeatureTrace | name=artifact_retrieval | branch_name=cache_hit
def _return_cached(artifact: Artifact) -> Artifact:
    """Cache hit path."""
    return artifact

# :: FeatureTrace | name=artifact_retrieval | branch_name=cache_miss
def _fetch_from_remote(key: str) -> Artifact:
    """Cache miss path - fetch from remote."""
    artifact = remote_repository.fetch(key)
    _update_cache(key, artifact)
    return artifact
```

### FeatureConverge

Marks a point where multiple branches merge back together.

**Format:**
```
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

**Example – Converge to main:**
```python
# :: FeatureBranch | name=validation | branches=fast_path,slow_path
def validate_input(data: dict) -> ValidationResult:
    """Validate input data."""
    if _is_simple_case(data):
        return _fast_validate(data)
    return _slow_validate(data)

# :: FeatureTrace | name=validation | branch_name=fast_path
def _fast_validate(data: dict) -> ValidationResult:
    """Fast validation path."""
    return ValidationResult(valid=True)

# :: FeatureTrace | name=validation | branch_name=slow_path
def _slow_validate(data: dict) -> ValidationResult:
    """Slow validation path with full checks."""
    return ValidationResult(valid=_check_all_rules(data))

# :: FeatureConverge | name=validation | branches=fast_path,slow_path
def _process_result(result: ValidationResult) -> ProcessedResult:
    """Both paths converge here - back to main flow."""
    return ProcessedResult(result)
```

**Example – Nested branching (converge to parent branch):**
```python
# :: FeatureBranch | name=resolution | branches=local,remote
def resolve_package(name: str) -> Package:
    """Resolve package from local or remote."""
    if _check_local_cache(name):
        return _resolve_local(name)
    return _resolve_remote(name)

# :: FeatureTrace | name=resolution | branch_name=remote
# :: FeatureBranch | name=resolution | branches=pypi,conda
def _resolve_remote(name: str) -> Package:
    """Remote resolution - branches to PyPI or Conda."""
    if _is_conda_package(name):
        return _fetch_conda(name)
    return _fetch_pypi(name)

# :: FeatureTrace | name=resolution | branch_name=pypi
def _fetch_pypi(name: str) -> Package:
    """Fetch from PyPI."""
    return pypi.fetch(name)

# :: FeatureTrace | name=resolution | branch_name=conda
def _fetch_conda(name: str) -> Package:
    """Fetch from Conda."""
    return conda.fetch(name)

# :: FeatureConverge | name=resolution | branches=pypi,conda | into=remote
def _cache_remote_package(pkg: Package) -> Package:
    """Sub-branches converge back to remote branch."""
    _update_cache(pkg)
    return pkg

# :: FeatureConverge | name=resolution | branches=local,remote
def _finalize_resolution(pkg: Package) -> Package:
    """Both local and remote converge back to main."""
    return _validate_package(pkg)
```

### FeatureEnd

Marks a definitive exit point where a feature flow completes.

**Format:**
```
:: FeatureEnd | name=FEATURE_NAME | branch_name=BRANCH | comment="DESC"
```

**Fields:**

| Field       | Required | Description                                    |
|-------------|----------|------------------------------------------------|
| name        | Yes      | Feature flow identifier (matches FeatureStart) |
| branch_name | No       | Branch identifier if ending a named branch     |
| comment     | No       | Human-readable description                     |

**Constraints:**
- Each feature flow must have at least one FeatureEnd or FeatureEndConditional
- If multiple ends exist, they represent different completion paths
- If on a named branch, must include `branch_name`

**Example – Single end:**
```python
# :: FeatureStart | name=dependency_resolution
def resolve(params: ResolutionParams) -> ResolutionResult:
    """Resolve package dependencies."""
    # ... resolution logic ...
    return _format_result(result)

# :: FeatureEnd | name=dependency_resolution
def _format_result(result: ResolveResult) -> ResolutionResult:
    """Format resolution result for output."""
    return ResolutionResult(
        requirements_by_env=_extract_requirements(result),
        resolved_wheels_by_env=_extract_wheels(result)
    )
```

**Example – Multiple ends for different paths:**
```python
# :: FeatureStart | name=validation_pipeline
def validate_config(config: dict) -> ValidationResult:
    """Validate configuration through pipeline."""
    if not _check_schema(config):
        return _fail_early(config)
    
    result = _validate_business_rules(config)
    if result.has_errors:
        return _format_errors(result)
    
    return _success_result(result)

# :: FeatureEnd | name=validation_pipeline
def _fail_early(config: dict) -> ValidationResult:
    """Early failure path - schema invalid."""
    return ValidationResult.failure("Schema validation failed")

# :: FeatureEnd | name=validation_pipeline
def _format_errors(result: ValidationResult) -> ValidationResult:
    """Business rule failure path."""
    return result.with_formatted_messages()

# :: FeatureEnd | name=validation_pipeline
def _success_result(result: ValidationResult) -> ValidationResult:
    """Success path."""
    return result.with_success_metadata()
```

**Example - Branch-specific end:**
```python
# :: FeatureBranch | name=payment | branches=credit_card,paypal
def process_payment(method: str, amount: float) -> PaymentResult:
    """Process payment via selected method."""
    if method == "credit_card":
        return _process_credit_card(amount)
    return _process_paypal(amount)

# :: FeatureEnd | name=payment | branch_name=credit_card
def _finalize_credit_card(result: CreditResult) -> PaymentResult:
    """Credit card branch ends here."""
    return PaymentResult.from_credit(result)

# :: FeatureEnd | name=payment | branch_name=paypal
def _finalize_paypal(result: PayPalResult) -> PaymentResult:
    """PayPal branch ends here."""
    return PaymentResult.from_paypal(result)
```

### FeatureEndConditional

Marks a conditional exit point where a feature flow may complete based on
runtime conditions.

**Format:**
```
:: FeatureEndConditional | name=FEATURE_NAME | on_condition=CONDITION | branch_name=BRANCH | comment="DESC"
```

**Fields:**

| Field        | Required | Description                                     |
|--------------|----------|-------------------------------------------------|
| name         | Yes      | Feature flow identifier (matches FeatureStart)  |
| on_condition | Yes      | Description of the condition that triggers exit |
| branch_name  | No       | Branch identifier if ending a named branch      |
| comment      | No       | Human-readable description                      |

**When to use:**
- Early termination based on runtime conditions
- Error paths that exit the feature
- Optional exit points in conditional logic

**Example:**
```python
# :: FeatureStart | name=data_processing
def process_data(data: dict) -> ProcessedData:
    """Process incoming data."""
    if not _validate_input(data):
        return _abort_processing(data)  # Conditional exit
    
    result = _transform_data(data)
    return _finalize_processing(result)

# :: FeatureEndConditional | name=data_processing
# :: | on_condition=invalid_input
def _abort_processing(data: dict) -> ProcessedData:
    """Conditional exit - invalid input."""
    return ProcessedData.error("Invalid input")

# :: FeatureEnd | name=data_processing
def _finalize_processing(result: TransformResult) -> ProcessedData:
    """Normal completion path."""
    return ProcessedData.success(result)
```

## Co-occurrence Rules

### Operation Decorators

MechanicalOperation and UtilityOperation have no co-occurrence constraints.
They apply only to the function they decorate.

### Feature Flow Decorators

Feature flow decorators have codebase-wide co-occurrence rules based on the
`name` field:

**FeatureStart constraints:**
- Must appear exactly once per feature flow
- Requires at least one of:
  - FeatureEnd with matching `name`, OR
  - FeatureEndConditional with matching `name`
- May have multiple FeatureEnd or FeatureEndConditional decorators
- May have zero or more FeatureTrace, FeatureBranch, FeatureConverge

**FeatureBranch constraints:**
- Each declared branch name must eventually:
  - Appear in a FeatureConverge `branches` field, OR
  - Reach a FeatureEnd or FeatureEndConditional with that `branch_name`

**FeatureConverge constraints:**
- All branch names in `branches` must have been declared by a FeatureBranch
- If `into` is present, it must reference a currently active branch
- After convergence, the listed branch names are retired

## Extraction During Ledger Generation

When analyzing a callable during ledger generation:

1. **Collect comments** immediately preceding the function or method signature
   - Look backwards from the function line
   - Include: single-line comments (`//`, `#`), multi-line comments
     (`/* */`), docstrings (`"""`, `'''`), doc comments (`///`, `/**`)
   - Stop at blank lines or code

2. **Search for decorator patterns**:
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
   - The first segment is the decorator name
   - Remaining segments are `key=value` pairs
   - Strip whitespace around delimiters
   - Remove surrounding quotes from values

5. **Add to ledger** as decorators array in the callable entry

## Ledger Output Format

Decorators appear in the ledger as an array under the `decorators` field:

**Operation decorator:**
```yaml
- id: C001M003
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

**Feature flow decorators:**
```yaml
- id: C001M001
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

**Multiple decorators on same function:**
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

**Branching decorators:**
```yaml
- id: C002M005
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

**Convergence decorators:**
```yaml
- id: C003M008
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

### Python – Multiple Decorators
```python
# :: FeatureTrace | name=dependency_resolution
# :: FeatureTrace | name=package_installation
def open_repository(repo_id: str, config: dict) -> Repository:
    """Open repository - shared by multiple features."""
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
- **Case-sensitive names**: Decorator names must match exactly
- **Multiple decorators**: A function may have multiple decorator annotations
- **Default behavior**: Functions without decorators are treated as business
  logic and included in flow tracing
- **Field validation**: Validate required fields and strict field values
  against the decorator schema
- **Co-occurrence validation**: Validate codebase-wide rules for feature flow
  decorators after all files are processed