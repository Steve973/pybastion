# Operation Metadata Specification

## Purpose

Mark functions/methods with metadata to control integration flow tracing and 
feature flow identification:

1. **Operation Types** (`MechanicalOperation`, `UtilityOperation`) - Mark 
   functions to be excluded from flow tracing (replaced with mocks/fixtures)
2. **Feature Flows** (`FeatureFlow`) - Mark boundaries and waypoints of 
   behavior flows for end-to-end testing

## Format

Place this pattern in any comment immediately before a function/method:
```
:: MetadataType | field=VALUE | field=VALUE
```

All metadata types follow pipe-delimited key-value format with whitespace 
tolerance.

## Operation Types: Mechanical and Utility Operations

### Format
```
:: OperationType | type=TYPE | alias=NAME | comment="DESCRIPTION"
```

**Fields:**
- `OperationType`: Either `MechanicalOperation` or `UtilityOperation` 
  (required)
- `type`: Operation category (required)
- `alias`: Alternative name for configuration matching (optional, defaults 
  to function name)
- `comment`: Human-readable description (optional)

By default, all marked operations are excluded from integration flow tracing 
(replaced with mocks/fixtures). Configuration can override this to include 
specific operations.

## Operation Types

### MechanicalOperation Types
- `serialization` - Convert objects to external format (JSON, dict, etc.)
- `deserialization` - Convert external format to objects
- `validation` - Schema/format checking without business logic
- `formatting` - String formatting, output rendering
- `conversion` - Type conversion, coercion
- `data_transform` - Pure data reshaping without decisions
- `presentation` - UI/display formatting

### UtilityOperation Types
- `logging` - Log statements, debug output
- `caching` - Cache get/set operations
- `config` - Configuration reading
- `observability` - Metrics, tracing, monitoring
- `audit` - Audit trail logging
- `data_structure` - Generic collection operations (flatten, sort, group)

## Feature Flows: Behavior Flow Boundaries

### Format
```
:: FeatureFlow | type=MARKER_TYPE | name=FLOW_NAME | variants=VARIANT_LIST | comment="DESCRIPTION"
```

**Fields:**
- `type`: Flow marker type (required)
  - `feature_start` - Entry point of the feature flow
  - `feature_end` - Exit point where the feature flow completes
  - `feature_breadcrumb` - Optional waypoint for ambiguous paths
- `name`: Unique identifier for this feature flow (required)
- `variants`: Comma-separated list of test variants (optional, only for `feature_start`)
  - Used to generate multiple test configurations for the same flow
  - Examples: `index_resolver,builder_resolver,hybrid_fallback`
- `comment`: Human-readable description (optional)

### Purpose

Feature flows mark the boundaries of meaningful behavior sequences that span 
multiple integration points. Use them to:

1. **Document behavior orchestration** - Make feature paths visible in code
2. **Guide automated flow tracing** - Help tools trace execution from start 
   to end
3. **Generate end-to-end tests** - Create tests that exercise complete 
   feature flows

### Marker Types

**`feature_start`** (required)
- Marks the entry point where a feature flow begins
- Typically placed on public API methods or internal orchestrators
- Required: Every feature flow must have exactly one start

**`feature_end`** (required)
- Marks the exit point where a feature flow completes
- Can be placed on functions that produce final output, return results, or 
  complete the orchestration
- Required: Every feature flow must have at least one end (multiple ends 
  for different completion paths)

**`feature_breadcrumb`** (optional)
- Waypoint marker for ambiguous or complex flows
- Add only when automatic flow tracing cannot determine the path
- Use sparingly - only where needed for disambiguation
- Multiple breadcrumbs can exist for a single flow

### When to Use Breadcrumbs

Add `feature_breadcrumb` markers when:
- Execution paths fork and multiple branches could lead to the end
- Shared functions are called by multiple features
- Complex conditional logic makes the path unclear
- Flow validation tools report ambiguous paths

Run flow validation scripts to identify where breadcrumbs are needed rather 
than adding them preemptively.

### Examples

**Simple flow with just start and end:**
```python
# :: FeatureFlow | type=feature_start | name=dependency_resolution
def resolve(params: ResolutionParams) -> ResolutionResult:
    """Resolve package dependencies."""
    configs = _normalize_strategy_configs(params.strategy_configs)
    repo = open_repository(params.repo_id, params.repo_config)
    services = load_services(repo, configs)
    result = rl_resolve(services, params.environments, roots)
    return _format_result(result)

# :: FeatureFlow | type=feature_end | name=dependency_resolution
def _format_result(result: ResolveResult) -> ResolutionResult:
    """Format resolution result for output."""
    return ResolutionResult(
        requirements_by_env=_extract_requirements(result),
        resolved_wheels_by_env=_extract_wheels(result)
    )
```

**Flow with test variants:**
```python
# :: FeatureFlow | type=feature_start | name=resolution | variants=index_resolver,builder_resolver,hybrid_fallback
def resolve(services: ResolutionServices, env: ResolutionEnv, roots: list) -> Result:
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

# :: FeatureFlow | type=feature_end | name=resolution
def _create_result(mapping: dict) -> ResolutionResult:
    """Create final resolution result."""
    return ResolutionResult(requirements=mapping)
```

**Flow with breadcrumb for disambiguation:**
```python
# :: FeatureFlow | type=feature_start | name=cache_miss_handling
def get_artifact(key: str) -> Artifact:
    """Retrieve artifact with cache fallback."""
    cached = _check_cache(key)
    if cached:
        return cached
    return _fetch_from_remote(key)

# :: FeatureFlow | type=feature_breadcrumb | name=cache_miss_handling
def _fetch_from_remote(key: str) -> Artifact:
    """Breadcrumb: confirms we took the cache miss path."""
    artifact = remote_repository.fetch(key)
    _update_cache(key, artifact)
    return artifact

# :: FeatureFlow | type=feature_end | name=cache_miss_handling
def _update_cache(key: str, artifact: Artifact) -> None:
    """Update cache after remote fetch."""
    cache.set(key, artifact)
```

**Multiple ends for different completion paths:**
```python
# :: FeatureFlow | type=feature_start | name=validation_pipeline
def validate_config(config: dict) -> ValidationResult:
    """Validate configuration through pipeline."""
    if not _check_schema(config):
        return _fail_early(config)
    
    result = _validate_business_rules(config)
    if result.has_errors:
        return _format_errors(result)
    
    return _success_result(result)

# :: FeatureFlow | type=feature_end | name=validation_pipeline
def _fail_early(config: dict) -> ValidationResult:
    """Early failure path - schema invalid."""
    return ValidationResult.failure("Schema validation failed")

# :: FeatureFlow | type=feature_end | name=validation_pipeline  
def _format_errors(result: ValidationResult) -> ValidationResult:
    """Business rule failure path."""
    return result.with_formatted_messages()

# :: FeatureFlow | type=feature_end | name=validation_pipeline
def _success_result(result: ValidationResult) -> ValidationResult:
    """Success path."""
    return result.with_success_metadata()
```

**Shared function with multiple flow memberships:**
```python
# :: FeatureFlow | type=feature_breadcrumb | name=dependency_resolution
# :: FeatureFlow | type=feature_breadcrumb | name=package_installation
def open_repository(repo_id: str, config: dict) -> Repository:
    """
    Open repository - used by multiple features.
    
    Breadcrumb for both dependency_resolution and package_installation flows.
    """
    return RepositoryFactory.create(repo_id, config)
```

## Examples by Language

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

### Java – Single Line Comment
```java
// :: MechanicalOperation | type=serialization
public Map<String, Object> toMapping() {
    return Map.of("name", this.name);
}
```

### Java - Inside Javadoc
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
 * :: MechanicalOperation | type=validation
 * :: alias=validate_security_policy
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

## Extraction During Ledger Generation

When analyzing a callable during ledger generation:

1. **Collect all comments** immediately preceding the function/method 
   signature
   - Look backwards from function line
   - Include: single-line comments (`//`, `#`), multi-line comments 
     (`/* */`), docstrings (`"""`, `'''`), doc comments (`///`, `/**`)
   - Stop at blank lines or code

2. **Search for metadata patterns**:
   - `:: MechanicalOperation`
   - `:: UtilityOperation`
   - `:: FeatureFlow`

3. **Extract matching lines**

4. **Parse fields**:
   - Split on `|` delimiter
   - First segment is metadata type
   - Remaining segments are `key=value` pairs
   - Remove surrounding quotes from values

5. **Add to ledger** as decorators:

**Operation metadata:**
```yaml
decorators:
  - name: MechanicalOperation
    kwargs:
      type: serialization
      alias: convert_to_dict
      comment: For JSON/YAML serialization
```

**Feature flow metadata:**
```yaml
decorators:
  - name: FeatureFlow
    kwargs:
      type: feature_start
      name: dependency_resolution
      comment: Main resolution orchestration
```

## Configuration: Flow Exclusion Overrides

By default, ALL `MechanicalOperation` and `UtilityOperation` are excluded 
from integration flow tracing (replaced with mocks/fixtures).

To include specific operations in flows, add to `integration_config.toml`:
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

## Decision Guide: When to Mark

### Mark as MechanicalOperation
- Pure data transformations (same input → same output)
- Schema/format validation without business rules
- Serialization/deserialization
- Type conversion
- String formatting for display

### Mark as UtilityOperation
- Logging, metrics, tracing
- Caching operations
- Configuration reading
- Infrastructure plumbing

### Do NOT Mark (Business Logic)
- Decision making based on business rules
- Orchestration/coordination between units
- State changes
- Domain logic
- Anything that would be interesting to trace in integration tests

## Examples: What to Mark vs Not Mark
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

## Ledger Output

When metadata is found, it appears in the ledger's `decorators` field:

**Operation metadata:**
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

**Feature flow metadata:**
```yaml
- id: C001M001
  kind: callable  
  name: resolve
  signature: 'resolve(params: ResolutionParams) -> ResolutionResult'
  decorators:
    - name: FeatureFlow
      kwargs:
        type: feature_start
        name: dependency_resolution
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
    - name: FeatureFlow
      kwargs:
        type: feature_breadcrumb
        name: dependency_resolution
        comment: ''
    - name: FeatureFlow
      kwargs:
        type: feature_breadcrumb
        name: package_installation
        comment: ''
  callable:
    branches: [...]
```

## Notes for AI Implementation

- **Flexible comment parsing**: Handle all comment styles for the target 
  language
- **Whitespace tolerant**: Strip whitespace around delimiters
- **Quote handling**: Remove surrounding quotes from values
- **Optional fields**: `alias` and `comment` may be absent
- **Multi-line**: Pattern may span multiple comment lines
- **Case-sensitive**: Metadata type names are exact (`MechanicalOperation`, 
  `UtilityOperation`, `FeatureFlow`)
- **Multiple decorators**: A function can have multiple metadata annotations
  - Multiple `FeatureFlow` markers for shared functions in different flows
  - Can combine operation and flow metadata on same function
- **Default behavior**: Absence of metadata means business logic (include 
  in flows)