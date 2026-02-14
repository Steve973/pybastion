# Operation Metadata Specification

## Purpose

Mark functions/methods as mechanical or utility operations to control 
integration flow tracing. By default, all marked operations are excluded 
from flow tracing (replaced with mocks/fixtures). Configuration can override 
this to include specific operations.

## Format

Place this pattern in any comment immediately before a function/method:
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

## Examples by Language

### Python - Regular Comment
```python
# :: MechanicalOperation | type=serialization | comment="For JSON/YAML"
def to_mapping(self) -> dict[str, Any]:
    """Convert package to dictionary."""
    return {"name": self.name, "version": str(self.version)}
```

### Python - Inside Docstring
```python
def validate_schema(data: dict) -> bool:
    """
    :: MechanicalOperation | type=validation
    
    Validate data against schema.
    """
    return MetadataSchema().validate(data)
```

### Java - Single Line Comment
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

2. **Search for pattern** `:: MechanicalOperation` or 
   `:: UtilityOperation`

3. **Extract the line** containing this pattern

4. **Parse fields**:
   - Split on `|` delimiter
   - First segment is operation type
   - Remaining segments are `key=value` pairs
   - Remove surrounding quotes from values

5. **Add to ledger** as a decorator:
```yaml
decorators:
  - name: MechanicalOperation
    kwargs:
      type: serialization
      alias: convert_to_dict
      comment: For JSON/YAML serialization
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

When operation metadata is found, it appears in the ledger's `decorators` 
field:
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

## Notes for AI Implementation

- **Flexible comment parsing**: Handle all comment styles for the target 
  language
- **Whitespace tolerant**: Strip whitespace around delimiters
- **Quote handling**: Remove surrounding quotes from values
- **Optional fields**: `alias` and `comment` may be absent
- **Multi-line**: Pattern may span multiple comment lines
- **Case sensitive**: `MechanicalOperation` and `UtilityOperation` are 
  exact
- **Default behavior**: Absence of metadata means business logic (include 
  in flows)