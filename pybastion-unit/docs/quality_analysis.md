# Code Quality Analysis System

Analyzes Python source files and generates graded quality metrics that can be merged into unit ledger reviews.

## Components

### 1. `quality_config.toml`
Configuration file that defines:
- Which analyzers to run (radon, mypy, vulture)
- Grading thresholds for each metric
- Output format preferences

### 2. `analyze_code_quality.py`
Main analyzer script that:
- Reads configuration
- Runs enabled analysis tools
- Applies threshold-based grading
- Outputs quality metrics in YAML/JSON

### 3. `quality_metrics_schema.json`
JSON schema defining the output format for validation

### 4. `example_integration.py`
Example showing how to integrate quality analysis into ledger generation workflow

## Usage

### Standalone Analysis

```bash
# Basic usage (outputs to stdout)
python analyze_code_quality.py my_module.py

# Specify config and output file
python analyze_code_quality.py my_module.py \
    --config quality_config.toml \
    --output my_module_quality.yaml

# Override output format
python analyze_code_quality.py my_module.py --format json
```

### Integrated with Ledger Generation

```python
from pathlib import Path
import subprocess
import yaml

# Generate ledger
subprocess.run(["python", "generate_ledger.py", "my_module.py"])

# Analyze quality
result = subprocess.run(
    ["python", "analyze_code_quality.py", "my_module.py"],
    capture_output=True,
    text=True
)
quality_metrics = yaml.safe_load(result.stdout)

# Check quality gates
if quality_metrics["overallGrade"] in ["poor", "critical"]:
    print("Fix your code before generating tests!")
    exit(1)

# Merge into ledger review section
# (see example_integration.py for details)
```

## Metrics Generated

### Cyclomatic Complexity
- **Source**: radon cc
- **Measures**: Number of linearly independent paths through code
- **Grades**: ≤5 (excellent), ≤10 (good), ≤15 (fair), ≤20 (poor), >20 (critical)
- **Flags**: Individual callables with poor/critical complexity

### Maintainability Index
- **Source**: radon mi
- **Measures**: Composite score of volume, complexity, and lines of code (0-100)
- **Grades**: ≥80 (excellent), ≥60 (good), ≥40 (fair), ≥20 (poor), <20 (critical)

### Function Length
- **Source**: radon raw
- **Measures**: Logical lines of code (LLOC) per callable
- **Grades**: ≤20 (excellent), ≤40 (good), ≤60 (fair), ≤80 (poor), >80 (critical)
- **Flags**: Individual callables exceeding thresholds

### Comment Ratio
- **Source**: radon raw
- **Measures**: Percentage of lines that are comments
- **Grades**: ≥20% (excellent), ≥10% (good), ≥5% (fair), ≥2% (poor), <2% (critical)

### Type Hint Coverage
- **Source**: mypy --strict
- **Measures**: Estimated percentage of parameters/returns with type hints
- **Grades**: ≥90% (excellent), ≥75% (good), ≥50% (fair), ≥25% (poor), <25% (critical)
- **Note**: Currently estimated from mypy error count; could be improved with AST parsing

### Dead Code Detection
- **Source**: vulture
- **Measures**: Unused functions, classes, variables
- **Grades**: 0 items (excellent), any items (critical)
- **Config**: Minimum confidence threshold (default 80%)

## Output Format

### YAML Example
```yaml
sourceFile: my_module.py
overallGrade: good
metrics:
  cyclomaticComplexity:
    value:
      average: 4.2
      max: 15
      count: 8
    grade: fair
    flagged:
      - callable: _allocate_path_for_key
        complexity: 15
        grade: fair
        line: 134
  maintainabilityIndex:
    value:
      score: 72.3
      rank: B
    grade: good
  functionLength:
    value:
      average: 25.4
      max: 45
      count: 8
    grade: good
  typeHintCoverage:
    value:
      estimatedPercentage: 85.0
      typeErrors: 3
    grade: good
  deadCode:
    value:
      itemsFound: 0
      minConfidence: 80
    grade: excellent
flaggedCallables:
  - callable: _allocate_path_for_key
    complexity: 15
    grade: fair
    line: 134
```

## Configuration

Edit `quality_config.toml` to customize:

```toml
[analyzers]
# Enable/disable analyzers
enabled = [
    "radon_complexity",
    "radon_maintainability",
    "type_hints",
    "vulture"
]

[thresholds.cyclomatic_complexity]
# Adjust grading thresholds
excellent = 5
good = 10
fair = 15
poor = 20

[output]
format = "yaml"  # or "json"
include_raw_data = false  # include full analyzer outputs
```

## Dependencies

```bash
pip install radon mypy vulture pyyaml tomli
```

## Quality Gates

The system can enforce quality gates before test generation:

```python
def check_quality_gates(quality_metrics: dict) -> bool:
    if quality_metrics["overallGrade"] in ["poor", "critical"]:
        print("QUALITY GATE FAILURE - Fix your code first!")
        return False
    return True
```

AI test generators can use this to refuse test generation for poor-quality code:

> "Your code has a 'poor' quality grade (cyclomatic complexity 35, no type hints). 
> Fix your shit before I waste tokens generating tests for this mess."

## Integration with Ledger Review

Quality metrics can be merged into the `ledger-generation-review` section:

```yaml
---
docKind: ledger-generation-review
schemaVersion: 1.0.0
unit:
  name: my_module
  callablesAnalyzed: 8
  totalExeItems: 98
overallQualityGrade: good  # ← added
qualityMetrics:            # ← added
  cyclomaticComplexity:
    # ... metrics here
qualityFlags:              # ← added
  - callable: _allocate_path_for_key
    complexity: 15
    grade: fair
```

## Future Enhancements

Potential additions:
- Parameter count analysis (detect functions with too many params)
- Nesting depth analysis (detect deeply nested code)
- Halstead metrics (program difficulty/effort)
- Better type hint coverage via AST parsing instead of mypy error counting
- Code duplication detection
- Security issue detection (bandit integration)