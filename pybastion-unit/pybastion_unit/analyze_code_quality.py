#!/usr/bin/env python3
"""
Code Quality Analyzer

Runs configured analysis tools against Python source files and generates
quality metrics with grades. Output can be merged into unit ledger reviews.

Usage:
    python analyze_code_quality.py <source_file> [--config quality_config.toml] [--output metrics.yaml]
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomli
except ImportError:
    import tomllib as tomli  # Python 3.11+

import yaml


@dataclass
class QualityMetric:
    """Individual quality metric with value and grade"""
    name: str
    value: Any
    grade: str  # excellent, good, fair, poor, critical
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Complete quality analysis report for a source file"""
    source_file: str
    overall_grade: str
    metrics: list[QualityMetric] = field(default_factory=list)
    flagged_callables: list[dict[str, Any]] = field(default_factory=list)
    raw_analyzer_output: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML/JSON output"""
        result = {
            "sourceFile": self.source_file,
            "overallGrade": self.overall_grade,
            "metrics": {},
            "flaggedCallables": self.flagged_callables,
        }

        for metric in self.metrics:
            result["metrics"][metric.name] = {
                "value": metric.value,
                "grade": metric.grade,
                **metric.details
            }

        if self.raw_analyzer_output:
            result["rawAnalyzerOutput"] = self.raw_analyzer_output

        return result


class QualityAnalyzer:
    """Runs quality analysis tools and generates graded metrics"""

    GRADE_ORDER = ["excellent", "good", "fair", "poor", "critical"]

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()
        self.thresholds = self.config.get("thresholds", {})
        self.enabled_analyzers = self.config["analyzers"]["enabled"]

    def _load_config(self) -> dict[str, Any]:
        """Load TOML configuration"""
        with open(self.config_path, "rb") as f:
            return tomli.load(f)

    def analyze(self, source_file: Path) -> QualityReport:
        """Run all enabled analyzers and generate quality report"""
        report = QualityReport(source_file=str(source_file), overall_grade="unknown")

        # Run each enabled analyzer
        for analyzer_name in self.enabled_analyzers:
            try:
                if analyzer_name == "radon_complexity":
                    self._analyze_radon_complexity(source_file, report)
                elif analyzer_name == "radon_maintainability":
                    self._analyze_radon_maintainability(source_file, report)
                elif analyzer_name == "radon_raw":
                    self._analyze_radon_raw(source_file, report)
                elif analyzer_name == "type_hints":
                    self._analyze_type_hints(source_file, report)
                elif analyzer_name == "vulture":
                    self._analyze_vulture(source_file, report)
            except Exception as e:
                print(f"Warning: {analyzer_name} failed: {e}", file=sys.stderr)

        # Determine overall grade (worst metric grade) after all metrics collected
        report.overall_grade = self._determine_overall_grade(report.metrics)

        return report

    def _analyze_radon_complexity(self, source_file: Path, report: QualityReport):
        """Analyze cyclomatic complexity using radon"""
        result = subprocess.run(
            ["radon", "cc", "-s", "-j", str(source_file)],
            capture_output=True,
            text=True
        )

        print(f"radon complexity output for '{source_file}': {result.stdout}")

        if result.returncode != 0:
            raise RuntimeError(f"radon cc failed: {result.stderr}")

        data = json.loads(result.stdout)
        complexities = []
        flagged = []

        for file_data in data.values():
            for item in file_data:
                complexity = item["complexity"]
                complexities.append(complexity)

                grade = self._grade_value("cyclomatic_complexity", complexity)
                if grade in ["poor", "critical"]:
                    flagged.append({
                        "callable": item["name"],
                        "complexity": complexity,
                        "grade": grade,
                        "line": item["lineno"]
                    })

        if complexities:
            avg_complexity = sum(complexities) / len(complexities)
            max_complexity = max(complexities)

            metric = QualityMetric(
                name="cyclomaticComplexity",
                value={
                    "average": round(avg_complexity, 1),
                    "max": max_complexity,
                    "count": len(complexities)
                },
                grade=self._grade_value("cyclomatic_complexity", max_complexity),
                details={"flagged": flagged} if flagged else {}
            )
            report.metrics.append(metric)
            report.flagged_callables.extend(flagged)

            if self.config["output"].get("include_raw_data"):
                report.raw_analyzer_output["radon_complexity"] = data

    def _analyze_radon_maintainability(self, source_file: Path, report: QualityReport):
        """Analyze maintainability index using radon"""
        result = subprocess.run(
            ["radon", "mi", "-s", "-j", str(source_file)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"radon mi failed: {result.stderr}")

        data = json.loads(result.stdout)

        # Radon MI output is per-file
        for file_data in data.values():
            mi_score = file_data["mi"]
            mi_rank = file_data["rank"]  # A, B, C

            metric = QualityMetric(
                name="maintainabilityIndex",
                value={
                    "score": round(mi_score, 1),
                    "rank": mi_rank
                },
                grade=self._grade_value("maintainability_index", mi_score)
            )
            report.metrics.append(metric)

            if self.config["output"].get("include_raw_data"):
                report.raw_analyzer_output["radon_maintainability"] = data

    def _analyze_radon_raw(self, source_file: Path, report: QualityReport):
        """Analyze raw metrics (LOC, LLOC, etc.) using radon"""
        result = subprocess.run(
            ["radon", "raw", "-s", "-j", str(source_file)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"radon raw failed: {result.stderr}")

        data = json.loads(result.stdout)

        for file_data in data.values():
            # File-level metrics
            loc = file_data["loc"]
            lloc = file_data["lloc"]
            sloc = file_data["sloc"]
            comments = file_data["comments"]

            # Comment ratio
            comment_ratio = (comments / sloc * 100) if sloc > 0 else 0

            metric = QualityMetric(
                name="rawMetrics",
                value={
                    "loc": loc,
                    "lloc": lloc,
                    "sloc": sloc,
                    "comments": comments,
                    "commentRatio": round(comment_ratio, 1)
                },
                grade="excellent" # show ratio but do not penalize
            )
            report.metrics.append(metric)

            # Analyze per-function metrics
            function_lengths = []
            flagged_functions = []

            for func in file_data.get("functions", []):
                func_lloc = func["lloc"]
                function_lengths.append(func_lloc)

                grade = self._grade_value("function_length", func_lloc)
                if grade in ["poor", "critical"]:
                    flagged_functions.append({
                        "callable": func["name"],
                        "length": func_lloc,
                        "grade": grade,
                        "line": func["lineno"]
                    })

            if function_lengths:
                avg_length = sum(function_lengths) / len(function_lengths)
                max_length = max(function_lengths)

                length_metric = QualityMetric(
                    name="functionLength",
                    value={
                        "average": round(avg_length, 1),
                        "max": max_length,
                        "count": len(function_lengths)
                    },
                    grade=self._grade_value("function_length", max_length),
                    details={"flagged": flagged_functions} if flagged_functions else {}
                )
                report.metrics.append(length_metric)
                report.flagged_callables.extend(flagged_functions)

            if self.config["output"].get("include_raw_data"):
                report.raw_analyzer_output["radon_raw"] = data

    def _analyze_type_hints(self, source_file: Path, report: QualityReport):
        """Analyze type hint coverage using mypy"""
        # Run mypy with strict type checking
        result = subprocess.run(
            ["mypy", "--strict", "--no-error-summary", str(source_file)],
            capture_output=True,
            text=True
        )

        # Parse mypy output to count type-related errors
        # This is a simplified approach - might need refinement
        lines = result.stderr.split('\n') if result.stderr else []
        type_errors = [l for l in lines if "type" in l.lower() or "annotation" in l.lower()]

        # Rough estimate: count function defs and compare to type errors
        # A proper implementation would use AST parsing
        with open(source_file) as f:
            source = f.read()
            func_count = source.count("def ")

        if func_count > 0:
            # Very rough estimate
            type_error_count = len(type_errors)
            estimated_coverage = max(0.0, 100 - (type_error_count / func_count * 100))

            metric = QualityMetric(
                name="typeHintCoverage",
                value={
                    "estimatedPercentage": round(estimated_coverage, 1),
                    "typeErrors": type_error_count
                },
                grade=self._grade_value("type_coverage", estimated_coverage),
                details={
                    "note": "Estimated from mypy strict mode errors"
                }
            )
            report.metrics.append(metric)

            if self.config["output"].get("include_raw_data"):
                report.raw_analyzer_output["mypy"] = {
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }

    def _analyze_vulture(self, source_file: Path, report: QualityReport):
        """Detect dead code using vulture"""
        min_confidence = self.config["analyzers"]["vulture"]["min_confidence"]

        result = subprocess.run(
            ["vulture", str(source_file), f"--min-confidence={min_confidence}"],
            capture_output=True,
            text=True
        )

        # Vulture returns non-zero if it finds dead code
        dead_code_items = []
        if result.stdout:
            for line in result.stdout.split('\n'):
                if line.strip():
                    dead_code_items.append(line.strip())

        metric = QualityMetric(
            name="deadCode",
            value={
                "itemsFound": len(dead_code_items),
                "minConfidence": min_confidence
            },
            grade="critical" if dead_code_items else "excellent",
            details={"items": dead_code_items} if dead_code_items else {}
        )
        report.metrics.append(metric)

        if self.config["output"].get("include_raw_data"):
            report.raw_analyzer_output["vulture"] = {
                "stdout": result.stdout,
                "returncode": result.returncode
            }

    def _grade_value(self, threshold_key: str, value: float) -> str:
        """Determine grade based on configured thresholds"""
        thresholds = self.thresholds.get(threshold_key, {})

        if not thresholds:
            return "unknown"

        # For metrics where higher is better (maintainability, type coverage, comments)
        if threshold_key in ["type_coverage", "comment_ratio"]:
            if value >= thresholds.get("excellent", 100):
                return "excellent"
            elif value >= thresholds.get("good", 75):
                return "good"
            elif value >= thresholds.get("fair", 50):
                return "fair"
            elif value >= thresholds.get("poor", 25):
                return "poor"
            else:
                return "critical"
        elif threshold_key == "maintainability_index":
            if value >= thresholds.get("excellent", 20):
                return "excellent"
            elif value >= thresholds.get("good", 15):
                return "good"
            elif value >= thresholds.get("fair", 10):
                return "fair"
            elif value >= thresholds.get("poor", 5):
                return "poor"
            else:
                return "critical"
        else:
            # For metrics where lower is better (complexity, length, nesting, params)
            if value <= thresholds.get("excellent", 0):
                return "excellent"
            elif value <= thresholds.get("good", 5):
                return "good"
            elif value <= thresholds.get("fair", 10):
                return "fair"
            elif value <= thresholds.get("poor", 15):
                return "poor"
            else:
                return "critical"

    def _determine_overall_grade(self, metrics: list[QualityMetric]) -> str:
        """Determine overall grade as worst individual metric grade"""
        if not metrics:
            return "unknown"

        grades = [m.grade for m in metrics]

        # Return worst grade
        for grade in reversed(self.GRADE_ORDER):
            if grade in grades:
                return grade

        return "unknown"


def main():
    # Get script directory for default config path
    script_dir = Path(__file__).parent
    default_config = script_dir / "quality_config.toml"

    parser = argparse.ArgumentParser(
        description="Analyze Python code quality and generate graded metrics"
    )
    parser.add_argument(
        "source_file",
        type=Path,
        help="Python source file to analyze"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"Path to quality configuration file (default: {default_config})"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--format",
        choices=["yaml", "json"],
        help="Output format (overrides config)"
    )

    args = parser.parse_args()

    if not args.source_file.exists():
        print(f"Error: Source file not found: {args.source_file}", file=sys.stderr)
        sys.exit(1)

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Run analysis
    analyzer = QualityAnalyzer(args.config)
    report = analyzer.analyze(args.source_file)

    # Determine output format
    output_format = args.format or analyzer.config["output"]["format"]

    # Generate output
    report_dict = report.to_dict()

    if output_format == "json":
        output = json.dumps(report_dict, indent=2)
    else:  # yaml
        output = yaml.dump(report_dict, sort_keys=False, default_flow_style=False)

    # Write output
    if args.output:
        args.output.write_text(output)
        print(f"Quality report written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()