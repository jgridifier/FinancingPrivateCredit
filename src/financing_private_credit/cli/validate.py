"""
Indicator validation tool.

Validates that an indicator follows framework conventions and
can be loaded and executed correctly.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ValidationResult:
    """Result of validating an indicator."""

    indicator_name: str
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    structure_checks: dict[str, bool] = field(default_factory=dict)
    registration_checks: dict[str, bool] = field(default_factory=dict)
    config_checks: dict[str, bool] = field(default_factory=dict)
    doc_checks: dict[str, Optional[bool]] = field(default_factory=dict)
    quality_checks: dict[str, Optional[bool]] = field(default_factory=dict)


def _get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root")


def _parse_indicator_module(indicator_path: Path) -> Optional[ast.Module]:
    """Parse the indicator.py file using AST."""
    indicator_file = indicator_path / "indicator.py"
    if not indicator_file.exists():
        return None

    try:
        with open(indicator_file, "r") as f:
            return ast.parse(f.read())
    except SyntaxError:
        return None


def _find_class_info(tree: ast.Module) -> dict[str, Any]:
    """Extract class information from AST."""
    info = {
        "classes": [],
        "has_register_decorator": False,
        "registered_name": None,
        "inherits_from": [],
        "methods": {},
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_info = {
                "name": node.name,
                "bases": [
                    base.id if isinstance(base, ast.Name) else
                    base.attr if isinstance(base, ast.Attribute) else
                    str(base)
                    for base in node.bases
                ],
                "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                "has_register_decorator": False,
                "registered_name": None,
            }

            # Check for @register_indicator decorator
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Name) and decorator.func.id == "register_indicator":
                        class_info["has_register_decorator"] = True
                        if decorator.args and isinstance(decorator.args[0], ast.Constant):
                            class_info["registered_name"] = decorator.args[0].value

            info["classes"].append(class_info)

    return info


def _check_indicator_structure(
    name: str,
    indicators_dir: Path,
    result: ValidationResult,
) -> bool:
    """Check the indicator package structure."""
    indicator_path = indicators_dir / name

    # Check directory exists
    if not indicator_path.exists():
        result.errors.append(f"Indicator directory not found: {indicator_path}")
        result.structure_checks["Directory exists"] = False
        return False
    result.structure_checks["Directory exists"] = True

    # Check __init__.py exists
    init_file = indicator_path / "__init__.py"
    result.structure_checks["Has __init__.py"] = init_file.exists()
    if not init_file.exists():
        result.errors.append("Missing __init__.py")

    # Check indicator.py exists
    indicator_file = indicator_path / "indicator.py"
    result.structure_checks["Has indicator.py"] = indicator_file.exists()
    if not indicator_file.exists():
        result.errors.append("Missing indicator.py")
        return False

    # Parse the indicator module
    tree = _parse_indicator_module(indicator_path)
    if tree is None:
        result.errors.append("Failed to parse indicator.py (syntax error?)")
        return False

    class_info = _find_class_info(tree)

    # Check for BaseIndicator inheritance
    has_base_indicator = any(
        "BaseIndicator" in c["bases"] or "BaseDecomposition" in c["bases"]
        for c in class_info["classes"]
    )
    result.structure_checks["Inherits from BaseIndicator"] = has_base_indicator
    if not has_base_indicator:
        result.errors.append("No class inherits from BaseIndicator or BaseDecomposition")

    # Check for @register_indicator decorator
    has_decorator = any(c["has_register_decorator"] for c in class_info["classes"])
    result.structure_checks["Has @register_indicator decorator"] = has_decorator
    if not has_decorator:
        result.errors.append("No class has @register_indicator decorator")

    # Check required methods
    indicator_class = next(
        (c for c in class_info["classes"] if "BaseIndicator" in c["bases"] or "BaseDecomposition" in c["bases"]),
        None,
    )
    if indicator_class:
        required_methods = ["get_metadata", "fetch_data", "calculate"]
        for method in required_methods:
            has_method = method in indicator_class["methods"]
            result.structure_checks[f"Implements {method}()"] = has_method
            if not has_method:
                result.errors.append(f"Missing required method: {method}()")

    return len(result.errors) == 0


def _check_registration(
    name: str,
    result: ValidationResult,
) -> bool:
    """Check that the indicator is properly registered and loadable."""
    try:
        from ..indicators import get_indicator, list_indicators

        # Check if name is in the registry
        registered_names = list_indicators()
        in_registry = name in registered_names
        result.registration_checks["In indicator registry"] = in_registry
        if not in_registry:
            result.errors.append(f"Indicator '{name}' not found in registry")
            return False

        # Try to instantiate
        try:
            indicator = get_indicator(name)
            result.registration_checks["Can instantiate"] = True
        except Exception as e:
            result.registration_checks["Can instantiate"] = False
            result.errors.append(f"Failed to instantiate: {e}")
            return False

        # Try to get metadata
        try:
            metadata = indicator.get_metadata()
            result.registration_checks["get_metadata() works"] = True

            # Validate metadata fields
            if not metadata.name:
                result.warnings.append("Metadata 'name' is empty")
            if not metadata.description:
                result.warnings.append("Metadata 'description' is empty")
            if not metadata.data_sources:
                result.warnings.append("Metadata 'data_sources' is empty")

        except Exception as e:
            result.registration_checks["get_metadata() works"] = False
            result.errors.append(f"get_metadata() failed: {e}")

        return True

    except ImportError as e:
        result.errors.append(f"Import error: {e}")
        return False


def _check_config_files(
    name: str,
    project_root: Path,
    result: ValidationResult,
) -> None:
    """Check for configuration files."""
    spec_dir = project_root / "config" / "model_specs"
    spec_file = spec_dir / f"{name}.json"

    has_spec = spec_file.exists()
    result.config_checks["Has spec file"] = has_spec

    if has_spec:
        try:
            import json
            with open(spec_file, "r") as f:
                spec = json.load(f)
            result.config_checks["Spec is valid JSON"] = True
        except json.JSONDecodeError as e:
            result.config_checks["Spec is valid JSON"] = False
            result.errors.append(f"Invalid JSON in spec file: {e}")
    else:
        result.warnings.append(f"No spec file found: {spec_file}")


def _check_documentation(
    name: str,
    indicators_dir: Path,
    result: ValidationResult,
) -> None:
    """Check documentation."""
    indicator_path = indicators_dir / name
    readme_path = indicator_path / "README.md"

    has_readme = readme_path.exists()
    result.doc_checks["Has README.md"] = has_readme

    if has_readme:
        content = readme_path.read_text()

        # Check for required sections
        sections = ["## Overview", "## Usage", "## Methodology"]
        for section in sections:
            has_section = section in content or section.lower() in content.lower()
            result.doc_checks[f"README has '{section}'"] = has_section
            if not has_section:
                result.warnings.append(f"README missing '{section}' section")
    else:
        result.warnings.append("Missing README.md")


def _check_code_quality(
    name: str,
    indicators_dir: Path,
    result: ValidationResult,
) -> None:
    """Check code quality indicators."""
    indicator_path = indicators_dir / name
    indicator_file = indicator_path / "indicator.py"

    if not indicator_file.exists():
        return

    content = indicator_file.read_text()

    # Check for type hints (simple heuristic)
    has_type_hints = "->" in content and ":" in content
    result.quality_checks["Has type hints"] = has_type_hints
    if not has_type_hints:
        result.warnings.append("Consider adding type hints")

    # Check for docstrings
    has_docstrings = '"""' in content or "'''" in content
    result.quality_checks["Has docstrings"] = has_docstrings
    if not has_docstrings:
        result.warnings.append("Consider adding docstrings")

    # Check for Polars usage (instead of pandas)
    uses_polars = "import polars" in content or "import pl" in content
    result.quality_checks["Uses Polars"] = uses_polars


def _check_tests(
    name: str,
    project_root: Path,
    result: ValidationResult,
) -> None:
    """Check for test files."""
    tests_dir = project_root / "tests"

    # Check for test file in various locations
    test_locations = [
        tests_dir / "indicators" / f"test_{name}.py",
        tests_dir / f"test_{name}.py",
        tests_dir / "test_indicators.py",  # May contain tests for this indicator
    ]

    has_test = any(p.exists() for p in test_locations)
    result.doc_checks["Has test file"] = has_test

    if not has_test:
        result.warnings.append(f"No test file found for {name}")


def validate_indicator(name: str) -> ValidationResult:
    """
    Validate an indicator implementation.

    Args:
        name: Indicator name

    Returns:
        ValidationResult with detailed check results
    """
    result = ValidationResult(indicator_name=name)

    try:
        project_root = _get_project_root()
    except RuntimeError as e:
        result.errors.append(str(e))
        result.passed = False
        return result

    indicators_dir = project_root / "src" / "financing_private_credit" / "indicators"

    # Run all checks
    _check_indicator_structure(name, indicators_dir, result)
    _check_registration(name, result)
    _check_config_files(name, project_root, result)
    _check_documentation(name, indicators_dir, result)
    _check_code_quality(name, indicators_dir, result)
    _check_tests(name, project_root, result)

    # Determine overall pass/fail
    result.passed = len(result.errors) == 0

    return result


def validate_all_indicators() -> list[ValidationResult]:
    """
    Validate all registered indicators.

    Returns:
        List of ValidationResults for each indicator
    """
    try:
        from ..indicators import list_indicators
        indicator_names = list_indicators()
    except ImportError:
        return []

    results = []
    for name in indicator_names:
        result = validate_indicator(name)
        results.append(result)

    return results


def get_validation_summary(results: list[ValidationResult]) -> dict[str, Any]:
    """
    Get a summary of validation results.

    Args:
        results: List of validation results

    Returns:
        Summary dictionary
    """
    return {
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "total_errors": sum(len(r.errors) for r in results),
        "total_warnings": sum(len(r.warnings) for r in results),
        "indicators": {r.indicator_name: r.passed for r in results},
    }
