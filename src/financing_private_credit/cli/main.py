"""
Main CLI entry point for financing-private-credit.

Usage:
    financing-private-credit new-indicator NAME [OPTIONS]
    financing-private-credit validate NAME [OPTIONS]
    financing-private-credit list
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from .scaffold import ScaffoldConfig, scaffold_indicator


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="financing-private-credit",
        description="CLI tools for the financing-private-credit indicator framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create a new indicator with default settings
    financing-private-credit new-indicator my_indicator --author "John Doe" --description "My indicator"

    # Create with nowcast support
    financing-private-credit new-indicator my_indicator --author "Jane Doe" --description "My indicator" --supports-nowcast

    # Validate an indicator
    financing-private-credit validate my_indicator

    # List all registered indicators
    financing-private-credit list
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # new-indicator command
    new_parser = subparsers.add_parser(
        "new-indicator",
        help="Create a new indicator from template",
        description="Scaffold a new indicator package with proper structure and registration",
    )
    new_parser.add_argument(
        "name",
        help="Indicator name in snake_case (e.g., credit_quality)",
    )
    new_parser.add_argument(
        "--author",
        "-a",
        required=True,
        help="Author name",
    )
    new_parser.add_argument(
        "--description",
        "-d",
        required=True,
        help="Brief description of the indicator",
    )
    new_parser.add_argument(
        "--supports-forecast",
        action="store_true",
        default=True,
        help="Include forecast.py (default: True)",
    )
    new_parser.add_argument(
        "--no-forecast",
        action="store_true",
        help="Exclude forecast.py",
    )
    new_parser.add_argument(
        "--supports-nowcast",
        action="store_true",
        help="Include nowcast.py and enable nowcasting",
    )
    new_parser.add_argument(
        "--supports-backtest",
        action="store_true",
        help="Include backtest.py",
    )
    new_parser.add_argument(
        "--supports-viz",
        action="store_true",
        help="Include viz.py for visualizations",
    )

    # validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate an indicator implementation",
        description="Check that an indicator follows framework conventions",
    )
    validate_parser.add_argument(
        "name",
        nargs="?",
        help="Indicator name to validate (omit for --all)",
    )
    validate_parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all indicators",
    )
    validate_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List all registered indicators",
    )
    list_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information",
    )

    return parser


def cmd_new_indicator(args: argparse.Namespace) -> int:
    """Handle the new-indicator command."""
    config = ScaffoldConfig(
        name=args.name,
        author=args.author,
        description=args.description,
        supports_forecast=not args.no_forecast,
        supports_nowcast=args.supports_nowcast,
        supports_backtest=args.supports_backtest,
        supports_viz=args.supports_viz,
    )

    print(f"Creating indicator: {config.name}")
    print(f"  Class name: {config.class_name}")
    print(f"  Author: {config.author}")
    print(f"  Supports forecast: {config.supports_forecast}")
    print(f"  Supports nowcast: {config.supports_nowcast}")
    print()

    result = scaffold_indicator(config)

    if result.success:
        print("Successfully created indicator!")
        print()
        print("Files created:")
        for f in result.files_created:
            print(f"  - {f}")
        print()
        print("Next steps:")
        print(f"  1. Edit src/financing_private_credit/indicators/{config.name}/indicator.py")
        print("     - Implement fetch_data() to gather required data")
        print("     - Implement calculate() with your indicator logic")
        print()
        print(f"  2. Run: financing-private-credit validate {config.name}")
        print()
        print(f"  3. Run tests: pytest tests/indicators/test_{config.name}.py")
        return 0
    else:
        print("Failed to create indicator:")
        for error in result.errors:
            print(f"  - {error}")
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Handle the validate command."""
    # Import here to avoid circular imports
    from .validate import validate_indicator, validate_all_indicators

    if args.all:
        results = validate_all_indicators()
        all_passed = all(r.passed for r in results)

        if args.json:
            import json
            output = [
                {
                    "indicator": r.indicator_name,
                    "passed": r.passed,
                    "errors": r.errors,
                    "warnings": r.warnings,
                }
                for r in results
            ]
            print(json.dumps(output, indent=2))
        else:
            print("Validation Results")
            print("=" * 50)
            for r in results:
                status = "PASS" if r.passed else "FAIL"
                symbol = "+" if r.passed else "x"
                print(f"[{symbol}] {r.indicator_name}: {status}")
                if r.errors:
                    for e in r.errors:
                        print(f"    ERROR: {e}")
                if r.warnings:
                    for w in r.warnings:
                        print(f"    WARNING: {w}")
            print()
            print(f"Total: {len(results)} indicators, {sum(1 for r in results if r.passed)} passed")

        return 0 if all_passed else 1

    elif args.name:
        result = validate_indicator(args.name)

        if args.json:
            import json
            output = {
                "indicator": result.indicator_name,
                "passed": result.passed,
                "errors": result.errors,
                "warnings": result.warnings,
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"Validating indicator: {result.indicator_name}")
            print("=" * 50)

            # Structure checks
            print()
            print("Structure")
            for check, passed in result.structure_checks.items():
                symbol = "+" if passed else "x"
                print(f"  [{symbol}] {check}")

            # Registration checks
            print()
            print("Registration")
            for check, passed in result.registration_checks.items():
                symbol = "+" if passed else "x"
                print(f"  [{symbol}] {check}")

            # Configuration checks
            print()
            print("Configuration")
            for check, passed in result.config_checks.items():
                symbol = "+" if passed else "x"
                print(f"  [{symbol}] {check}")

            # Documentation checks
            print()
            print("Documentation")
            for check, passed in result.doc_checks.items():
                symbol = "+" if passed else "-" if passed is None else "x"
                print(f"  [{symbol}] {check}")

            # Code quality checks
            print()
            print("Code Quality")
            for check, passed in result.quality_checks.items():
                symbol = "+" if passed else "-" if passed is None else "x"
                print(f"  [{symbol}] {check}")

            print()
            if result.errors:
                print("Errors:")
                for e in result.errors:
                    print(f"  x {e}")
            if result.warnings:
                print("Warnings:")
                for w in result.warnings:
                    print(f"  - {w}")

            print()
            status = "PASS" if result.passed else "FAIL"
            n_warnings = len(result.warnings)
            if n_warnings > 0:
                print(f"Overall: {status} with {n_warnings} warning(s)")
            else:
                print(f"Overall: {status}")

        return 0 if result.passed else 1

    else:
        print("Error: specify an indicator name or use --all")
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    """Handle the list command."""
    from ..indicators import list_indicators, get_indicator

    indicators = list_indicators()

    if not indicators:
        print("No indicators registered")
        return 0

    print("Registered Indicators")
    print("=" * 50)

    for name in sorted(indicators):
        if args.verbose:
            try:
                ind = get_indicator(name)
                metadata = ind.get_metadata()
                nowcast = "Yes" if ind.supports_nowcast else "No"
                print(f"\n{name}")
                print(f"  Name: {metadata.name}")
                print(f"  Version: {metadata.version}")
                print(f"  Frequency: {metadata.update_frequency}")
                print(f"  Nowcast: {nowcast}")
                print(f"  Sources: {', '.join(metadata.data_sources)}")
            except Exception as e:
                print(f"\n{name}")
                print(f"  Error loading: {e}")
        else:
            print(f"  - {name}")

    print()
    print(f"Total: {len(indicators)} indicators")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "new-indicator":
        return cmd_new_indicator(args)
    elif args.command == "validate":
        return cmd_validate(args)
    elif args.command == "list":
        return cmd_list(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
