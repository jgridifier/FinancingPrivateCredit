"""
CLI tools for financing-private-credit framework.

Commands:
    financing-private-credit new-indicator NAME - Create a new indicator from template
    financing-private-credit validate NAME - Validate an indicator implementation
    financing-private-credit list - List all registered indicators
"""

from .main import main

__all__ = ["main"]
