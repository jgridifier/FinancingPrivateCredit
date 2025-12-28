"""
Template Indicator Package

Copy this directory to create a new indicator:
    cp -r _template my_indicator

Then update:
1. Rename TemplateIndicator to YourIndicator
2. Update the @register_indicator decorator
3. Implement fetch_data() and calculate()
4. Export from __init__.py
"""

from .indicator import TemplateIndicator

__all__ = ["TemplateIndicator"]
