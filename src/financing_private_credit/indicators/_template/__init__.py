"""
Template Indicator Package

Copy this directory to create a new indicator:
    cp -r _template my_indicator

Then update:
1. Rename TemplateIndicator to YourIndicator
2. Update the @register_indicator decorator
3. Implement fetch_data() and calculate()
4. Export from __init__.py

Files in this template:
- indicator.py: Core indicator class (REQUIRED)
- forecast.py: Forecasting models (optional)
- nowcast.py: High-frequency updates (optional)

See CONTRIBUTING.md for detailed instructions.
"""

from .indicator import TemplateIndicator, TemplateSpec
from .forecast import TemplateForecaster, TemplateForecasterSpec
from .nowcast import TemplateNowcaster, NowcastSpec, NowcastResult

__all__ = [
    # Indicator
    "TemplateIndicator",
    "TemplateSpec",
    # Forecast
    "TemplateForecaster",
    "TemplateForecasterSpec",
    # Nowcast
    "TemplateNowcaster",
    "NowcastSpec",
    "NowcastResult",
]
