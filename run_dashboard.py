#!/usr/bin/env python3
"""
Convenience script to run the Credit Boom Leading Indicator Dashboard.

Usage:
    python run_dashboard.py

Or with streamlit directly:
    streamlit run src/financing_private_credit/dashboard.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    # Get the path to the dashboard module
    dashboard_path = Path(__file__).parent / "src" / "financing_private_credit" / "dashboard.py"

    if not dashboard_path.exists():
        print(f"Error: Dashboard not found at {dashboard_path}")
        sys.exit(1)

    # Run streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]

    print("Starting Credit Boom Leading Indicator Dashboard...")
    print(f"Running: {' '.join(cmd)}")
    print("\nOpen your browser to: http://localhost:8501")
    print("Press Ctrl+C to stop the server.\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Error running dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
