"""
RFUAV ChannelSim Viewer Entry Point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run via Python module:
    python -m ChannelSimulation.viewer.app

Or run this file directly:
    python ChannelSimulation/viewer/run.py
"""

import sys
from pathlib import Path

# This file is only intended for development use.
# For proper installation, run: pip install -e d:/ML/RFUAV

# Import and run main application
from ChannelSimulation.viewer.app import main

if __name__ == "__main__":
    sys.exit(main())
