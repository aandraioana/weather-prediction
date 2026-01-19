#!/usr/bin/env python3
"""
Weather Prediction Project - Main Entry Point

This project predicts temperature using weather data from three environments:
- Savanna Preserve
- Clean Urban Air
- Resilient Fields

Usage:
    python main.py [command]

Commands:
    train       - Train the linear regression model (default)
    nn          - Train the neural network model
    features    - Run feature engineering
    importance  - Analyze feature importance
    anomalies   - Run anomaly detection
    qqplot      - Generate Q-Q plots
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def run_module(module_path: str):
    """Run a Python module as a script."""
    subprocess.run([sys.executable, str(PROJECT_ROOT / module_path)], check=True)


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "train"

    commands = {
        "train": "src/models/linear_regression.py",
        "nn": "src/models/neural_network.py",
        "features": "src/features/engineering.py",
        "importance": "src/features/importance.py",
        "anomalies": "src/exploration/anomalies.py",
        "qqplot": "src/exploration/qqplot.py",
    }

    if command == "help" or command == "--help" or command == "-h":
        print(__doc__)
        return

    if command not in commands:
        print(f"Unknown command: {command}")
        print(f"Available commands: {', '.join(commands.keys())}")
        sys.exit(1)

    print(f"Running: {command}")
    run_module(commands[command])


if __name__ == "__main__":
    main()
