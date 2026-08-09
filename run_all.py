#!/usr/bin/env python
# =====================================================================
# FILE: run_all.py
# =====================================================================
"""
One-command regeneration of all figures.

Usage:
    python run_all.py              # Show all output (default)
    python run_all.py --quiet      # Suppress detailed output
    python run_all.py --verbose    # Show all output (explicit)
"""
import os
import sys
import subprocess
import argparse


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run all HW2 experiments."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output from experiments"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all output from experiments (default behavior)"
    )
    args = parser.parse_args()

    # Determine verbosity: verbose=True by default unless --quiet is used
    verbose = not args.quiet or args.verbose

    # Set student ID
    os.environ["STUDENT_ID"] = "121314"

    # Add project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("data/samples", exist_ok=True)

    experiments = [
        ("B1: SmallCNN", "experiments/train_cnn.py"),
        ("B2: Transfer Learning", "experiments/transfer_compare.py"),
        ("B3: Augmentation Study", "experiments/augment_compare.py"),
        ("Bonus: Inference", "experiments/bonus_inference.py"),
    ]

    print("=" * 60)
    print("Running all HW2 experiments")
    print(f"Student ID: 121314")
    print(f"Verbose mode: {verbose}")
    print("=" * 60)

    for name, script in experiments:
        print(f"\n>>> {name} ({script})")
        print("-" * 40)

        # Set up environment
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

        # Determine output behavior
        if verbose:
            # Show all output (current behavior)
            stdout = None
            stderr = None
        else:
            # Suppress output, but keep showing progress messages from run_all.py
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL

        result = subprocess.run(
            [sys.executable, script],
            capture_output=False,
            check=False,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )

        if result.returncode != 0:
            print(f"ERROR: {name} failed with code {result.returncode}")
            if not verbose:
                print("Run with --verbose to see detailed error output")
            sys.exit(result.returncode)

    print("\n" + "=" * 60)
    print("All experiments complete!")
    print("Generated figures:")
    print("  - figures/cnn_curves.png")
    print("  - figures/transfer_compare.png")
    print("  - figures/augment_compare.png")
    print("  - figures/zoo_*.png (bonus)")
    print("=" * 60)


if __name__ == "__main__":
    main()