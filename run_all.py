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
    python run_all.py --debug      # Show debug output
"""

import argparse
import glob
import os
import subprocess
import sys

# Add project root to Python path before importing local modules
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import logging after path is set
from src.log_config import setup_logging


def check_bonus_has_images():
    """Check if there are sample images for the bonus."""
    sample_dir = "data/samples"
    if not os.path.exists(sample_dir):
        return False
    sample_images = [
        f
        for f in os.listdir(sample_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return len(sample_images) > 0


def run_bonus_with_validation(env, verbose_level, logger):
    """
    Run the bonus script and validate it produced output files.
    Returns True if successful, False otherwise.
    """
    # First, check if there are sample images
    if not check_bonus_has_images():
        logger.warning("No sample images found in data/samples/.")
        logger.info("Please add 3-5 of your own photos to data/samples/")
        logger.info("Bonus skipped (no images to process)")
        return False

    # Run the bonus script
    result = subprocess.run(
        [sys.executable, "experiments/bonus_inference.py"],
        capture_output=False,
        check=False,
        env=env,
    )

    if result.returncode != 0:
        logger.error(f"Bonus script failed with code {result.returncode}")
        return False

    # Verify output files were created
    zoo_files = glob.glob("figures/zoo_*.png")
    if not zoo_files:
        logger.warning("Bonus script ran but produced no output files.")
        logger.info(
            "Expected: figures/zoo_detection_*.png, zoo_segmentation_*.png, zoo_pose_*.png"
        )
        return False

    logger.success(f"Bonus produced {len(zoo_files)} output file(s):")
    for f in zoo_files[:5]:  # Show first 5
        logger.info(f"  - {f}")
    return True


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run all HW2 experiments.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output from experiments (WARNING and above)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all output from experiments (INFO and above - default)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug output from experiments (DEBUG and above)",
    )
    args = parser.parse_args()

    # Determine verbosity level
    if args.debug:
        verbose_level = 2
    elif args.verbose or (not args.quiet and not args.debug):
        verbose_level = 1
    else:  # --quiet
        verbose_level = 0

    # Setup logging
    logger = setup_logging(verbose_level=verbose_level)

    # Set student ID
    os.environ["STUDENT_ID"] = "121314"

    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("data/samples", exist_ok=True)

    # Define experiments
    experiments = [
        ("B1: SmallCNN", "experiments/train_cnn.py"),
        ("B2: Transfer Learning", "experiments/transfer_compare.py"),
        ("B3: Augmentation Study", "experiments/augment_compare.py"),
    ]

    # Set up environment with PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    # ============================================================
    # Run Core Experiments (B1, B2, B3)
    # ============================================================
    logger.info("=" * 60)
    logger.info("Running HW2 Core Experiments (B1, B2, B3)")
    logger.info("Student ID: 121314")
    logger.info(f"Verbose level: {['WARNING', 'INFO', 'DEBUG'][verbose_level]}")
    logger.info("=" * 60)

    for name, script in experiments:
        logger.info(f"\n>>> {name} ({script})")
        logger.info("-" * 40)

        result = subprocess.run(
            [sys.executable, script],
            capture_output=False,
            check=False,
            env=env,
        )

        if result.returncode != 0:
            logger.error(f"{name} failed with code {result.returncode}")
            if verbose_level < 2:
                logger.info("Run with --debug to see detailed output")
            sys.exit(result.returncode)

        logger.success(f"{name} completed successfully")

    # ============================================================
    # Run Bonus (with validation)
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info(">>> Bonus: Inference")
    logger.info("-" * 40)

    bonus_success = run_bonus_with_validation(env, verbose_level, logger)

    if bonus_success:
        logger.success("Bonus completed successfully!")
    else:
        logger.warning("Bonus was skipped or failed — not required for full credit.")

    # ============================================================
    # Final Summary
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.success("All core experiments complete!")

    # Check what was generated
    figure_files = glob.glob("figures/*.png")
    logger.info(f"Generated {len(figure_files)} figure(s):")
    for f in sorted(figure_files):
        logger.info(f"  - {f}")

    logger.info("=" * 60)

    # Exit with appropriate code
    # If bonus failed due to missing images, still exit 0 (bonus optional)
    sys.exit(0)


if __name__ == "__main__":
    main()
