#!/usr/bin/env python3
"""
swe_plot.py

Looping plot of time evolution of eta = h - H0 from CSV files dumped by swe_sim.py.

Usage:
    python swe_plot.py [output_dir]

Default output_dir is "swe_output".
"""

import sys
import glob
import re
import numpy as np
import matplotlib.pyplot as plt


def extract_step_number(path):
    """
    Extract integer step from filename like 'eta_step_00100.csv'.
    Returns None if pattern not found.
    """
    m = re.search(r"eta_step_(\d+)\.csv", path)
    if m:
        return int(m.group(1))
    return None


def load_eta_files(output_dir):
    pattern = output_dir.rstrip("/") + "/eta_step_*.csv"
    files = glob.glob(pattern)
    if not files:
        raise RuntimeError(f"No CSV files found matching {pattern}")

    # Sort by step number
    files_with_steps = [(f, extract_step_number(f)) for f in files]
    files_with_steps = [fs for fs in files_with_steps if fs[1] is not None]
    files_with_steps.sort(key=lambda fs: fs[1])

    sorted_files = [fs[0] for fs in files_with_steps]
    steps = [fs[1] for fs in files_with_steps]
    return sorted_files, steps


def main():
    # Get output directory from command line or default
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "swe_output"

    files, steps = load_eta_files(output_dir)
    print(f"Found {len(files)} eta files in {output_dir}")

    # Load the first frame to get shape
    eta0 = np.loadtxt(files[0], delimiter=",")
    ny, nx = eta0.shape

    # Set up plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        eta0,
        origin="lower",
        cmap="RdBu_r",
        extent=[0, nx, 0, ny],
    )
    cbar = fig.colorbar(im, ax=ax, label="eta (m)")
    title = ax.set_title(f"Step {steps[0]}")
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    plt.tight_layout()
    plt.show(block=False)

    print("Playing frames in a loop. Close the window or Ctrl+C to stop.")

    try:
        while True:  # outer loop: restart when done
            for f, step in zip(files, steps):
                # If the figure was closed, exit cleanly
                if not plt.get_fignums():
                    print("Figure closed, exiting.")
                    plt.ioff()
                    return

                eta = np.loadtxt(f, delimiter=",")
                im.set_data(eta)
                im.set_clim(vmin=np.min(eta), vmax=np.max(eta))
                title.set_text(f"Step {step}")
                plt.pause(0.05)  # adjust playback speed

            # when we reach here, one full pass is done
            # loop will restart from the first file
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Exiting.")
    finally:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()

