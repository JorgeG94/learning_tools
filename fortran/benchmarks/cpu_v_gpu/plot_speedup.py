#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

CPU_FILE = "large_cpu.txt"
GPU_FILE = "large_gpu.txt"

def read_single_line_file(filename):
    """
    Expect format:
      Nz,vertical->i->j,i->j->vertical,j->vertical->i,vertical->j->i,j->i->vertical
        400, ...
    Returns: methods (list of names), times (1D array over methods)
    """
    with open(filename, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    header_line = None
    data_line = None
    for l in lines:
        if l.lower().startswith("nz"):
            header_line = l
        elif l[0].isdigit():
            data_line = l
            break

    if header_line is None or data_line is None:
        raise RuntimeError(f"Could not parse {filename}")

    header_parts = [p.strip() for p in header_line.split(",")]
    methods = header_parts[1:]  # skip "Nz"

    data_parts = [p.strip() for p in data_line.split(",")]
    # nz_value = int(data_parts[0])  # not used
    times = np.array([float(x) for x in data_parts[1:]], dtype=float)

    return methods, times

# Read CPU and GPU data
methods_cpu, times_cpu = read_single_line_file(CPU_FILE)
methods_gpu, times_gpu = read_single_line_file(GPU_FILE)

# Check consistency
if methods_cpu != methods_gpu:
    raise RuntimeError("Method labels differ between CPU and GPU files")

methods = methods_cpu
nmethods = len(methods)

# Calculate speedup (CPU time / GPU time)
speedup = times_cpu / times_gpu

x = np.arange(nmethods)

# Set up for high-quality 300 DPI output
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# --- Speedup plot ---
bars_speedup = ax.bar(x, speedup, width=0.6, color='skyblue')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=30, ha="right")
ax.set_ylabel("Speedup (CPU/GPU)")
ax.set_title("GPU Speedup over CPU (Nk = 400)")
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.axhline(y=1, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='No speedup (1x)')
ax.legend()

# Add annotations on top of bars
for i, (bar, speedup_val) in enumerate(zip(bars_speedup, speedup)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{speedup_val:.2f}x',
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()

# Save as high-quality 300 DPI PNG
output_filename = "gpu_speedup.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print(f"Speedup plot saved as {output_filename} (300 DPI)")

plt.show()
