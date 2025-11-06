#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Filenames and human-readable GPU names
files = {
    "V100":  "v100.txt",
    "PVC":   "pvc.txt",
    "MI250X": "mi250x.txt",  # make sure this matches your filename
}

def read_gpu_file(filename):
    """
    Read a file with format:
      Nz,vertical->i->j,i->j->vertical,j->vertical->i,vertical->j->i,j->i->vertical
        10, ...
        25, ...
    Returns: Nz (1D array), methods (list of strings), times (2D array [nNz, nMethods])
    """
    Nz = []
    data_rows = []
    methods = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("nz"):
                parts = [p.strip() for p in line.split(",")]
                methods = parts[1:]  # skip "Nz"
            else:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                Nz.append(int(parts[0]))
                data_rows.append([float(x) for x in parts[1:]])

    if methods is None or not data_rows:
        raise RuntimeError(f"Could not parse data from {filename}")

    Nz = np.array(Nz, dtype=int)
    times = np.array(data_rows, dtype=float)  # shape (nNz, nMethods)
    return Nz, methods, times


# Read all three files
gpu_Nz = {}
gpu_methods = {}
gpu_times = {}

for gpu, fname in files.items():
    Nz, methods, times = read_gpu_file(fname)
    gpu_Nz[gpu] = Nz
    gpu_methods[gpu] = methods
    gpu_times[gpu] = times

# Consistency checks
reference_gpu = list(files.keys())[0]
ref_Nz = gpu_Nz[reference_gpu]
ref_methods = gpu_methods[reference_gpu]

for gpu in files.keys():
    if not np.array_equal(ref_Nz, gpu_Nz[gpu]):
        raise RuntimeError(f"Nz mismatch between {reference_gpu} and {gpu}")
    if gpu_methods[gpu] != ref_methods:
        raise RuntimeError(f"Method label mismatch between {reference_gpu} and {gpu}")

Nz = ref_Nz
methods = ref_methods
ngpu = len(files)
nmethods = len(methods)

# Prepare a big figure with subplots: 3 rows × 2 cols (last subplot unused)
nrows, ncols = 3, 2
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10))
axes = axes.flatten()

x = np.arange(len(Nz))
bar_width = 0.25
gpu_list = list(files.keys())

for m_idx, method in enumerate(methods):
    ax = axes[m_idx]

    for g_idx, gpu in enumerate(gpu_list):
        times = gpu_times[gpu][:, m_idx]
        offset = (g_idx - (ngpu - 1) / 2.0) * bar_width
        ax.bar(x + offset, times, width=bar_width, label=gpu)

    ax.set_title(method)
    ax.set_xticks(x)
    ax.set_xticklabels(Nz)
    ax.set_xlabel("Nz")
    ax.set_ylabel("Time (s)")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

# Hide any unused subplots (since we have 5 methods but 6 axes)
for idx in range(nmethods, nrows * ncols):
    fig.delaxes(axes[idx])

# Put a single legend for all GPUs at the top
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=ngpu)

fig.suptitle("GPU Performance Comparison per Loop Ordering", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

