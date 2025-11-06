import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_benchmark_file(path):
    """
    Parse the text file with blocks like:

    ! serial, triple nested do
     Nz,vertical->i->j,i->j->vertical,j->vertical->i,vertical->j->i
       10, ...

    Returns:
        Nz: 1D numpy array of Nz values (shared by all blocks)
        loop_order_labels: list of 4 strings
        configs: dict[name] -> 2D numpy array shape (len(Nz), 4)
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    configs = {}
    loop_order_labels = None
    shared_Nz = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for block start
        if line.startswith('!'):
            # Extract human-readable name after '!'
            name_text = line.lstrip('!').strip()
            # Make a compact key for filenames / legends
            key = re.sub(r'[^0-9a-zA-Z]+', '_', name_text).strip('_').lower()

            # Next line should be header
            i += 1
            if i >= len(lines):
                break
            header_line = lines[i].strip()
            if not header_line.lower().startswith('nz'):
                # skip until we find an 'Nz' header, just in case
                while i < len(lines) and not lines[i].strip().lower().startswith('nz'):
                    i += 1
                if i >= len(lines):
                    break
                header_line = lines[i].strip()

            # Parse header to get loop labels once
            header_parts = [h.strip() for h in header_line.split(',')]
            if loop_order_labels is None:
                # header_parts[0] is 'Nz', the rest are loop orders
                loop_order_labels = header_parts[1:]

            # Now read data lines until blank or next '!' block
            nz_vals = []
            data_rows = []
            i += 1
            while i < len(lines):
                l = lines[i]
                stripped = l.strip()
                if not stripped:
                    break
                if stripped.startswith('!'):
                    # next block starts here
                    i -= 1  # step back so outer loop sees this '!'
                    break
                # Expect a CSV-style line
                parts = [p.strip() for p in stripped.split(',')]
                if len(parts) < 5:
                    i += 1
                    continue
                nz_vals.append(int(parts[0]))
                data_rows.append([float(x) for x in parts[1:5]])
                i += 1

            nz_arr = np.array(nz_vals, dtype=int)
            data_arr = np.array(data_rows, dtype=float)

            # Store / verify Nz consistency
            if shared_Nz is None:
                shared_Nz = nz_arr
            else:
                if not np.array_equal(shared_Nz, nz_arr):
                    print(f"Warning: Nz mismatch in block {name_text}")

            configs[key] = {
                "name": name_text,
                "data": data_arr
            }

        i += 1

    return shared_Nz, loop_order_labels, configs


def plot_per_loop_order(Nz, loop_order_labels, configs, outdir="plots"):
    """
    For each loop ordering, plot time vs Nz, grouped by execution config.
    """
    os.makedirs(outdir, exist_ok=True)
    x = np.arange(len(Nz))
    cfg_keys = list(configs.keys())
    bar_width = 0.8 / max(1, len(cfg_keys))  # keep total width reasonable

    for loop_idx, loop_label in enumerate(loop_order_labels):
        plt.figure()
        for ci, key in enumerate(cfg_keys):
            cfg = configs[key]
            data = cfg["data"]
            human_name = cfg["name"]
            offset = (ci - (len(cfg_keys)-1)/2) * bar_width
            plt.bar(x + offset, data[:, loop_idx], width=bar_width, label=human_name)

        plt.xticks(x, Nz)
        plt.xlabel("Nz")
        plt.ylabel("Time (s)")
        plt.title(f"Timing vs Nz for loop order: {loop_label}")
        plt.legend(fontsize="small")
        plt.tight_layout()

        fname_safe = re.sub(r'[^0-9a-zA-Z]+', '_', loop_label).strip('_').lower()
        outpath = os.path.join(outdir, f"timings_by_loop_{fname_safe}.png")
        plt.savefig(outpath)
        plt.close()
        print("Saved", outpath)


def plot_per_execution_method(Nz, loop_order_labels, configs, outdir="plots"):
    """
    For each execution method/config, plot time vs Nz, grouped by loop order.
    """
    os.makedirs(outdir, exist_ok=True)
    x = np.arange(len(Nz))
    bar_width = 0.8 / max(1, len(loop_order_labels))

    for key, cfg in configs.items():
        human_name = cfg["name"]
        data = cfg["data"]

        plt.figure()
        for li, loop_label in enumerate(loop_order_labels):
            offset = (li - (len(loop_order_labels)-1)/2) * bar_width
            plt.bar(x + offset, data[:, li], width=bar_width, label=loop_label)

        plt.xticks(x, Nz)
        plt.xlabel("Nz")
        plt.ylabel("Time (s)")
        plt.title(f"Timing vs Nz for execution method: {human_name}")
        plt.legend(fontsize="small")
        plt.tight_layout()

        key_safe = re.sub(r'[^0-9a-zA-Z]+', '_', human_name).strip('_').lower()
        outpath = os.path.join(outdir, f"timings_by_exec_{key_safe}.png")
        plt.savefig(outpath)
        plt.close()
        print("Saved", outpath)


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_benchmarks.py <data_file.txt>")
        sys.exit(1)

    data_path = sys.argv[1]
    Nz, loop_order_labels, configs = parse_benchmark_file(data_path)

    if Nz is None or loop_order_labels is None or not configs:
        print("No valid data parsed. Check input file format.")
        sys.exit(1)

    print("Parsed Nz:", Nz)
    print("Loop orders:", loop_order_labels)
    print("Configs:", [cfg["name"] for cfg in configs.values()])

    plot_per_loop_order(Nz, loop_order_labels, configs)
    plot_per_execution_method(Nz, loop_order_labels, configs)


if __name__ == "__main__":
    main()

