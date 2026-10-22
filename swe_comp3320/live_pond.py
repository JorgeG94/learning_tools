#!/usr/bin/env python3
"""live_pond.py -- watch the shallow-water model from the lecture run live.

The model itself runs in C (sw_lib.c, OpenMP offload). Python only asks for
a snapshot of eta every few steps and draws it.

    python3 live_pond.py                    # 256 x 256, live window, built with nvc for a V100
    python3 live_pond.py -n 1024 -k 20      # bigger grid, 20 steps per frame
    python3 live_pond.py --save pond.mp4    # no window: write a movie (needs ffmpeg)
    python3 live_pond.py --save pond.gif    # no window, no ffmpeg needed
    python3 live_pond.py --cc gcc           # CPU-only build, e.g. on a laptop

Click in the window to drop more water.
"""
import argparse, ctypes, os, subprocess, sys, time
import numpy as np
import matplotlib

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "sw_lib.c")
LIB = os.path.join(HERE, "libsw.so")

BUILD = {
    "gcc":   ["gcc", "-O3", "-fopenmp", "-shared", "-fPIC", SRC, "-lm", "-o", LIB],
    "clang": ["clang", "-O3", "-fopenmp", "-shared", "-fPIC", SRC, "-lm", "-o", LIB],
    "nvc":   ["nvc", "-O3", "-mp=gpu", "-gpu=cc70,nordc", "-shared", "-fPIC", SRC, "-o", LIB],
}

def build(cc, force):
    if force or not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC):
        print("building:", " ".join(BUILD[cc]))
        subprocess.run(BUILD[cc], check=True)

def load():
    lib = ctypes.CDLL(LIB)
    lib.sw_init.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double]
    lib.sw_drop.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.sw_step.argtypes = [ctypes.c_int]
    lib.sw_get_eta.argtypes = [np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")]
    lib.sw_get_eta.restype = ctypes.c_double
    lib.sw_dt.restype = ctypes.c_double
    return lib

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", type=int, default=256, help="cells per side (default 256)")
    p.add_argument("-k", type=int, default=5, help="time steps per frame (default 5)")
    p.add_argument("--frames", type=int, default=600, help="frames to run (default 600)")
    p.add_argument("--cc", choices=BUILD, default="nvc", help="compiler for sw_lib.c (default nvc, V100 = cc70)")
    p.add_argument("--rebuild", action="store_true", help="force a rebuild")
    p.add_argument("--save", metavar="FILE", help="write a movie instead of a live window (.mp4 needs ffmpeg; .gif works without)")
    args = p.parse_args()

    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    build(args.cc, args.rebuild)
    sw = load()
    os.environ.setdefault("OMP_TARGET_OFFLOAD", "DEFAULT")
    if sw.sw_init(args.n, 0.35, 0.55) != 0:
        sys.exit("sw_init failed (out of memory?)")
    where = "host (CPU)" if sw.sw_on_host() else "device (GPU)"
    dt = sw.sw_dt()

    eta = np.empty((args.n, args.n))
    mass0 = sw.sw_get_eta(eta)

    fig, ax = plt.subplots(figsize=(6.4, 6.8))
    fig.patch.set_facecolor("#FBFAF7")
    im = ax.imshow(eta, origin="lower", cmap="RdBu_r", vmin=-0.2, vmax=0.2,
                   extent=[0, args.n, 0, args.n], interpolation="bilinear")
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02); cb.set_label("surface height η (m)")
    title = ax.set_title("")
    info = fig.text(0.02, 0.01, "", fontsize=9, color="#717784", family="monospace")
    state = {"step": 0, "t_compute": 0.0, "t_copy": 0.0}

    def on_click(event):
        if event.inaxes is ax and event.xdata is not None:
            sw.sw_drop(event.xdata / args.n, event.ydata / args.n, 1.0)
    fig.canvas.mpl_connect("button_press_event", on_click)

    def update(_):
        t0 = time.perf_counter(); sw.sw_step(args.k); t1 = time.perf_counter()
        mass = sw.sw_get_eta(eta);                     t2 = time.perf_counter()
        state["step"] += args.k
        state["t_compute"] += t1 - t0; state["t_copy"] += t2 - t1
        im.set_data(eta)
        title.set_text(f"A drop in a pond   t = {state['step'] * dt / 60:6.1f} min")
        cells = args.n * args.n * state["step"]
        info.set_text(f"running on {where} | {args.n}x{args.n}, {args.k} steps/frame | "
                      f"compute {state['t_compute']:.2f}s  copies {state['t_copy']:.2f}s | "
                      f"{cells / max(state['t_compute'], 1e-9) / 1e6:7.1f} Mcell-steps/s | "
                      f"mass {mass:.6f} (start {mass0:.6f})")
        return im, title, info

    anim = FuncAnimation(fig, update, frames=args.frames, interval=1, blit=False, repeat=False)
    if args.save:
        if args.save.endswith(".gif"):
            from matplotlib.animation import PillowWriter
            anim.save(args.save, writer=PillowWriter(fps=20), dpi=80)
        else:
            anim.save(args.save, fps=30, dpi=120)
        print(f"wrote {args.save}; final mass {sw.sw_get_eta(eta):.12f} (start {mass0:.12f})")
    else:
        plt.show()
    sw.sw_finalize()

if __name__ == "__main__":
    main()
