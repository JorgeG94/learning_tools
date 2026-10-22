# Live pond (COMP3320)

The shallow-water model from the lecture: the physics runs in C with OpenMP
offload (`sw_lib.c`), and Python drives it and draws the surface height live
(`live_pond.py`). Click in the window to drop more water.

## On a Gadi GPU node (V100)

```bash
module load nvidia-hpc-sdk python3        # module names/versions: check `module avail`
python3 live_pond.py --save pond.gif      # builds libsw.so with nvc (cc70), writes a GIF
python3 live_pond.py --save pond.mp4      # same, as MP4 (needs ffmpeg)
python3 live_pond.py                      # live window: only with X forwarding (qsub -I -X)
```

`live_pond.py` builds `libsw.so` the first time (and whenever `sw_lib.c` changes):

```bash
nvc -O3 -mp=gpu -gpu=cc70,nordc -shared -fPIC sw_lib.c -o libsw.so
```

Use `--cc gcc` for a CPU-only build on a laptop. Needs Python 3 with numpy and matplotlib.

## Options

| Option | Meaning |
|---|---|
| `-n N` | grid is N x N cells (default 256) |
| `-k K` | time steps between frames (default 5) |
| `--frames F` | number of frames (default 600) |
| `--save FILE` | write `.gif` or `.mp4` instead of opening a window |
| `--cc nvc/gcc/clang` | compiler for `sw_lib.c` (default `nvc`) |
| `--rebuild` | force a rebuild of `libsw.so` |

The status line shows where the kernels ran (host or device), time spent computing vs
copying eta back for each frame, throughput, and the total mass, which must stay constant.
Reference: 256 x 256 after 900 steps, mass = 402.1238596594.

## Things to try in class

- `-k 1` vs `-k 50`: every frame copies eta back to the host (`target update from`).
  That copy is the MOVE question: watch the copy time and the throughput change.
- `-n 128` vs `-n 2048`: small grids cannot keep a GPU busy.
- `OMP_TARGET_OFFLOAD=MANDATORY python3 live_pond.py --save x.gif`: proves it ran on the GPU.
