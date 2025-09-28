import numpy as np
import matplotlib.pyplot as plt
import random

# Parameters
nx, ny = 64,64         # base grid resolution
max_refine = 3          # refinement levels (each level = 2^refine subcells per base cell)
snake_length = 50
steps = 200

# Snake position (in base grid indices)
snake = [(nx // 2, ny // 2)]

# Grid refinement levels
refine = np.zeros((nx, ny), dtype=int)

# Directions for snake motion
unit = 1
dirs = [(unit,0), (-unit,0), (0,unit), (0,-unit)]

def move_snake(snake):
    head = snake[0]
    dx, dy = random.choice(dirs)
    new_head = ((head[0] + dx) % nx, (head[1] + dy) % ny)
    snake.insert(0, new_head)
    if len(snake) > snake_length:
        snake.pop()
    return snake

def update_refinement(snake, refine):
    # decay refinement
    refine = np.maximum(refine - 1, 0)
    for (i,j) in snake:
        # increase refinement level near snake
        refine[i,j] = max_refine
        # also refine neighbors one level down
        for di,dj in dirs:
            ni, nj = (i+di) % nx, (j+dj) % ny
            refine[ni,nj] = max(max_refine-1, refine[ni,nj])
    return refine

def draw_grid(ax, nx, ny, color="gray", lw=0.3, alpha=0.5):
    """Draws the base grid lines"""
    for i in range(nx+1):
        ax.axvline(i, color=color, lw=lw, alpha=alpha)
    for j in range(ny+1):
        ax.axhline(j, color=color, lw=lw, alpha=alpha)

def draw_refinement(ax, refine):
    """Draws refined subcells inside refined cells"""
    for i in range(nx):
        for j in range(ny):
            level = refine[i, j]
            if level > 0:
                # cell corners
                x0, x1 = i, i+1
                y0, y1 = j, j+1
                # how many subdivisions
                sub = 2**level
                dx = (x1 - x0) / sub
                dy = (y1 - y0) / sub
                # draw subgrid
                for ii in range(1, sub):
                    ax.axvline(x0 + ii*dx, ymin=y0/ny, ymax=y1/ny,
                               color="blue", lw=0.5, alpha=0.7)
                for jj in range(1, sub):
                    ax.axhline(y0 + jj*dy, xmin=x0/nx, xmax=x1/nx,
                               color="blue", lw=0.5, alpha=0.7)

# Visualization
plt.ion()
fig, ax = plt.subplots(figsize=(6,6))

for step in range(steps):
    snake = move_snake(snake)
    refine = update_refinement(snake, refine)
    
    ax.clear()
    ax.set_title(f"Step {step}, snake head={snake[0]}")
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_aspect("equal")
    ax.axis("off")

    # draw base grid
    draw_grid(ax, nx, ny)

    # draw refinement overlays
    draw_refinement(ax, refine)

    plt.pause(0.1)

plt.ioff()
plt.show()

