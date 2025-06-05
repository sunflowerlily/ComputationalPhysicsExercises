import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def draw_galton_board(n_rows, n_cols):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for i in range(n_rows):
        for j in range(n_cols):
            if i % 2 == 0 and j % 2 == 0:
                ax.plot(j, -i, 'ko', markersize=10)
            elif i % 2 == 1 and j % 2 == 1:
                ax.plot(j, -i, 'ko', markersize=10)
    
    ax.set_xlim(-1, n_cols)
    ax.set_ylim(-n_rows, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig, ax

def simulate_ball_drop(n_rows, n_cols, n_balls):
    fig, ax = draw_galton_board(n_rows, n_cols)
    
    ball_positions = []
    for _ in range(n_balls):
        position = [n_cols // 2, 0]
        path = [position.copy()]
        for _ in range(n_rows):
            if np.random.rand() < 0.5:
                position[0] -= 1
            else:
                position[0] += 1
            position[1] -= 1
            path.append(position.copy())
        ball_positions.append(path)
    
    def update(frame):
        ax.clear()
        draw_galton_board(n_rows, n_cols)
        for path in ball_positions:
            if frame < len(path):
                ax.plot(path[frame][0], path[frame][1], 'ro', markersize=10)
    
    anim = FuncAnimation(fig, update, frames=n_rows+1, interval=500, repeat=False)
    plt.show()

n_rows = 10
n_cols = 11
n_balls = 5

simulate_ball_drop(n_rows, n_cols, n_balls)
