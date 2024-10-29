import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


"""
    Renders a heatmap of policy values
"""
def render_heatmap(frame, values, rows, cols, env_name, algo_name, ax=None):

    cell_height = frame.shape[0] // rows
    values = np.array(values).reshape((rows, cols))

    if ax is None:
        ax = plt.gca()
    cmap = cm.get_cmap('Greys')

    im = ax.imshow(values, cmap=cmap, extent=[0, cols, 0, rows])
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Values", rotation=-90, va="bottom")

    for i in range(rows):
        for j in range(cols):
            ax.text(j + 0.5, rows - i - 0.5, f'{values[i, j]:.2f}',
                    ha="center", va="center", color="red", fontsize=cell_height * 0.2)

    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(1, values.shape[1]+1), minor=True)
    ax.set_yticks(np.arange(1, values.shape[0]+1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(f"Heatmap of {algo_name} value function on {env_name}.")



"""
    Renders rows*cols arrows on the frame


    movement_dirs are \in {"LEFT", "RIGHT"}, etc., translated from the action
    indices to the real actions. Expected shape is (rows * cols, )

    rows/cols describe the number of states in each dimension of the rendered
    map

"""
def render_actions(frame, movement_dirs, rows, cols, env_name, algo_name,
        ax=None):

    if ax is None:
        ax = plt.gca()

    def rgb2gray(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    nS = len(movement_dirs)

    height, width, _ = frame.shape

    frame = rgb2gray(frame)

    env_image = ax.imshow(frame, cmap='gray')

    cell_height = height // rows
    cell_width = width // cols

    dir_to_arrow = {
        "LEFT": (-0.25 * cell_width, 0),   
        "DOWN": (0, 0.25 * cell_height),    
        "RIGHT": (0.25 * cell_width, 0),    
        "UP": (0, -0.25 * cell_height)     
    }

    for state in range(nS):

        row = state // cols
        col = state % cols

        # get arrow from action description
        direction = dir_to_arrow[movement_dirs[state]]
        
        center_x = col * cell_width + cell_width // 2
        center_y = row * cell_height + cell_height // 2

        dx, dy = direction
        
        # draw arrow in the current cell, pointing in the dir of optimal action
        ax.arrow(center_x, center_y, dx, dy, color='red', 
                  head_width=cell_width * 0.2, head_length=cell_height * 0.2)


    ax.set_title(f"{env_name} with {algo_name} policy actions")
    return ax

