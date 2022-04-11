def create_chess_grid(grid):
    """Create the grid for the chessboard of gridsize [x, y] with dimensions [xmm, ymm]"""

    import cv2
    import numpy as np

    # unpack grid
    gridsize = grid[0]
    griddims = grid[1]

    # create the grid [x y z] for the chessboard. Z is assume 0 on the board.
    objp = np.zeros(((gridsize[0] * gridsize[1]), 3), np.float32)
    grid = np.mgrid[0 : gridsize[0], 0 : gridsize[1]].T.reshape(-1, 2)
    # scale the x and y axis to be in mm
    grid[:, 0] = grid[:, 0] * griddims[0]
    grid[:, 1] = grid[:, 1] * griddims[1]
    objp[:, :2] = grid

    return objp
