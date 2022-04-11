def calibrate_camera():
    """Find the camera matrix and distortion coefficients for the camera.
    The directory should contain approx 20 photos of the chessboard from different angles.
    Returns the camera matrix and distortion coefficients. Also saves a file containing the
    coefficients to the same directory as the program.
    """

    import pickle
    import os
    import glob
    import cv2
    from archive.create_chess_grid import create_chess_grid
    import numpy as np

    # Save working directory for later
    workdir = os.getcwd()

    # Directory in which the chessboard photos are stored
    os.chdir("/calibrationphotos")

    # Size of the chessboard in the photos.
    grid = (7, 8)
    # Dimensions of the chessboard. Poss not required.
    dims = (20, 22)

    # Define the coordinates of the chessboard
    objp = create_chess_grid((grid, dims))

    # Intialise containers
    objpoints = []
    imgpoints = []

    # Make a list of calibration photos
    images = glob.glob("*.JPG")

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        ret, corners = cv2.findChessboardCorners(gray, grid, None, cv2.CALIB_CB_FILTER_QUADS)
        if ret:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

    # Check chessboard found in all images
    if len(images) != len(objpoints):
        print("Warning, chessboard not detected in all images")

    # Perform camera calibration
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Calculate the mean error of calibration
    mean_error = 0
    for i in xrange(len(objpoints)):
        imgpointsrec, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(imgpoints[i], imgpointsrec, cv2.NORM_L2) / len(imgpointsrec)
        mean_error = +error

    print("Mean error = {} pixels".format(mean_error))

    # Save the coefficients
    print("Saving camera information to file")
    os.chdir(workdir)
    pickle.dump((K, dist), open("camera_data.p", "wb"))

    return K, dist
