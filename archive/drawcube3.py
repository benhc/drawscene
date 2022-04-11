"""A standalone function to experiment with the drawing of the cube from the supplied points. 
    USE OPENCV3.2
    """

import os
import pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt
from archive.draw_flow import *
from archive.calibrate_camera import *

image = "fountain"

if image == "fountain":
    # Camera params for the fountain
    K = np.array([[2759.48, 0, 1520.69, 0, 2764.16, 1006.81, 0, 0, 1]]).reshape(3, 3)
    dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)
    # Import images as greyscale
    img1 = cv2.imread("../0005.jpg", 0)  # left image
    img2 = cv2.imread("../0004.jpg", 0)  # right image

else:
    try:
        (K, dist) = pickle.load(open("camera_data.p", "rb"))
        print("Loaded camera data from file")
    except Exception:
        print("No camera data on file, calibrating")
        K, dist = calibrate_camera()

    # Load image pair to perform reconstruction on
    path = "/examplepath/"
    os.chdir(path)
    img1 = cv2.imread("im_1.JPG", 0)  # left image
    img2 = cv2.imread("im_2.JPG", 0)  # right image
    # Correct if the image is portrait
    if img1.shape[0] < img1.shape[1]:
        K[0][0], K[1][1] = K[1][1], K[0][0]
        K[0][2], K[1][2] = K[1][2], K[0][2]


# Convert to 3 channel image
img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
# Undistort the images
img1 = cv2.undistort(img1, K, dist)
img2 = cv2.undistort(img2, K, dist)
print(img1.shape)

# Create the keypoint matches
# Initialise FAST feature detetor
fast = cv2.FastFeatureDetector_create()

# Find FAST keypoints
pts = fast.detect(img1, None)
kp1 = np.float32(np.array([p.pt for p in pts]))

# Find corresponding features using optic flow
kp2, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, kp1, None)
draw_flow(img1, kp1, kp2)

# Filter the keypoints with no match or with large error
cond = (status == 1) * (err < 10.0)
filtcond = np.concatenate((cond, cond), 1)
kp1 = kp1[filtcond].reshape(-1, 2)
kp2 = kp2[filtcond].reshape(-1, 2)
draw_flow(img1, kp1, kp2)

# Find the essential matrix
# Find the essential matrix
E, mask = cv2.findEssentialMat(kp1, kp2, K, cv2.RANSAC, threshold=0.5)
kp1 = kp1[mask.ravel() == 1]
kp2 = kp2[mask.ravel() == 1]
draw_flow(img1, kp1, kp2)

# take svd
U, s, Vt = np.linalg.svd(E)
print(s)
# orthogonal matrix.
W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

# 4 poss solutions.
R = [np.dot(U, np.dot(W, Vt)), np.dot(U, np.dot(np.transpose(W), Vt))]
T = [U[:, 2], -U[:, 2]]

print("Possible R matrices are {}".format(R))
print("Possible T matrices are {}".format(T))
print(K)

P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
P2 = np.dot(K, np.hstack((R[0], T[0].reshape(-1, 1))))

# Perform image rectification
R1, R2, Pa, Pb, Q, roi1, roi2 = cv2.stereoRectify(K, dist, K, dist, img2.shape[:2][::-1], R[0], T[0])
print(P1, P2)
map1x, map1y = cv2.initUndistortRectifyMap(K, dist, R1, Pa[:3][:3], img1.shape[:2][::-1], cv2.CV_32F)
map2x, map2y = cv2.initUndistortRectifyMap(K, dist, R2, Pb[:3][:3], img2.shape[:2][::-1], cv2.CV_32F)

img1rect = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
img2rect = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

rectimg = np.hstack((img1, img2rect))

# Put some horizontal lines in.
for i in range(20, rectimg.shape[0], 100):
    cv2.line(rectimg, (0, i), (rectimg.shape[1], i), (255, 0, 0))

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", 600, 600)
cv2.imshow("image", rectimg)
cv2.waitKey()

homimgpoints = np.transpose(cv2.triangulatePoints(P1, P2, np.transpose(kp1), np.transpose(kp2), None))

# Get the points out of homogenous coordinates
imgpoints = homimgpoints[:, :3] / np.repeat(homimgpoints[:, 3], 3).reshape(-1, 3)

# Debug - reproject points back to the 2d scene, check they line up with the original points.
projpoints1 = np.dot(P1, homimgpoints.T).T
projpoints1 = projpoints1[:, :2] / np.repeat(projpoints1[:, 2], 2).reshape(-1, 2)
plt.figure()
plt.scatter(kp1[:, 0], kp1[:, 1])
plt.scatter(projpoints1[:, 0], projpoints1[:, 1], c="red")

projpoints2 = np.dot(P2, homimgpoints.T).T
projpoints2 = projpoints2[:, :2] / np.repeat(projpoints2[:, 2], 2).reshape(-1, 2)
plt.figure()
plt.scatter(kp2[:, 0], kp2[:, 1])
plt.scatter(projpoints2[:, 0], projpoints2[:, 1], c="red")

# Plot the 3d points. Implement a colour depth map.
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    imgpoints[:, 0],
    imgpoints[:, 1],
    imgpoints[:, 2],
    marker="o",
    edgecolors="none",
    c=imgpoints[:, 2],
    cmap="plasma",
)
plt.show()


def is_in_front(kp1, kp2, R, T, K):
    """
    Function to resolve ambguity in projection matrix. kp1 and kp2 contain the
    matched keypoints in images 1 and 2 respectively
    """

    # Use R and T to find the 3d coordinates and then check if they have the right
    # coordinates to be in front of the camera
    # Make the projection matrices
    P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(K, np.hstack((R, T.reshape(3, 1))))
    # Triangulate the world coordinate points
    homimgpoints = np.transpose(cv2.triangulatePoints(P1, P2, np.transpose(kp1), np.transpose(kp2), None))
    imgpoints = homimgpoints[:, :3] / np.repeat(homimgpoints[:, 3], 3).reshape(-1, 3)

    # Put the world coordinates into camera centred coordinates for each camera.
    pts1 = [i for i in imgpoints]
    pts2 = [np.reshape(np.dot(R, i.reshape(3, 1)) + T.reshape(3, 1), (1, 3))[0] for i in imgpoints]
    # If the camera cant see it i .e −ve Z value, put False in the array
    viewpts1 = np.reshape([True if p[2] > 0 else False for p in pts1], (-1, 1))
    viewpts2 = np.reshape([True if p[2] > 0 else False for p in pts2], (-1, 1))

    # Count the number of points that can’t be seen
    viewcond = viewpts1 * viewpts2
    return sum(viewcond)
