"""A standalone function to experiment with the drawing of the cube from the supplied points. """

import os
import pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as cls
from mpl_toolkits.mplot3d import Axes3D
from draw_flow import draw_flow 
 
image = 'cube'

if image == 'fountain':
    #Camera params for the fountain
    K = np.array([[2759.48, 0, 1520.69, 0, 2764.16, 1006.81, 0, 0, 1]]).reshape(3, 3)
    dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)

    # Import images as greyscale
    path = "/home/benhc/Documents/IIB/Project/Images/stereo_calc"
    os.chdir(path)
    img1 = cv2.imread('0005.jpg',0)  #left image
    img2 = cv2.imread('0004.jpg',0) #right image

elif image == 'cube':
    try:
        (K, dist) = pickle.load(open("camera_data.p", "rb"))
        print("Loaded camera data from file")
    except Exception:
        print("No camera data on file, calibrating")
        K, dist = calibrate_camera()
    # Load image pair to perform reconstruction on
    path = "/home/benhc/Documents/IIB/Project/Images/optic flow"
    os.chdir(path)
    img1 = cv2.imread('im_1.JPG',0)  #left image
    img2 = cv2.imread('im_2.JPG',0) #right image
    # Correct if the image is portrait
    if img1.shape[0]<img1.shape[1]:
        K[0][0], K[1][1] = K[1][1], K[0][0]
        K[0][2], K[1][2] = K[1][2], K[0][2]

# Convert to 3 channel image
img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
# Undistort the images
img1 = cv2.undistort(img1, K, dist)
img2 = cv2.undistort(img2, K, dist)

# Create the keypoint matches
# Initialise FAST feature detector
fast = cv2.FastFeatureDetector()

# Find FAST keypoints
pts = fast.detect(img1, None)
kp1 = np.float32(np.array([p.pt for p in pts]))
# Find corresponding features using optic flow
kp2, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, kp1)


draw_flow(img1, kp1, kp2)
# Filter the keypoints with no match or with large error
cond = (status == 1)*(err < 5.)
filtcond = np.concatenate((cond, cond), 1)
kp1 = kp1[filtcond].reshape(-1, 2)
kp2 = kp2[filtcond].reshape(-1, 2)
draw_flow(img1, kp1, kp2)
#Lets get the F matrix

kp1 = [[1460,	488],
[1859,	351],
[2266,	234],
[2672,	117],
[2905,	371],
[2507,	504],
[2117,	639],
[1642,	785],
[1490,	783],
[1600,	1272],
[1858,	1859],
[2072,	1567],
[2062,  1812],
[2052,	2029],
[2042,	2187],
[2551,	1364],
[2390,	2022],
[3015,	1146],
[2751,	1838],
[3055,	1710]]


kp2 = [[1709,	635],
[2096,	490],
[2488,	364],
[2873,	240],
[3127,	503],
[2753,	646],
[2386,	789],
[1932,	947],
[1704,	921],
[1756,	1393],
[2068,	1987],
[2461,	1758],
[2389,	1978],
[2324,	2175],
[2279,	2318],
[2907,	1536],
[2607,	2137],
[3329,	1299],
[2949,	1939],
[3235,	1797]]

kp1 = np.float32(kp1)
kp2 = np.float32(kp2)

# draw_flow(img1, kp1, kp2)
# F, mask = cv2.findFundamentalMat(kp1, kp2, cv2.FM_RANSAC)#, 0.1
# # Apply the mask to the keypoints
# kp1 = kp1[mask.ravel() == 1]
# kp2 = kp2[mask.ravel() == 1]
# print("The rank of F is {}. (Expect 2)".format(np.linalg.matrix_rank(F)))
# print(F)
draw_flow(img1, kp1, kp2)
F, mask = cv2.findFundamentalMat(kp1, kp2, cv2.FM_RANSAC)
kp1 = kp1[mask.ravel() == 1]
kp2 = kp2[mask.ravel() == 1]
# F, mask = cv2.findFundamentalMat(kp1, kp2, cv2.FM_8POINT)
# print("The rank of F is {}. (Expect 2)".format(np.linalg.matrix_rank(F)))
# print(F)
# draw_flow(img1, kp1, kp2)

# Get the essential matrix
E = np.dot(np.transpose(K), np.dot(F, K))
#svd
U, s, Vt = np.linalg.svd(E)
print(s)
#orthogonal matrix.
W = np.array([[0.0, -1.0, 0.0],
              [1.0, 0.0, 0.0], 
              [0.0, 0.0, 1.0]])

#4 poss solutions. 
R=[np.dot(U, np.dot(W, Vt)), np.dot(U, np.dot(np.transpose(W), Vt))]
T=[U[:, 2], -U[:, 2]]

print("Possible R matrices are {}".format(R))
print("Possible T matrices are {}".format(T))
print(K)

P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
P2 = np.dot(K, np.hstack((R[1], T[0].reshape(-1, 1))))

######### rectification
R1, R2, Pa, Pb, Q, roi1, roi2 = cv2.stereoRectify(K, dist, K, dist, img2.shape[:2][::-1], R[1], T[1])

print(P1, P2)
map1x, map1y = cv2.initUndistortRectifyMap(K, dist, R1, Pa[:3][:3], img1.shape[:2][::-1], cv2.CV_32F)
map2x, map2y = cv2.initUndistortRectifyMap(K, dist, R2, Pb[:3][:3], img2.shape[:2][::-1], cv2.CV_32F)

img1rect = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
img2rect = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

rectimg = np.hstack((img1, img2rect))

# Put some horizontal lines in. 
for i in range(20, rectimg.shape[0], 100):
    cv2.line(rectimg, (0, i), (rectimg.shape[1], i), (255, 0, 0))

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)
cv2.imshow('image', rectimg)
cv2.waitKey()
###########


homimgpoints = np.transpose(cv2.triangulatePoints(P1, P2, np.transpose(kp1), np.transpose(kp2), None))

# Get the points out of homogenous coordinates 
imgpoints = homimgpoints[:, :3]/np.repeat(homimgpoints[:, 3], 3).reshape(-1, 3)

# reproject points back to the 2d scene
projpoints1 = np.dot(P1, homimgpoints.T).T
projpoints1 = projpoints1[:, :2]/np.repeat(projpoints1[:, 2], 2).reshape(-1, 2)

plt.figure()
plt.scatter(kp1[:, 0], kp1[:, 1])
plt.scatter(projpoints1[:, 0], projpoints1[:, 1])

projpoints2 = np.dot(P2, homimgpoints.T).T
projpoints2 = projpoints2[:, :2]/np.repeat(projpoints2[:, 2], 2).reshape(-1, 2)

# for p in projpoints2:
#     cv2.circle(img2, (int(p[0]), int(p[1])), 10, (0, 0, 255), 3)

# cv2.namedWindow('image',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('image', 600,600)
# cv2.imshow("image", img2)
# cv2.waitKey()

plt.figure()
plt.scatter(kp2[:, 0], kp2[:, 1])
plt.scatter(projpoints2[:, 0], projpoints2[:, 1])

# Plot the 3d points. 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(imgpoints[:,0], imgpoints[:,1], imgpoints[:,2])#, c=imgpoints[:,0],cmap='plasma')#, norm=cls.LogNorm(imgpoints[:,2].min, imgpoints[:,2].max))
# ax = fig.add_subplot(111)
# ax.scatter(imgpoints[:,1], imgpoints[:,2], c=imgpoints[:,0], cmap='viridis')
plt.show()


# ######### METHOD 1 #########
# # Use the epipole of the F matrix. 
# #Lets get the F matrix
# F, mask = cv2.findFundamentalMat(kp1, kp2, cv2.FM_RANSAC, 10)

# # Apply the mask to the keypoints
# kp1 = kp1[mask.ravel() == 1]
# kp2 = kp2[mask.ravel() == 1]
# print("Was 20 points, now {}".format(len(kp1)))
# print("The rank of F is {}. (Expect 2)".format(np.linalg.matrix_rank(F)))
# print(F)

# U, s, V = np.linalg.svd(F)
# print("singular values are: {}".format(s))

# null = np.transpose(U)[2, :]
# print("The left null space is {}".format(null))

# # Test that the null space is indeed the epipole ie eTF=0
# print(np.dot(np.transpose(null), F))

# # That seems good, so lets construct the projection matrices. 
# e = null
# ecross = [[0, -e[2], e[1]],
#           [e[2], 0, -e[0]],
#           [-e[1], e[0], 0]]

# print(np.linalg.matrix_rank(np.dot(ecross, F)))

# P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
# P2 = np.hstack((np.dot(ecross, F), e.reshape(-1, 1)))
# print("P1 is {}".format(P1))
# print("P2 is {}".format(P2))

# # We can now theoretically just go ahead and reconstruct
# # Triangulate the points using the projection matrices
# homimgpoints = np.transpose(cv2.triangulatePoints(P1, P2, np.transpose(kp1), np.transpose(kp2), None))

# # Get the points out of homogenous coordinates 
# imgpoints = homimgpoints[:, :3]/np.repeat(homimgpoints[:, 3], 3).reshape(-1, 3)

# #Draw 
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(imgpoints[:,0], imgpoints[:,1], imgpoints[:,2])

# plt.show()

# #Ok that doesnt look right, how about trying to reproject those points back onto the image.
# rvec1 = cv2.Rodrigues(P1[:3, :3])[0]
# tvec1 = P1[:, 3]
# pts1, _= cv2.projectPoints(imgpoints, (rvec1), tvec1, K, dist)
# rvec2 = cv2.Rodrigues(P2[:3, :3])[0]

# tvec2 = P2[:, 3]
# pts2, _ = cv2.projectPoints(imgpoints, (rvec2), tvec2, K, dist)
# print(pts1)
# print(pts2)
# for i in range(0, len(pts1)):
#     cv2.circle(img1, tuple(pts1[i][0]), 10, (0, 0, 255), 3)
#     cv2.circle(img2, tuple(pts2[i][0]), 10, (0, 0, 255), 3)

# # combine the images
# rows1 = img1.shape[0]
# cols1 = img1.shape[1]
# rows2 = img2.shape[0]
# cols2 = img2.shape[1]
# out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
# # Place the first image to the left
# out[:rows1, :cols1] = np.dstack([img1])
# # Place the next image to the right of it
# out[:rows2, cols1:] = np.dstack([img2])

# cv2.namedWindow('image',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('image', 600,600)
# cv2.imshow("image", out)
# cv2.waitKey()