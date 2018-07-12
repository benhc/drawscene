# Overview

A program which performs 3D reconstruction of the Fountain K6 dataset using two images. 
The method used for feature matching is FAST feature detection with pyramidal Lucas-Kanade optical flow. This program exploits 
the findEssentialMat function in OpenCV3 to reduce the degrees of freedom of the projective matrices optimisation problem, making
the system much more robust to noise. 

# Usage

Use OpenCV 3.x.
Reconstruction of the fountain can be carried out by using python drawcube3.py. There is scope for reconstruction of more general 
by changing the image type and path in drawcube3, which will also require a dataset of calibration images as detailed in 
calibratecamera. 

