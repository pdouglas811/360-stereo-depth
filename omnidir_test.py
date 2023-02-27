import cv2
import numpy as np 

# Define calibration dataset parameters

num_frames = 53
cbx = 6
cby = 8
square_size_in_mm = 80.8

# Define Termination criteria for calibrations

iterations = 1000
minimum_error = 0.0001
termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, minimum_error)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

objp = np.zeros((cbx * cby, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbx, 0:cby].T.reshape(-1, 2)
objp = objp * square_size_in_mm

# create arrays to store object points and image points from all the images

objpoints = []    # 3d point in real world space
imgpoints_R = []  # 2d points in image plane
imgpoints_L = []  # 2d points in image plane
