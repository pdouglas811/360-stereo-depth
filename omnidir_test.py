import cv2
import numpy as np

# Set up calibration parameters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objpoints = []  # 3D points in real world space
imgpoints1 = []  # 2D points in image plane for camera 1
imgpoints2 = []  # 2D points in image plane for camera 2

# Define omnidirectional camera model
K1 = np.array([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]])
xi1 = np.array([xi1])
D1 = np.array([k11, k12, k13, k14])
camera_matrix1 = K1
dist_coeffs1 = D1

K2 = np.array([[fx2, 0, cx2], [0, fy2, cy2], [0, 0, 1]])
xi2 = np.array([xi2])
D2 = np.array([k21, k22, k23, k24])
camera_matrix2 = K2
dist_coeffs2 = D2

# Load calibration images
images1 = []
images2 = []
for i in range(1, num_images+1):
    img1 = cv2.imread('camera1_%02d.png' % i)
    img2 = cv2.imread('camera2_%02d.png' % i)
    images1.append(img1)
    images2.append(img2)

# Find chessboard corners in calibration images
pattern_size = (chessboard_cols, chessboard_rows)
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
for i in range(num_images):
    img1 = images1[i]
    img2 = images2[i]
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    found1, corners1 = cv2.findChessboardCorners(gray1, pattern_size, None)
    found2, corners2 = cv2.findChessboardCorners(gray2, pattern_size, None)

    # Add object points and image points to lists
    if found1 and found2:
        objpoints.append(objp)
        corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
        imgpoints1.append(corners1)
        corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
        imgpoints2.append(corners2)

# Perform stereo calibration
criteria_calib = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-5)
flags = cv2.omnidir.CALIB_FIX_SKEW + cv2.omnidir.CALIB_USE_GUESS + cv2.omnidir.CALIB_FIX_PRINCIPAL_POINT
(rms, K, xi, D, rvecs, tvecs) = cv2.omnidir.stereoCalibrate(
    objpoints, imgpoints1, imgpoints2, camera_matrix
