import numpy as np
import cv2

# Define the number of calibration images
num_calibration_images = 53

# Define the number of corners in the calibration pattern
pattern_size = (6, 8)

# Define the size of each square on the calibration pattern in meters
square_size = 0.0808

# Define the omnidirectional camera model
omnidir_model = cv2.omnidir.DUAL_EQUIRECTANGULAR

# Define the calibration flags
calib_flags = cv2.omnidir.CALIB_USE_GUESS + cv2.omnidir.CALIB_FIX_SKEW

# Create arrays to store the object points and image points for each calibration image
object_points = []
image_points_left = []
image_points_right = []

# Generate the object points for the calibration pattern
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# Load the calibration images from both cameras
for i in range(num_calibration_images):
    img_left = cv2.imread(f"/media/AC10-0657/Images/Set_1/Left/{i}")
    img_right = cv2.imread(f"/media/AC10-0657/Images/Set_1/Right{i}")

    # Find the corners of the calibration pattern in both images
    ret_left, corners_left = cv2.findChessboardCorners(img_left, pattern_size)
    ret_right, corners_right = cv2.findChessboardCorners(img_right, pattern_size)

    # If the corners are found in both images, add the object and image points to their respective arrays
    if ret_left and ret_right:
        object_points.append(objp)
        image_points_left.append(corners_left)
        image_points_right.append(corners_right)

# Perform stereo calibration to obtain the intrinsic and extrinsic parameters of the cameras
ret, K_left, D_left, K_right, D_right, R, T = cv2.omnidir.stereoCalibrate(object_points, image_points_left, image_points_right, None, None, None, None, img_left.shape[::-1], omnidir_model, calib_flags)

# Save the calibration parameters to a file
np.savez('calibration_params.npz', K_left=K_left, D_left=D_left, K_right=K_right, D_right=D_right, R=R, T=T)
