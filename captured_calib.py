import cv2
import numpy as np 

# Define calibration dataset parameters

num_frames = 90
cbx = 6
cby = 8
square_size_in_mm = 80.8

# Define Termination criteria for calibrations

iterations = 100
minimum_error = 0.001
termination_criteria_subpix = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    iterations,
    minimum_error)

# Define function for drawing lines on rectified output images

def draw_lines(img, grid_shape, color, thickness):
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

objp = np.zeros((cbx * cby, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbx, 0:cby].T.reshape(-1, 2)
objp = objp * square_size_in_mm

# create arrays to store object points and image points from all the images

# create arrays to store object points and image points from all the images

# ... both for paired chessboard detections (L AND R detected)

objpoints_pairs = []         # 3d point in real world space
imgpoints_right_paired = []  # 2d points in image plane.
imgpoints_left_paired = []   # 2d points in image plane.

# ... and for left and right independantly (L OR R detected, OR = logical OR)

objpoints_left_only = []   # 3d point in real world space
imgpoints_left_only = []   # 2d points in image plane.

objpoints_right_only = []   # 3d point in real world space
imgpoints_right_only = []   # 2d points in image plane.

# count number of chessboard detection (across both images)

# Begin loop to find chessboard corners

for i in range(1, num_frames + 1):
        
        # Fetch image pairs from relevant directory

        frameL = cv2.imread(f"/media/AC10-0657/images/Set_3/Left/{i}.jpg")
        frameR = cv2.imread(f"/media/AC10-0657/images/Set_3/Right/{i}.jpg")

        # Convert image pairs to grayscale

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners in the images

        retL, cornersL = cv2.findChessboardCorners(grayL, (cbx, cby))
        retR, cornersR = cv2.findChessboardCorners(grayR, (cbx, cby))

        # -- > detected in left (only or also in right)

        if (retL):

            chessboard_pattern_detections_left += 1

            # add object points to left list

            objpoints_left_only.append(objp)

            # refine corner locations to sub-pixel accuracy and then add to list

            corners_sp_L = cv2.cornerSubPix(
                grayL, cornersL, (11, 11), (-1, -1), termination_criteria_subpix)
            imgpoints_left_only.append(corners_sp_L)

        # -- > detected in right (only or also in left)

        if (retR):

            chessboard_pattern_detections_right += 1

            # add object points to left list

            objpoints_right_only.append(objp)

            # refine corner locations to sub-pixel accuracy and then add to list

            corners_sp_R = cv2.cornerSubPix(
                grayR, cornersR, (11, 11), (-1, -1), termination_criteria_subpix)
            imgpoints_right_only.append(corners_sp_R)

        # -- > detected in left and right

        if ((retR) and (retL)):

            chessboard_pattern_detections_paired += 1

            # add object points to global list

            objpoints_pairs.append(objp)

            # add previously refined corner locations to list

            imgpoints_left_paired.append(corners_sp_L)
            imgpoints_right_paired.append(corners_sp_R)

print("Pairs Detected: " + str(len(objpoints)))

# perform calibration on both cameras - uses [Zhang, 2000]

termination_criteria_intrinsic = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    iterations,
    minimum_error)

print("START - intrinsic calibration ...")

rms_int_L, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
    objpoints_left_only, imgpoints_left_only, grayL.shape[::-1],
    None, None, criteria=termination_criteria_intrinsic)
rms_int_R, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
    objpoints_right_only, imgpoints_right_only, grayR.shape[::-1],
    None, None, criteria=termination_criteria_intrinsic)

print("FINISHED - intrinsic calibration")

print()
print("LEFT: RMS left intrinsic calibation re-projection error: ",
        rms_int_L)
print("RIGHT: RMS right intrinsic calibation re-projection error: ",
        rms_int_R)
print()

frameL = cv2.imread(f"/media/AC10-0657/images/Set_3/Left/41.jpg")
frameR = cv2.imread(f"/media/AC10-0657/images/Set_3/Right/41.jpg")

undistortedL = cv2.undistort(frameL, mtxL, distL, None, None)
undistortedR = cv2.undistort(frameR, mtxR, distR, None, None)

# draw lines on image to observe quality of undistortion

undistortedL_lines = draw_lines(undistortedL, (10,10), (0, 0, 255), 1)
undistortedR_lines = draw_lines(undistortedR, (10,10), (0, 0, 255), 1)

tot_errorL = 0
for i in range(len(objpoints_left_only)):
    imgpoints_left_only2, _ = cv2.projectPoints(
        objpoints_left_only[i], rvecsL[i], tvecsL[i], mtxL, distL)
    errorL = cv2.norm(
        imgpoints_left_only[i],
        imgpoints_left_only2,
        cv2.NORM_L2) / len(imgpoints_left_only2)
    tot_errorL += errorL

print("LEFT: mean re-projection error (absolute, px): ",
        tot_errorL / len(objpoints_left_only))

tot_errorR = 0
for i in range(len(objpoints_right_only)):
    imgpoints_right_only2, _ = cv2.projectPoints(
        objpoints_right_only[i], rvecsR[i], tvecsR[i], mtxR, distR)
    errorR = cv2.norm(
        imgpoints_right_only[i],
        imgpoints_right_only2,
        cv2.NORM_L2) / len(imgpoints_right_only2)
    tot_errorR += errorR

print("RIGHT: mean re-projection error (absolute, px): ",
        tot_errorR / len(objpoints_right_only))

# STAGE 3: perform extrinsic calibration (recovery of relative camera
# positions)

# this takes the existing calibration parameters used to undistort the
# individual images as well as calculated the relative camera positions
# - represented via the fundamental matrix, F

# alter termination criteria to (perhaps) improve solution - ?

termination_criteria_extrinsics = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    iterations,
    minimum_error)

print()
print("START - extrinsic calibration ...")
(rms_stereo,
camera_matrix_l,
dist_coeffs_l,
camera_matrix_r,
dist_coeffs_r,
R,
T,
E,
F) = cv2.stereoCalibrate(objpoints_pairs,
                        imgpoints_left_paired,
                        imgpoints_right_paired,
                        mtxL,
                        distL,
                        mtxR,
                        distR,
                        grayL.shape[::-1],
                        criteria=termination_criteria_extrinsics,
                        flags=0)

print("FINISHED - extrinsic calibration")

print()
print("Intrinsic Camera Calibration:")
print()
print("Intrinsic Camera Calibration Matrix, K - from \
        intrinsic calibration:")
print("(format as follows: fx, fy - focal lengths / cx, \
        cy - optical centers)")
print("[fx, 0, cx]\n[0, fy, cy]\n[0,  0,  1]")
print()
print("Intrinsic Distortion Co-effients, D - from intrinsic calibration:")
print("(k1, k2, k3 - radial p1, p2 - tangential distortion coefficients)")
print("[k1, k2, p1, p2, k3]")
print()
print("K (left camera)")
print(camera_matrix_l)
print("distortion coeffs (left camera)")
print(dist_coeffs_l)
print()
print("K (right camera)")
print(camera_matrix_r)
print("distortion coeffs (right camera)")
print(dist_coeffs_r)

print()
print("Extrinsic Camera Calibration:")
print("Rotation Matrix, R (left -> right camera)")
print(R)
print()
print("Translation Vector, T (left -> right camera)")
print(T)
print()
print("Essential Matrix, E (left -> right camera)")
print(E)
print()
print("Fundamental Matrix, F (left -> right camera)")
print(F)

print()
print("STEREO: RMS left to  right re-projection error: ", rms_stereo)

# # define display window names

# window_nameL = "LEFT Camera Input"  # window name
# window_nameR = "RIGHT Camera Input"  # window name

# # create window by name (as resizable)

# cv2.namedWindow(window_nameL, cv2.WINDOW_NORMAL)
# cv2.namedWindow(window_nameR, cv2.WINDOW_NORMAL)

# # set sizes and set windows

# height, width, channels = undistortedL_lines.shape
# cv2.resizeWindow(window_nameL, width, height)
# height, width, channels = undistortedR_lines.shape
# cv2.resizeWindow(window_nameR, width, height)

# # display image

# cv2.imshow(window_nameL, undistortedL_lines)
# cv2.imshow(window_nameR, undistortedR_lines)
# cv2.waitKey(0)