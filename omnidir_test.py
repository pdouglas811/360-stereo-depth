import cv2
import numpy as np 

# Define calibration dataset parameters

num_frames = 53
cbx = 6
cby = 8
square_size_in_mm = 80.8

# Define Termination criteria for calibrations

iterations = 100
minimum_error = 0.001
termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, minimum_error)
calib_flags = cv2.omnidir.CALIB_FIX_SKEW # + cv2.omnidir.CALIB_USE_GUESS

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

objp = np.zeros((cbx * cby, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbx, 0:cby].T.reshape(-1, 2)
objp = objp * square_size_in_mm

# create arrays to store object points and image points from all the images

objpoints = []    # 3d point in real world space
imgpoints_R = []  # 2d points in image plane
imgpoints_L = []  # 2d points in image plane

# Begin loop to find chessboard corners

for i in range(1, num_frames + 1):
        
        # Fetch image pairs from relevant directory

        frameL = cv2.imread(f"/media/AC10-0657/images/Set_1/Left/{i}.jpg")
        frameR = cv2.imread(f"/media/AC10-0657/images/Set_1/Right/{i}.jpg")

        # Convert image pairs to grayscale

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners in the images

        retL, cornersL = cv2.findChessboardCorners(grayL, (cbx, cby))
        retR, cornersR = cv2.findChessboardCorners(grayR, (cbx, cby))

        if ((retR) and (retL)):

                # add object points to global list

                objpoints.append(objp)

                # refine corner locations to sub-pixel accuracy

                corners_sp_L = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), termination_criteria)
                corners_sp_R = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), termination_criteria)

                # add previously refined corner locations to list

                imgpoints_L.append(corners_sp_L)
                imgpoints_R.append(corners_sp_R)

print("Pairs Detected: " + str(len(objpoints)))

termination_criteria_intrinsic = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, minimum_error)

print("START - intrinsic calibration ...")

rms_int_L, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
        objpoints, imgpoints_L, grayL.shape[::-1],
        None, None, criteria=termination_criteria_intrinsic)
rms_int_R, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
        objpoints, imgpoints_R, grayR.shape[::-1],
        None, None, criteria=termination_criteria_intrinsic)
print("FINISHED - intrinsic calibration")

print()
print("LEFT: RMS left intrinsic calibation re-projection error: ",  rms_int_L)
print("RIGHT: RMS right intrinsic calibation re-projection error: ", rms_int_R)
print()

undistortedL = cv2.undistort(frameL, mtxL, distL, None, None)
undistortedR = cv2.undistort(frameR, mtxR, distR, None, None)

# display image

cv2.imshow("Left", undistortedL)
cv2.imshow("Right", undistortedR)
cv2.waitKey(0)

# # Need to resize objPoints from (num_detected_images, num_chessboard_squares, 3) to (num_detected_images, 1, num_chessboard_squares, 3) 
# # to avoid an error in omnidir.stereoCalibrate

# objpoints = np.reshape(objpoints,(len(objpoints), 1, cby * cbx, 3))


# # Need to resize imgPoints from (num_detected_images, num_chessboard_squares, 2) to (num_detected_images, 1, num_chessboard_squares, 2) for same L = nL), 1, cby * cbx, 2))

# imgpoints_L = np.reshape(imgpoints_L, (len(imgpoints_L), 1, cby * cbx, 2))
# imgpoints_R = np.reshape(imgpoints_R, (len(imgpoints_R), 1, cby * cbx, 2))

# # now calibrate

# retval, objPoints, imgPoints_1, imgPoints_2, K1, \
# xiL, D1, K2, xiR, D2, rvec, \
# tvec, rvecs, tvecs, idx = cv2.omnidir.stereoCalibrate(objpoints,imgpoints_L,imgpoints_R,
#                                                               grayL.shape[::-1], grayR.shape[::-1],
#                                                               None, None, None, None, None, None, #K1, xi_1, D1, K1, xi_2, D2,
#                                                               calib_flags, termination_criteria)

# print()
# print("K (left camera)")
# print(K1)
# print("distortion coeffs (left camera)")
# print(D1)
# print()
# print("K (right camera)")
# print(K2)
# print("distortion coeffs (right camera)")
# print(D1)                          
# print()
# print(grayL.shape[::-1])

# (imgHeight, imgWidth) = grayR.shape[::-1]

# R1F, R2F = cv2.omnidir.stereoRectify(rvec, tvec)

# knew = np.array([[imgWidth / (np.pi), 0, 0], [0, imgHeight / (np.pi), 0], [0, 0, 1]], np.double)

# mapL1, mapL2 = cv2.omnidir.initUndistortRectifyMap(K1, D1, xiL, R1F, knew, (imgHeight, imgWidth), cv2.CV_32FC1, cv2.omnidir.RECTIFY_LONGLATI)
# mapR1, mapR2 = cv2.omnidir.initUndistortRectifyMap(K2, D2, xiR, R2F, knew, (imgHeight, imgWidth), cv2.CV_32FC1, cv2.omnidir.RECTIFY_LONGLATI)

# undistorted_L = cv2.remap(frameL, mapL1, mapL2, cv2.INTER_LINEAR)
# undistorted_R = cv2.remap(frameR, mapR1, mapR2, cv2.INTER_LINEAR)

# cv2.imshow("Left", undistorted_L)
# cv2.imshow("Right", undistorted_R)
# cv2.waitKey(0)