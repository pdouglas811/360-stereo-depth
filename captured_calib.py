import cv2
import numpy as np 
import os
import time
import sys

# Defines the entire code as a function

def captured_calib(iterations, minimum_error):

    # Define calibration dataset parameters

    num_frames = 10
    cbx = 6
    cby = 8
    square_size_in_mm = 80.8

    # Define Termination criteria for calibrations

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

                # add object points to left list

                objpoints_left_only.append(objp)

                # refine corner locations to sub-pixel accuracy and then add to list

                corners_sp_L = cv2.cornerSubPix(
                    grayL, cornersL, (11, 11), (-1, -1), termination_criteria_subpix)
                imgpoints_left_only.append(corners_sp_L)

            # -- > detected in right (only or also in left)

            if (retR):

                # add object points to left list

                objpoints_right_only.append(objp)

                # refine corner locations to sub-pixel accuracy and then add to list

                corners_sp_R = cv2.cornerSubPix(
                    grayR, cornersR, (11, 11), (-1, -1), termination_criteria_subpix)
                imgpoints_right_only.append(corners_sp_R)

            # -- > detected in left and right

            if ((retR) and (retL)):

                # add object points to global list

                objpoints_pairs.append(objp)

                # add previously refined corner locations to list

                imgpoints_left_paired.append(corners_sp_L)
                imgpoints_right_paired.append(corners_sp_R)

    print("Pairs Detected: " + str(len(objpoints_pairs)))

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

    RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(
        camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r,
        grayL.shape[::-1], R, T, alpha=-1)

    mapL1, mapL2 = cv2.initUndistortRectifyMap(
        camera_matrix_l, dist_coeffs_l, RL, PL, grayL.shape[::-1],
        cv2.CV_32FC1)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(
        camera_matrix_r, dist_coeffs_r, RR, PR, grayR.shape[::-1],
        cv2.CV_32FC1)

    print()
    print("-> displaying rectification")

    # undistort and rectify based on the mappings (could improve interpolation
    # and image border settase make sure you have the correct access rightings here)

    frameL = cv2.imread(f"/media/AC10-0657/images/Set_3/Left/41.jpg")
    frameR = cv2.imread(f"/media/AC10-0657/images/Set_3/Right/41.jpg")

    undistorted_rectifiedL = cv2.remap(frameL, mapL1, mapL2, cv2.INTER_LINEAR)
    undistorted_rectifiedR = cv2.remap(frameR, mapR1, mapR2, cv2.INTER_LINEAR)

    # draw lines on image to observe quality of undistortion and rectification

    undistorted_rectifiedL_lines = draw_lines(undistorted_rectifiedL, (10,10), (0, 0, 255), 1)
    undistorted_rectifiedR_lines = draw_lines(undistorted_rectifiedR, (10,10), (0, 0, 255), 1)

    # remember to convert to grayscale (as the disparity matching works on
    # grayscale)

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

     undistort and rectify based on the mappings (could improve interpolation
    # and image border settings here)
    # N.B. mapping works independant of number of image channels

    undistorted_rectifiedL = cv2.remap(grayL, mapL1, mapL2, cv2.INTER_LINEAR)
    undistorted_rectifiedR = cv2.remap(grayR, mapR1, mapR2, cv2.INTER_LINEAR)

    # compute disparity image from undistorted and rectified versions
    # (which for reasons best known to the OpenCV developers is returned
    # scaled by 16)

    disparity = stereoProcessor.compute(
        undistorted_rectifiedL, undistorted_rectifiedR)
    cv2.filterSpeckles(disparity, 0, 40, max_disparity)

    # scale the disparity to 8-bit for viewing
    # divide by 16 and convert to 8-bit image (then range of values should
    # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
    # so we fix this also using a initial threshold between 0 -> max_disparity
    # as disparity=-1 means no disparity available

    _, disparity = cv2.threshold(
        disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)
    disparity_scaled = (disparity / 16.).astype(np.uint8)

    # Saves calibrations and images generated

    try:
        os.mkdir('calibration')
    except OSError:
        print("Exporting to existing calibration archive directory.")
    os.chdir('calibration')
    folderName = time.strftime('%d-%m-%y-%H-%M-rms-') + \
        '%.2f' % rms_stereo
    os.mkdir(folderName)
    os.chdir(folderName)
    np.save('mapL1', mapL1)
    np.save('mapL2', mapL2)
    np.save('mapR1', mapR1)
    np.save('mapR2', mapR2)
    cv_file = cv2.FileStorage("calibration.xml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("source", ' '.join(sys.argv[0:]))
    cv_file.write(
        "description",
        "camera matrices K for left and right, distortion coefficients " +
        "for left and right, 3D rotation matrix R, 3D translation " +
        "vector T, Essential matrix E, Fundamental matrix F, disparity " +
        "to depth projection matrix Q")
    cv_file.write("K_l", camera_matrix_l)
    cv_file.write("K_r", camera_matrix_r)
    cv_file.write("distort_l", dist_coeffs_l)
    cv_file.write("distort_r", dist_coeffs_r)
    cv_file.write("R", R)
    cv_file.write("T", T)
    cv_file.write("E", E)
    cv_file.write("F", F)
    cv_file.write("Q", Q)
    cv_file.release()
    print()
    print("Exported to path: ", folderName)
    print()

    pathL = "/media/AC10-0657/images/Calibrations/Left"
    pathR = "/media/AC10-0657/images/Calibrations/Right"
    pathD = "/media/AC10-0657/images/Disparity"

    imgName = folderName + '-%f' % iterations + '-%f' % minimum_error

    def save_my_img(imgName, path, img):
        name = f'{imgName}.jpg'
        cv2.imwrite(os.path.join(path, name), img)

    # Save undistorted rectified image for calibration quality check

    save_my_img(imgName, pathL, undistorted_rectifiedL_lines)
    save_my_img(imgName, pathR, undistorted_rectifiedR_lines)
    print()
    print("Saved calibrated images to:", imgName)
    print()

    # Save disparity map for calibration quality check

    save_my_img(imgName, pathD, disparity_scaled)
    print()
    print("Saved calibrated images to:", imgName)
    print()

    # # define display window names

    # window_nameL = "LEFT Camera Input"  # window name
    # window_nameR = "RIGHT Camera Input"  # window name

    # # create window by name (as resizable)

    # cv2.namedWindow(window_nameL, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(window_nameR, cv2.WINDOW_NORMAL)

    # # set sizes and set windows

    # height, width, channels = undistorted_rectifiedL_lines.shape
    # cv2.resizeWindow(window_nameL, width, height)
    # height, width, channels = undistorted_rectifiedR_lines.shape
    # cv2.resizeWindow(window_nameR, width, height)

    # # display image

    # cv2.imshow(window_nameL, undistorted_rectifiedL_lines)
    # cv2.imshow(window_nameR, undistorted_rectifiedR_lines)
    # cv2.waitKey(0)

iterations = [100]#, 200, 300, 400, 500, 600, 700, 800, 1000]
minimum_error = [0.1]#, 0.01, 0.001, 0.0001, 0.00001]

for i in iterations:

    for j in minimum_error:

        captured_calib(i, j)
        
        print()
        print("Calibration complete!")
        print()

print()
print("All calibrations complete!")
print()

# captured_calib(iterations, minimum_error)