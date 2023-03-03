import cv2
import numpy as np 

# Define calibration dataset parameters

num_frames = 64
cbx = 6
cby = 8
square_size_in_mm = 80.8

# Define Termination criteria for calibrations

iterations = 100
minimum_error = 0.001
termination_criteria_subpix = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    args.iterations,
    args.minimum_error)

# Define function for drawing lines on rectified output images

def draw_lines(img, grid_shape, color, thickness):
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        cv.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

objp = np.zeros((patternX * patternY, 3), np.float32)
objp[:, :2] = np.mgrid[0:patternX, 0:patternY].T.reshape(-1, 2)
objp = objp * square_size_in_mm

# create arrays to store object points and image points from all the images

objpoints = []    # 3d point in real world space
imgpoints_R = []  # 2d points in image plane
imgpoints_L = []  # 2d points in image plane

# Begin loop to find chessboard corners

for i in range(1, num_frames + 1):
        
        # Fetch image pairs from relevant directory

        frameL = cv2.imread(f"/media/AC10-0657/images/Set_2/Left/{i}.jpg")
        frameR = cv2.imread(f"/media/AC10-0657/images/Set_2/Right/{i}.jpg")

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

# perform calibration on both cameras - uses [Zhang, 2000]

termination_criteria_intrinsic = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    args.iterations,
    args.minimum_error)

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

    