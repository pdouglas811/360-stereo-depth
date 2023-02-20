#####################################################################

# Software to capture stereo images of chessboards for the purpose of
# automated calibration testing

import cv2
import sys
import numpy as np
import os
import argparse
import time
import math
import camera_stream

#####################################################################
# define target framerates in fps (for calibration step only)

calibration_capture_framerate = 2

#####################################################################
# wrap different kinds of stereo camera - standard (v4l/vfw), ximea, ZED


class StereoCamera:
    def __init__(self, args):

        self.xiema = args.ximea
        self.zed = args.zed
        self.cameras_opened = False

        if args.ximea:

            # ximea requires specific API offsets in the open commands

            self.camL = cv2.VideoCapture()
            self.camR = cv2.VideoCapture()

            if not(
                (self.camL.open(
                    cv2.CAP_XIAPI)) and (
                    self.camR.open(
                        cv2.CAP_XIAPI +
                    1))):
                print("Cannot open pair of Ximea cameras connected.")
            exit()

        elif args.zed:

            # ZED is a single camera interface with L/R returned as 1 image

            try:
                # to use a non-buffered camera stream (via a separate thread)
                # no T-API use, unless additional code changes later

                import camera_stream
                self.camZED = camera_stream.CameraVideoStream(0)

            except BaseException:
                # if not then just use OpenCV default

                print("INFO: camera_stream class not found - \
                        camera input may be buffered")
                self.camZED = cv2.VideoCapture()

            if not(self.camZED.open(args.camera_to_use)):
                print(
                    "Cannot open connected ZED stereo camera as camera #: ",
                    args.camera_to_use)
                exit()

            # report resolution currently in use for ZED (as various
            # options exist) can use .get()/.set() to read/change also

            _, frame = self.camZED.read()
            height, width, channels = frame.shape
            print()
            print("ZED left/right resolution: ",
                  int(width / 2), " x ", int(height))
            print()

        else:

            # by default two standard system connected cams from the default
            # video backend
            import camera_stream
            self.camL = camera_stream.CameraVideoStream()
            self.camR = camera_stream.CameraVideoStream()
            if not(
                (self.camL.open("thetauvcsrc ! decodebin ! autovideoconvert ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink", cv2.CAP_GSTREAMER)) and (
                    self.camR.open("thetauvcsrc ! decodebin ! autovideoconvert ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink", cv2.CAP_GSTREAMER))):
                print(
                    "Cannot open pair of system cameras connected \
                    starting at camera #:",
                    args.camera_to_use)
                exit()

        self.cameras_opened = True

    def swap_cameras(self):
        if not(self.zed):
            # swap the cameras - for all but ZED camera
            tmp = self.camL
            self.camL = self.camR
            self.camR = tmp

    def get_frames(self):  # return left, right
        if self.zed:

            # grab single frame from camera (read = grab/retrieve)
            # and split into Left and Right

            _, frame = self.camZED.read()
            height, width, channels = frame.shape
            frameL = frame[:, 0:int(width / 2), :]
            frameR = frame[:, int(width / 2):width, :]
        else:
            # grab frames from camera (to ensure best time sync.)

            self.camL.grab()
            self.camR.grab()

            # then retrieve the images in slow(er) time
            # (do not be tempted to use read() !)

            _, frameL = self.camL.retrieve()
            _, frameRf = self.camR.retrieve()
            frameR = cv2.flip(frameRf, -1) #to work with the setup of having one of the cameras upside down

            # Because the cameras are further apart in the vertical rather than horizontal, must rotate each image by 90 deg.
            
            frameL = cv2.rotate(frameL, cv2.ROTATE_90_CLOCKWISE)
            frameR = cv2.rotate(frameR, cv2.ROTATE_90_CLOCKWISE)

        return frameL, frameR

#####################################################################
# deal with optional arguments


parser = argparse.ArgumentParser(
    description='Perform full stereo calibration and SGBM matching.')
parser.add_argument(
    "--ximea",
    help="use a pair of Ximea cameras",
    action="store_true")
parser.add_argument(
    "--zed",
    help="use a Stereolabs ZED stereo camera",
    action="store_true")
parser.add_argument(
    "-c",
    "--camera_to_use",
    type=int,
    help="specify camera to use",
    default=0)
parser.add_argument(
    "-cbx",
    "--chessboardx",
    type=int,
    help="specify number of internal chessboard squares (corners) \
            in x-direction",
    default=8)
parser.add_argument(
    "-cby",
    "--chessboardy",
    type=int,
    help="specify number of internal chessboard squares (corners) \
        in y-direction",
    default=6)
parser.add_argument(
    "-cbw",
    "--chessboardw",
    type=float,
    help="specify width/height of chessboard squares in mm",
    default=80.8)
parser.add_argument(
    "-cp",
    "--calibration_path",
    type=str,
    help="specify path to calibration files to load",
    default=-1)
parser.add_argument(
    "-i",
    "--iterations",
    type=int,
    help="specify number of iterations for each stage of optimisation",
    default=100)
parser.add_argument(
    "-e",
    "--minimum_error",
    type=float,
    help="specify lower error threshold upon which to stop \
            optimisation stages",
    default=0.001)

args = parser.parse_args()

#####################################################################

# flag values to enter processing loops - do not change

keep_processing = True
do_calibration = False

#####################################################################

# STAGE 1 - open 2 connected cameras

# define video capture object

stereo_camera = StereoCamera(args)

# define display window names

window_nameL = "LEFT Camera Input"  # window name
window_nameR = "RIGHT Camera Input"  # window name

# create window by name (as resizable)

cv2.namedWindow(window_nameL, cv2.WINDOW_NORMAL)
cv2.namedWindow(window_nameR, cv2.WINDOW_NORMAL)

# set sizes and set windows

frameL, frameR = stereo_camera.get_frames()

height, width, channels = frameL.shape
cv2.resizeWindow(window_nameL, width, height)
height, width, channels = frameR.shape
cv2.resizeWindow(window_nameR, width, height)

# controls

print("s : swap cameras left and right")
print("e : export camera calibration to file")
print("l : load camera calibration from file")
print("x : exit")
print()
print("space : continue to next stage")
print()

while (keep_processing):

    # get frames from camera

    frameL, frameR = stereo_camera.get_frames()

    # display image
    
    cv2.imshow(window_nameL, frameL)
    cv2.imshow(window_nameR, frameR)

    # start the event loop - essential

    key = cv2.waitKey(2) & 0xFF  # wait 2ms

    # loop control - space to continue; x to exit; s - swap cams; l - load

    if (key == ord(' ')):
        keep_processing = False
    elif (key == ord('x')):
        exit()
    elif (key == ord('s')):
        # swap the cameras if specified

        stereo_camera.swap_cameras()
    elif (key == ord('l')):

        if (args.calibration_path == -1):
            print("Error - no calibration path specified:")
            exit()

        # load calibration from file

        os.chdir(args.calibration_path)
        print('Using calibration files: ', args.calibration_path)
        mapL1 = np.load('mapL1.npy')
        mapL2 = np.load('mapL2.npy')
        mapR1 = np.load('mapR1.npy')
        mapR2 = np.load('mapR2.npy')

        keep_processing = False
        do_calibration = True  # set to True to skip next loop

#####################################################################

# STAGE 2: perform intrinsic calibration (removal of image distortion in
# each image)

termination_criteria_subpix = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    args.iterations,
    args.minimum_error)

# set up a set of real-world "object points" for the chessboard pattern

patternX = args.chessboardx
patternY = args.chessboardy
square_size_in_mm = args.chessboardw

if (patternX == patternY):
    print("*****************************************************************")
    print()
    print("Please use a chessboard pattern that is not equal dimension")
    print("in X and Y (otherwise a rotational ambiguity exists!).")
    print()
    print("*****************************************************************")
    print()
    exit()

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

objp = np.zeros((patternX * patternY, 3), np.float32)
objp[:, :2] = np.mgrid[0:patternX, 0:patternY].T.reshape(-1, 2)
objp = objp * square_size_in_mm

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

chessboard_pattern_detections_paired = 0
chessboard_pattern_detections_left = 0
chessboard_pattern_detections_right = 0

print()
print("--> hold up chessboard")
print("press space when ready to start calibration stage  ...")
print()

while (not(do_calibration)):

    # get frames from camera

    frameL, frameR = stereo_camera.get_frames()

    # convert to grayscale

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners in the images
    # (change flags to perhaps improve detection - see OpenCV manual)

    retR, cornersR = cv2.findChessboardCorners(
        grayR, (patternX, patternY), None, cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
    retL, cornersL = cv2.findChessboardCorners(
        grayL, (patternX, patternY), None, cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)

    # when found, add object points, image points (after refining them)

    # N.B. to allow for maximal coverage of the FoV of the L and R images
    # for instrinsic calc. without an image region being underconstrained
    # record and process detections for 3 conditions in 3 differing list
    # structures

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

        # save chessboard pairs to directory for use later

        def save_my_img(frame_count, path, img):
            name = f'{frame_count}.jpg'
            cv2.imwrite(os.path.join(path, name), img)

    save_my_img(chessboard_pattern_detections_paired, path, frameL)
    save_my_img(chessboard_pattern_detections_paired, path, frameR)    




    # display detections / chessboards

    text = 'detected L: ' + str(chessboard_pattern_detections_left) + \
           ' detected R: ' + str(chessboard_pattern_detections_right)
    cv2.putText(frameL, text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 8)
    text = 'detected (L AND R): ' + str(chessboard_pattern_detections_paired)
    cv2.putText(frameL, text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 8)

    # draw the corners / chessboards

    drawboardL = cv2.drawChessboardCorners(
                            frameL, (patternX, patternY), cornersL, retL)
    drawboardR = cv2.drawChessboardCorners(
                            frameR, (patternX, patternY), cornersR, retR)
    cv2.imshow(window_nameL, drawboardL)
    cv2.imshow(window_nameR, drawboardR)

    # start the event loop

    key = cv2.waitKey(int(1000 / calibration_capture_framerate)) & 0xFF
    if (key == ord(' ')):
        do_calibration = True
    elif (key == ord('x')):
        exit()

# perform calibration on both cameras - uses [Zhang, 2000]

termination_criteria_intrinsic = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    args.iterations,
    args.minimum_error) display detections / chessboards