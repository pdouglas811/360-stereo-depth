import cv2
import camera_stream

cap = camera_stream.CameraVideoStream("thetauvcsrc ! decodebin ! autovideoconvert ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink", cv2.CAP_GSTREAMER)
cap2 = camera_stream.CameraVideoStream("thetauvcsrc ! decodebin ! autovideoconvert ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink", cv2.CAP_GSTREAMER)

cv2.namedWindow("Stream 1", cv2.WINDOW_NORMAL)
cv2.namedWindow("Stream 2", cv2.WINDOW_NORMAL)


while True:
    
    ret, framef = cap.read()
    frame = cv2.flip(framef, -1)

    ret2, frame2 = cap2.read()

    frame = cv2.resize(frame, (800, 400))
    frame2 = cv2.resize(frame2, (800, 400))

    if (ret and ret2):
        
        cv2.imshow('Stream 1', frame)
        cv2.imshow('Stream 2', frame2)

        if cv2.waitKey(20) & 0xFF==ord('x'):
            break
    else:
        break

cap.release()
cap2.release()

cv2.destroyAllWindows()
