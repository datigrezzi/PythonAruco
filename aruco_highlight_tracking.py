import numpy as np
import cv2
import os
import time
# for simple noise reduction we use deque
from collections import deque
# passing arguments from command line interface
import argparse
parser = argparse.ArgumentParser(description = 'Marker Tracking & Pose Estimation')
parser.add_argument('--inputVideo', type = str, default = False, help = 'Path to the video of the object to be tracked')
parser.add_argument('--referenceImage', type = str, help = 'Path to an image of the object to track including markers')
parser.add_argument('--outputVideo', type = str, default = None, help = 'Optional - Path to output video')
args = parser.parse_args()

# load video file
cap = cv2.VideoCapture(args.inputVideo)
# open video file for writing
if args.outputVideo is not None:
    videoOut = cv2.VideoWriter(args.outputVideo, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(5)), (int(cap.get(3)), int(cap.get(4))))

# define an empty custom dictionary with 
aruco_dict = cv2.aruco.custom_dictionary(0, 4, 1)
# add empty bytesList array to fill with 3 markers later
aruco_dict.bytesList = np.empty(shape = (3, 2, 4), dtype = np.uint8)
# add new marker(s)
mybits = np.array([[1,1,0,0],[1,1,0,1],[0,1,1,1],[1,1,1,1]], dtype = np.uint8)
aruco_dict.bytesList[0] = cv2.aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[0,1,0,0],[0,1,1,0],[1,0,1,0],[1,1,1,0]], dtype = np.uint8)
aruco_dict.bytesList[1] = cv2.aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[1,1,1,0],[0,0,1,1],[1,1,1,1],[0,1,0,0]], dtype = np.uint8)
aruco_dict.bytesList[2] = cv2.aruco.Dictionary_getByteListFromBits(mybits)

# adjust dictionary parameters for better marker detection
parameters =  cv2.aruco.DetectorParameters_create()
parameters.cornerRefinementMethod = 3
parameters.errorCorrectionRate = 0.2

# load reference image
refImage = cv2.cvtColor(cv2.imread(args.referenceImage), cv2.COLOR_BGR2GRAY)
# detect markers in reference image
refCorners, refIds, refRejected = cv2.aruco.detectMarkers(refImage, aruco_dict, parameters = parameters)
# create bounding box from reference image dimensions
rect = np.array([[[0,0],
                  [refImage.shape[1],0],
                  [refImage.shape[1],refImage.shape[0]],
                  [0,refImage.shape[0]]]], dtype = "float32")

# a little helper function for getting dettected marker ids
def which(x, value):
    indices = []
    for i, ii in enumerate(list(x)):
        if ii == value:
            indices.append(i)
    return indices

# simple noise reduction
h_array = deque(maxlen = 15)

while(True):
    ret, frame = cap.read()
    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.aruco.detectMarkers(gray, aruco_dict, parameters = parameters)
        if res[1] is not None:
            idx = which(refIds, res[1][0])
            if len(idx) > 0:
                refMarkerIdx = idx[0]
                h, s = cv2.findHomography(refCorners[refMarkerIdx], res[0][0])
                h_array.append(h)
                smooth_h = np.mean(h_array, axis = 0)
                newRect = cv2.perspectiveTransform(rect, smooth_h, (gray.shape[1],gray.shape[0]))
                frame = cv2.polylines(frame, np.int32(newRect), True, (0,0,0), 10)
        if len(res[0]) > 0:
            cv2.aruco.drawDetectedMarkers(frame,res[0],res[1])
        # Display the resulting frame
        if args.outputVideo is not None:
            videoOut.write(frame)
        frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5)
        cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
if args.outputVideo is not None:
    videoOut.release()
cap.release()
cv2.destroyAllWindows()