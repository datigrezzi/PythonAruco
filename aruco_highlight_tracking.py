import numpy as np
import cv2
import os
import time
from distutils import util
import argparse

parser = argparse.ArgumentParser(description = 'Marker Tracking & Pose Estimation')
parser.add_argument('--inputVideo', type = str, help = 'Path to the video of the object to be tracked')
parser.add_argument('--referenceImage', type = str, help = 'Path to an image of the object to track including markers')
parser.add_argument('--outputVideo', type = str, default = None, help = 'Optional - Path to output video')
parser.add_argument('--vector', type = util.strtobool, default = True, help = 'Should pose vector be drawn?')
parser.add_argument('--smooth', type = util.strtobool, default = False, help = 'Should smooth transformation matrix?')
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

# camera matrix estimate
focal_length = cap.get(3)
center = (cap.get(3)/2, cap.get(4)/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
 
dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

# a little helper function for getting all dettected marker ids
# from the reference image markers
def which(x, values):
    indices = []
    for ii in list(values):
        if ii in x:
            indices.append(list(x).index(ii))
    return indices

if args.smooth:
    # for simple noise reduction we use deque
    from collections import deque
    # simple noise reduction
    h_array = deque(maxlen = 5)

while(True):
    # read next frame from VideoCapture
    ret, frame = cap.read()
    if frame is not None:
        # convert frame to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect aruco markers in gray frame
        res_corners, res_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters = parameters)
        # if markers were detected
        if res_ids is not None:
            # find which markers in frame match those in reference image
            idx = which(refIds, res_ids)
            # if any detected marker in frame is also in the reference image
            if len(idx) > 0:
                # flatten the array of corners in the frame and reference image
                these_res_corners = np.concatenate(res_corners, axis = 1)
                these_ref_corners = np.concatenate([refCorners[x] for x in idx], axis = 1)
                # estimate homography matrix
                h, s = cv2.findHomography(these_ref_corners, these_res_corners, cv2.RANSAC, 5.0)
                # if we want smoothing
                if args.smooth:
                    h_array.append(h)
                    this_h = np.mean(h_array, axis = 0)
                else:
                    this_h = h
                # transform the rectangle using the homography matrix
                newRect = cv2.perspectiveTransform(rect, this_h, (gray.shape[1],gray.shape[0]))
                # draw the rectangle on the frame
                frame = cv2.polylines(frame, np.int32(newRect), True, (0,0,0), 10)
                # if we want the pose estimation
                if args.vector:
                    # add a distance estimate to the markers in the reference image (Z axis from camera) - here it is 50
                    these_ref_corners_3d = np.append(these_ref_corners, np.add(np.zeros((1,these_ref_corners.shape[1],1)), 500), axis = 2)
                    # estimate rotation and translation vectors with solvePnP
                    success, rotation_vector, translation_vector = cv2.solvePnP(these_ref_corners_3d, these_res_corners, camera_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)
                    # project the start and end points of the pose vector from reference image to frame
                    (pose_point_2d, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 500.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                    p1 = (int(pose_point_2d[0][0][0]), int(pose_point_2d[0][0][1]))
                    p2 = (int(pose_point_2d[1][0][0]), int(pose_point_2d[1][0][1]))
                    # draw the veector as a line in the frame
                    cv2.line(frame, p1, p2, (0,0,255), 10)
            # draw detected markers in frame with their ids
            cv2.aruco.drawDetectedMarkers(frame,res_corners,res_ids)
        # if video is to be saved
        if args.outputVideo is not None:
            videoOut.write(frame)
        # resize frame to half of both axes (because my screen is small!)
        frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5)
        # Display the resulting frame
        cv2.imshow('frame',frame)
    # exit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and video output
if args.outputVideo is not None:
    videoOut.release()
cap.release()
# close cv2 window
cv2.destroyAllWindows()