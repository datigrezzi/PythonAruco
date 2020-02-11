import numpy as np
import cv2
from collections import deque
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# passing arguments from command line interface
import argparse
parser = argparse.ArgumentParser(description = 'Marker Tracking & Pose Estimation')
parser.add_argument('--inputVideo', type = str, help = 'Path to the video of the object to be tracked')
parser.add_argument('--referenceImage', type = str, help = 'Path to an image of the object to track including markers')
parser.add_argument('--object', type = str, help = 'Path to an image of the object to track including markers')
parser.add_argument('--outputVideo', type = str, help = 'Optional - Path to output video')
args = parser.parse_args()

# helper functions

# getting dettected marker ids
def which(x, values):
    indices = []
    for ii in list(values):
        if ii in x:
            indices.append(list(x).index(ii))
    return indices

# generate custom aruco dicitonary and parameters
def customAruco():
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
    return aruco_dict, parameters

# load 3d object file (.obj)
def read_obj(filename, center_origin = True, flip_vertical = False, flip_yz = True, scale = 1):
    triangles = []
    vertices = []
    with open(filename) as file:
        for line in file:
            components = line.strip(' \n').split(' ')
            if components[0] == "f": # face data
                indices = list(map(lambda c: int(c.split('/')[0]) - 1, components[2:]))
                for i in range(0, len(indices) - 2):
                    triangles.append(indices[i: i+3])
            elif components[0] == "v": # vertex data
                vertex = list(map(lambda c: float(c), components[2:]))
                vertices.append(vertex)
    # center coordinates origin to center of object
    vertices = np.array(vertices)
    if center_origin:
        vertices[:,0] = vertices[:,0] - np.divide(vertices[:,0].max(),2)
        vertices[:,1] = vertices[:,1] - np.divide(vertices[:,1].max(),2)
        vertices[:,2] = vertices[:,2] - np.divide(vertices[:,2].min(),2)
    if scale != 1:
        vertices = np.multiply(vertices, scale)
    # flip object on vertical axis, assuming it is centered at zero
    if flip_vertical:
        vertices[:,1] = vertices[:,1] * -1
    if flip_yz:
        y = vertices[:,1]
        z = vertices[:,2]
        vertices[:,1] = z
        vertices[:,2] = y
    return vertices, np.array(triangles)

# convert pyplot canvas to an RGB numpy array
def canvas2rgb_array(canvas):
    canvas.draw()
    buf = canvas.tostring_rgb()
    ncols, nrows = canvas.get_width_height()
    return np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

# merge two images horizontally to the height of the second image
def merge_images(image1, image2, horiz = True):
    # if images have different heights
    if horiz:
        if image1.shape[0] != image2.shape[0]:
            # calculate the size for the first image to have same height as second while keeping the proportion
            newsize = (int(round((image1.shape[1]*image2.shape[0])/image1.shape[0])), image2.shape[0])
            # resize first image
            image1 = cv2.resize(image1, newsize)
        # if second image has alpha channel, remove it!
        if image2.shape[2] > image1.shape[2]:
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGBA2RGB)
        # return an array with both images merged horizontally
        return np.concatenate((image1, image2), axis=1)
    else:
        if image1.shape[1] != image2.shape[1]:
            newsize = (int(round((image1.shape[0]*image2.shape[1])/image1.shape[1])), image2.shape[1])
            # resize first image
            image1 = cv2.resize(image1, newsize)
        # if second image has alpha channel, remove it!
        if image2.shape[2] > image1.shape[2]:
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGBA2RGB)
        # return an array with both images merged horizontally
        return np.concatenate((image1, image2), axis=0)

# 3d plot an object and return axes for further plotting
def plot3d(vertices, triangles, elevation, azimut, size = (6,6), lims = [-50, 50], axes = True, draw = True, flat = False):
    figure = plt.figure(figsize = size)
    ax = plt.gca(projection='3d')
    ax.view_init(elev=elevation, azim=azimut)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_zlim(lims)
    plt.subplots_adjust(bottom=0, left=0, right=1, top=1)
    if not axes:
        ax.axis('off')
    if draw:
        if flat:
            ax.plot(vertices[:,0], vertices[:,1], color='green')
        else:
            ax.plot_trisurf(vertices[:,0], vertices[:,1], triangles, vertices[:,2], shade=True, color='green')
    return ax

# create aruco dict and params
aruco_dict, parameters = customAruco()
# load 3D object
vertices, triangles = read_obj(args.object, flip_vertical=True, flip_yz=False, scale=3)
# load video file
cap = cv2.VideoCapture(args.inputVideo)
total_frames = cap.get(7)
# load reference image
refImage = cv2.cvtColor(cv2.imread(args.referenceImage), cv2.COLOR_BGR2GRAY)
# detect markers in reference image
refCorners, refIds, refRejected = cv2.aruco.detectMarkers(refImage, aruco_dict, parameters = parameters)

# camera matrix parameters; for better results, these should be calculated for each camera
focal_length = cap.get(3)
center = (cap.get(3)/2, cap.get(4)/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

# create a list with fixed length, for simple noise reduction
pos_list_len = 15
camera_position_list = deque(maxlen=pos_list_len)

# create plots
elev_list = [0, -90, 0] # top, front, side
azim_list = [-90, -90, 0]

# add plots to a list to update in a loop later
plots = []
for pl in range(len(elev_list)):
    plots.append(plot3d(vertices, triangles, elevation = elev_list[pl], azimut = azim_list[pl], axes = False, draw = True))

# open video file for writing
if args.outputVideo is not None:
    # read first frame and merge to get final frame size
    _ , frame = cap.read()
    merged_frame_1 = merge_images(canvas2rgb_array(plots[0].figure.canvas), canvas2rgb_array(plots[1].figure.canvas))
    merged_frame_2 = merge_images(frame, np.rot90(canvas2rgb_array(plots[2].figure.canvas),3))
    finalFrameSize = merge_images(merged_frame_1, merged_frame_2, horiz = False).shape
    videoOut = cv2.VideoWriter(args.outputVideo, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(5)), (int(finalFrameSize[1]), int(finalFrameSize[0])))

lastPosition = None
frameN = -1
while frameN < total_frames:
    ret, frame = cap.read()
    frameN = int(cap.get(1))
    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res_corners, res_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters = parameters)
        if res_ids is not None:
            idx = which(refIds, res_ids)
            if len(idx) > 0:
                these_res_corners = np.concatenate(res_corners, axis = 1)
                these_ref_corners = np.concatenate([refCorners[x] for x in idx], axis = 1)
                these_ref_corners = np.append(these_ref_corners, np.add(np.zeros((1,these_ref_corners.shape[1],1)), 50), axis = 2)
                success, rotation_vector, translation_vector = cv2.solvePnP(these_ref_corners, these_res_corners, camera_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                camera_position = -np.matrix(rotation_matrix).T * np.matrix(translation_vector)
                camera_position = np.divide(camera_position, 100).tolist()
                # add current position to buffer
                camera_position_list.append(camera_position)
                # gather enough data in list for noise reduction
                if len(camera_position_list) == pos_list_len:
                    x_nr = np.median([x[0][0] for x in camera_position_list])
                    y_nr = np.median([x[1][0] for x in camera_position_list])
                    z_nr = np.median([x[2][0] for x in camera_position_list])
                    thisPosition = [x_nr, y_nr, z_nr]
                    if lastPosition is not None:
                        x = [lastPosition[0], thisPosition[0]]
                        y = [lastPosition[1], thisPosition[1]]
                        z = [lastPosition[2], thisPosition[2]]
                        for pl in range(len(elev_list)):
                            plots[pl].plot(x , y, z, "k", linewidth = 2)
                    lastPosition = thisPosition
                    print("Progress: %s" %int(round((frameN + 1) * 100 / total_frames)), end = "\r")
                else:
                    nr_progress = round((len(camera_position_list)+1)/pos_list_len*100)
                    print("Collecting samples for noise reduction %s" %nr_progress, end = "\r")
                    if nr_progress == 100:
                        print("")

        # Display the resulting frame
        merged_frame_1 = merge_images(canvas2rgb_array(plots[0].figure.canvas), canvas2rgb_array(plots[1].figure.canvas))
        merged_frame_2 = merge_images(frame, np.rot90(canvas2rgb_array(plots[2].figure.canvas), 3))
        merged_frame = merge_images(merged_frame_1, merged_frame_2, horiz = False)
        if args.outputVideo is not None:
            videoOut.write(merged_frame)
        cv2.imshow('frame',merged_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
if args.outputVideo is not None:
    videoOut.release()
cap.release()
cv2.destroyAllWindows()