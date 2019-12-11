import numpy as np
import cv2
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# passing arguments from command line interface
import argparse
parser = argparse.ArgumentParser(description = 'Marker Tracking & 3D Object Rotation')
parser.add_argument('--inputVideo', type = str, default="/Users/iyad/projects/python/aruco_homography/materials/IMG_4471.MOV", help = 'Path to the video of the object to be tracked')
parser.add_argument('--referenceImage', type = str, default = "/Users/iyad/projects/python/aruco_homography/materials/IMG_4468.jpg", help = 'Path to an image of the object to track including markers')
parser.add_argument('--object', type = str, default = "/Users/iyad/Downloads/model.dae.obj", help = 'Path to an image of the object to track including markers')
parser.add_argument('--outputVideo', type = str, default = "/Users/iyad/projects/python/aruco_homography/materials/object_rotation.mp4", help = 'Optional - Path to output video')
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
def read_obj(filename):
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
    # center coordinates origin
    centered_vertices = np.array(vertices)
    centered_vertices[:,0] = centered_vertices[:,0] - np.divide(centered_vertices[:,0].max(),2)
    centered_vertices[:,1] = centered_vertices[:,1] - np.divide(centered_vertices[:,1].max(),2)
    centered_vertices[:,2] = centered_vertices[:,2] - np.divide(centered_vertices[:,2].min(),2)
    return centered_vertices, np.array(triangles)

# plot 3d object with pyplot
def plot3d(vertices, triangles):
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca(projection='3d')
    ax.view_init(elev=270, azim=270)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])
    ax.plot_trisurf(vertices[0], vertices[1], triangles, vertices[2], shade=True, color='green')
    ax.axis('off')
    return canvas

# convert pyplot canvas to an RGB numpy array
def canvas2rgb_array(canvas):
    canvas.draw()
    buf = canvas.tostring_rgb()
    ncols, nrows = canvas.get_width_height()
    return np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

# merge two images horizontally to the height of the second image
def merge_images(image1, image2):
    # if images have different heights
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

# create aruco dict and params
aruco_dict, parameters = customAruco()

# load 3D object
vertices, triangles = read_obj(args.object)

# load video file
cap = cv2.VideoCapture(args.inputVideo)

# open video file for writing
if args.outputVideo is not None:
    # read first frame and merge to get final frame size
    _ , frame = cap.read()
    points = (vertices[:,0], vertices[:,1], vertices[:,2])
    canvas = plot3d(points, triangles)
    graph_image = canvas2rgb_array(canvas)
    finalFrameSize = merge_images(frame, graph_image).shape
    videoOut = cv2.VideoWriter(args.outputVideo, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(5)), (int(finalFrameSize[1]), int(finalFrameSize[0])))

# load reference image
refImage = cv2.cvtColor(cv2.imread(args.referenceImage), cv2.COLOR_BGR2GRAY)
# detect markers in reference image
refCorners, refIds, refRejected = cv2.aruco.detectMarkers(refImage, aruco_dict, parameters = parameters)
# add a distance estimate to the markers in the reference image (Z axis from camera) - here we set it to 50
refCorners = [np.append(theseCorners, np.add(np.zeros((1,4,1)), 50), axis = 2) for theseCorners in refCorners]
# camera matrix estimate
focal_length = cap.get(3)
center = (cap.get(3)/2, cap.get(4)/2)
camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]], dtype = "double"
                )
dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

while(True):
    # read new frame from VideoCapture
    ret, frame = cap.read()
    if frame is not None:
        # convert frame to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect aruco markers in the gray frame
        res_corners, res_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters = parameters)
        if res_ids is not None:
            # get indices of detected markers from the reference image markers detected before
            idx = which(refIds, res_ids)
            # if the markers detected in the frame exist in the reference image
            if len(idx) > 0:
                # flatten all detected markers' corners in frame and reference image
                these_res_corners = np.concatenate(res_corners, axis = 1)
                these_ref_corners = np.concatenate([refCorners[x] for x in idx], axis = 1)
                # estimate rotation and translation vectors with solvePnP
                success, rotation_vector, translation_vector = cv2.solvePnP(these_ref_corners, these_res_corners, camera_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)
                # convert rotation vector to matrix
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                # use rotation matrix to modify all 3d object's vertices
                new_points = np.array([np.matmul(rotation_matrix, np.array([p[0], -p[1], p[2]])) for p in vertices])
                # reshape axes of modified vertices
                points = (new_points[:,0], new_points[:,1], new_points[:,2])
                # plot vertices with their triangles (to fill the surface)
                canvas = plot3d(points, triangles)
                # convert the plot as an array
                graph_image = canvas2rgb_array(canvas)
        # merge color frame and plot image
        frame = merge_images(frame, graph_image)
        if args.outputVideo is not None:
            videoOut.write(frame)
        # Display the resulting frame
        cv2.imshow('frame',frame)
    # if q is pressed, quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and output video file
if args.outputVideo is not None:
    videoOut.release()
cap.release()
# close cv2 window
cv2.destroyAllWindows()