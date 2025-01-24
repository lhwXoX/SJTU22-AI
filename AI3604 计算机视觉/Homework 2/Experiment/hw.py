import cv2
import numpy as np
import glob

def read_images(image_directory):
    # Read all jpg images from the specified directory
    return [cv2.imread(image_path) for image_path in glob.glob(f"{image_directory}/*.jpg")]

def find_image_points(images, pattern_size):
    world_points = []
    image_points = []
    
    # TODO: Initialize the chessboard world coordinate points
    def init_world_points(pattern_size):
        # Students should fill in code here to generate the world coordinates of the chessboard
        world_points = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2).astype(np.float32)
        return world_points
    
    # TODO: Detect chessboard corners in each image
    def detect_corners(image, pattern_size):
        # Students should fill in code here to detect corners using cv2.findChessboardCorners or another method
        found, corners = cv2.findChessboardCorners(image, pattern_size, None)
        if found:
            return corners.reshape(-1, 2)
        else:
            return None

    # TODO: Complete the loop below to obtain the corners of each image and the corresponding world coordinate points
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #To grayscale or there's not enough images(RGB:2, Gray:4)
        corners = detect_corners(image, pattern_size)
        if corners is not None:
            # Add image corners
            image_points.append(corners)
            # Add the corresponding world points
            world_points.append(init_world_points(pattern_size))
    
    return world_points, image_points

def calibrate_camera(world_points, image_points):
    assert len(world_points) == len(image_points), "The number of world coordinates and image coordinates must match"
    num_points = len(world_points) #view?
    A = []
    B = []
    K = np.zeros((4, 4))
    P = []

    # TODO main loop, use least squares to solve for P and then decompose P to get K and R
    # The steps are as follows:
    # 1. Construct the matrix A and B
    # 2. Solve for P using least squares
    # 3. Decompose P to get K and R
    C = np.zeros((2 * num_points, 6), dtype=np.float32)
    for i in range(num_points):
        world_point = world_points[i]
        image_point = image_points[i]
        #choose 300 points in the central region
        image_points_arr = np.array(image_points).astype(np.float32)
        center_x = np.mean(image_points_arr[:, 0])
        center_y = np.mean(image_points_arr[:, 1])
        center = np.array([center_x, center_y], dtype=np.float32)
        idx = np.argsort(np.linalg.norm(image_point - center, axis=1))[:300]
        world_point, image_point = world_point[idx], image_point[idx]
        
        #matrix A in chessboard
        num = world_point.shape[0]
        A = np.zeros((2 * num, 9), dtype=np.float32)
        for j in range(num):
            A[2 * j] = np.array([world_point[j, 0], world_point[j, 1], 1, 0, 0, 0, -image_point[j, 0] * world_point[j, 0], -image_point[j, 0] * world_point[j, 1], -image_point[j, 0]])
            A[2 * j + 1] = np.array([0, 0, 0, world_point[j, 0], world_point[j, 1], 1, -image_point[j, 1] * world_point[j, 0], -image_point[j, 1] * world_point[j, 1], -image_point[j, 1]])
        #H in chessboard
        value, vector = np.linalg.eig(A.T @ A)
        H = vector[:, np.argmin(value)].reshape(3, 3)
        P.append(H) #H for this view
        h11, h12, h13, h21, h22, h23, h31, h32, h33 = H.reshape(-1)
        #Solve B
        C[2 * i] = np.array([h11 * h11 - h12 * h12, 2 * (h11 * h21 - h12 * h22), 2 * (h11 * h31 - h12 * h32), h21 * h21 - h22 * h22, 2 * (h21 * h31 - h22 * h32), h31 * h31 - h32 * h32])
        C[2 * i + 1] = np.array([h11 * h12, h11 * h22 + h12 * h21, h11 * h32 + h12 * h31, h21 * h22, h21 * h32 + h22 * h31, h31 * h32])
    value, vector = np.linalg.eig(C.T @ C)
    b = vector[:, np.argmin(value)]
    B = np.array([[b[0], b[1], b[2]], [b[1], b[3], b[4]], [b[2], b[4], b[5]]]) #symmetic B
    K = np.linalg.cholesky(B)
    K = np.linalg.inv(K.T)
    K = K / K[2, 2] #1 in (3, 3)
    # Please ensure that the diagonal elements of K are positive
    return K, P

# Main process
image_path = 'E:/Homework/CV_HW2/Sample_Calibration_Images'
images = read_images(image_path)

# TODO: I'm too lazy to count the number of chessboard squares, count them yourself
pattern_size = (31, 23)  # The pattern size of the chessboard 

world_points, image_points = find_image_points(images, pattern_size)

camera_matrix, camera_extrinsics = calibrate_camera(world_points, image_points)

print("Camera Calibration Matrix:")
print(camera_matrix)

def test(image_directory, pattern_size):
    # In this function, you are allowed to use OpenCV to verify your results. This function is optional and will not be graded.
    # return None, directly print the results
    images = read_images(image_directory)
    pattern_size = (31, 23)
    world_points, image_points = find_image_points(images, pattern_size)
    #expand
    expand = np.zeros((world_points[0].shape[0], 1), dtype=np.float32)
    expand_world_points = [np.append(view, expand, axis=1) for view in world_points]
    _, camera_matrix, _, _, _ = cv2.calibrateCamera(expand_world_points, image_points, images[0].shape[:2][::-1], None, None)
    
    print("Camera Calibration Matrix by OpenCV:")
    print("Camera Matrix:\n", camera_matrix)

def reprojection_error(world_points, image_points, camera_matrix):
    # In this function, you are allowed to use OpenCV to verify your results.
    # show the reprojection error of each image
    error_list = []

    for i in range(len(world_points)):
        #expand 1
        expand = np.ones((world_points[i].shape[0], 1), dtype=np.float32)
        expand_world_points = np.append(world_points[i], expand, axis=1)
        #Projection
        projected_points = camera_matrix[i] @ expand_world_points.T
        projected_points /= projected_points[2]
        #error
        norm_error = np.linalg.norm(image_points[i].T - projected_points[:2], axis=0)
        error_list.append(np.mean(norm_error))
    print("Reprojection error:", error_list)

test(image_path, pattern_size)
reprojection_error(world_points, image_points, camera_extrinsics) #input matrix H to reproject points