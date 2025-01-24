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
        pass
    
    # TODO: Detect chessboard corners in each image
    def detect_corners(image, pattern_size):
        # Students should fill in code here to detect corners using cv2.findChessboardCorners or another method
        pass

    # TODO: Complete the loop below to obtain the corners of each image and the corresponding world coordinate points
    for image in images:
        corners = detect_corners(image, pattern_size)
        if corners is not None:
            # Add image corners
            image_points.append(corners)
            # Add the corresponding world points
            world_points.append(init_world_points(pattern_size))
    
    return world_points, image_points

def calibrate_camera(world_points, image_points):
    assert len(world_points) == len(image_points), "The number of world coordinates and image coordinates must match"
    
    num_points = len(world_points)
    A = []
    B = []
    K = np.zeros((4, 4))
    P = None

    # TODO main loop, use least squares to solve for P and then decompose P to get K and R
    # The steps are as follows:
    # 1. Construct the matrix A and B
    # 2. Solve for P using least squares
    # 3. Decompose P to get K and R
    for i in range(num_points):
        pass
    
    # Please ensure that the diagonal elements of K are positive
    
    return K, P

# Main process
image_path = 'Sample_Calibration_Images'
images = read_images(image_path)

# TODO: I'm too lazy to count the number of chessboard squares, count them yourself
pattern_size = (0, 0)  # The pattern size of the chessboard 

world_points, image_points = find_image_points(images, pattern_size)

camera_matrix, camera_extrinsics = calibrate_camera(world_points, image_points)

print("Camera Calibration Matrix:")
print(camera_matrix)

def test(image_directory, pattern_size):
    # In this function, you are allowed to use OpenCV to verify your results. This function is optional and will not be graded.
    # return None, directly print the results
    # TODO
    print("Camera Calibration Matrix by OpenCV:")
    print("Camera Matrix:\n", None)

def reprojection_error(world_points, image_points, camera_matrix):
    # In this function, you are allowed to use OpenCV to verify your results.
    # show the reprojection error of each image
    pass

print("Camera Calibration Matrix by OpenCV:")
test(image_path, pattern_size)
