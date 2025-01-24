#!/usr/bin/env python3
import cv2
import numpy as np
import math
import sys

def detect_edges(image):
  """Find edge points in a grayscale image.

  Args:
  - image (2D uint8 array): A grayscale image.

  Return:
  - edge_image (2D float array): A heat map where the intensity at each point
      is proportional to the edge magnitude.
  """
  sobel_3x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
  sobel_3y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
  image = image.astype(float)
  edge_image_x = np.zeros_like(image)
  edge_image_y = np.zeros_like(image)
  edge_image = np.zeros_like(image)
  image = np.pad(image, 1)
  for i in range(edge_image_x.shape[0]):
    for j in range(edge_image_x.shape[1]):
      edge_image_x[i, j] = np.sum(image[i: i+3, j: j+3] * sobel_3x)
      edge_image_y[i, j] = np.sum(image[i: i+3, j: j+3] * sobel_3y)
      edge_image[i, j] = math.sqrt(math.pow(edge_image_x[i, j], 2) + math.pow(edge_image_y[i, j], 2))
  edge_image = 255 * edge_image / np.max(edge_image)
  return edge_image
  
      

def hough_circles(edge_image, edge_thresh, radius_values):
  """Threshold edge image and calculate the Hough transform accumulator array.

  Args:
  - edge_image (2D float array): An H x W heat map where the intensity at each
      point is proportional to the edge magnitude.
  - edge_thresh (float): A threshold on the edge magnitude values.
  - radius_values (1D int array): An array of R possible radius values.

  Return:
  - thresh_edge_image (2D bool array): Thresholded edge image indicating
      whether each pixel is an edge point or not.
  - accum_array (3D int array): Hough transform accumulator array. Should have
      shape R x H x W.
  """
  thresh_edge_image = np.zeros_like(edge_image)
  accum_array = np.zeros((len(radius_values), edge_image.shape[0], edge_image.shape[1]))
  for i in range(thresh_edge_image.shape[0]):
    for j in range(thresh_edge_image.shape[1]):
      if edge_image[i, j] >= edge_thresh:
        thresh_edge_image[i, j] = 1
      else:
        thresh_edge_image[i, j] = 0
  thresh_edge_image = thresh_edge_image.astype(bool)
  for i in range(len(radius_values)):
    for j in range(edge_image.shape[0]):
      for k in range(edge_image.shape[1]):
        if thresh_edge_image[j, k] == 1:
          theta = np.linspace(0, 2 * math.pi, 180)
          circles_x = j + np.cos(theta) * radius_values[i]
          circles_y = k + np.sin(theta) * radius_values[i]
          circles_x = np.round(circles_x).astype(int)
          circles_y = np.round(circles_y).astype(int)
          valid_xy = np.argwhere((circles_x >= 0) & (circles_x < edge_image.shape[0]) & (circles_y >= 0) & (circles_y < edge_image.shape[1]))
          accum_array[i, circles_x[valid_xy], circles_y[valid_xy]] += 1
  show_accum_array = accum_array[8, :, :]
  show_accum_array = 255 * show_accum_array / np.max(show_accum_array)
  cv2.imwrite('output/' + 'coins' + '_edge_accum_array_28.png', show_accum_array)
  return thresh_edge_image, accum_array

def find_circles(image, accum_array, radius_values, hough_thresh):
  """Find circles in an image using output from Hough transform.

  Args:
  - image (3D uint8 array): An H x W x 3 BGR color image. Here we use the
      original color image instead of its grayscale version so the circles
      can be drawn in color.
  - accum_array (3D int array): Hough transform accumulator array having shape
      R x H x W.
  - radius_values (1D int array): An array of R radius values.
  - hough_thresh (int): A threshold of votes in the accumulator array.

  Return:
  - circles (list of 3-tuples): A list of circle parameters. Each element
      (r, y, x) represents the radius and the center coordinates of a circle
      found by the program.
  - circle_image (3D uint8 array): A copy of the original image with detected
      circles drawn in color.
  """
  
  circles = []
  for i in range(len(radius_values)):
      for j in range(image.shape[0]):
          for k in range(image.shape[1]):
              if accum_array[i, j, k] >= hough_thresh:
                  rmin, rmax = max(0, i - 3), min(len(radius_values), i + 3)
                  xmin, xmax = max(0, j - 3), min(image.shape[0], j + 3)
                  ymin, ymax = max(0, k - 3), min(image.shape[1], k + 3)
                  if np.max(accum_array[rmin:rmax, xmin:xmax, ymin:ymax]) == accum_array[i, j, k]:
                      circles.append((radius_values[i], j, k))
  for point in circles:                 
    cv2.circle(image, (point[2], point[1]), point[0], (0, 255, 0), 2)
  return circles, image

def main(argv):
  img_name = argv[0]
  edge_thresh = float(argv[1])
  hough_thresh = int(argv[2])
  img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  edge_image = detect_edges(gray_image)
  radius_value = np.arange(20, 41)
  thresh_edge_image, accum_array = hough_circles(edge_image, edge_thresh, radius_value)
  circles, circle_image = find_circles(img, accum_array, radius_value, hough_thresh)
  
  cv2.imwrite('output/' + img_name + '_edge_sobel.png', edge_image)
  cv2.imwrite('output/' + img_name + '_thresh_edge_image.png', thresh_edge_image * 255)
  cv2.imwrite('output/' + img_name + '_circle_image.png', circle_image)
  print(circles)
  
if __name__ == '__main__':
  main(sys.argv[1:])