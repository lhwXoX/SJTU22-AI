#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import math

def binarize(gray_image, thresh_val):
  binary_image = np.zeros((gray_image.shape[0], gray_image.shape[1]))
  for x in range(gray_image.shape[0]):
    for y in range(gray_image.shape[1]):
      if gray_image[x, y] >= thresh_val:
        binary_image[x, y] = 255
  return binary_image

class equivalence_table:
  def __init__(self, num):
    self.trace = np.arange(num)
  def traceroot(self, lable):
    lable = int(lable)
    while(lable != self.trace[lable]):
      lable = self.trace[lable]
    return lable
  def addequivalence(self, a, b):
    a, b = int(a), int(b)
    a = self.traceroot(a)
    b = self.traceroot(b)
    self.trace[a] = b
    
def lable(binary_image):
  labled_image = np.zeros((binary_image.shape[0], binary_image.shape[1]))
  num_lable = 1
  lookuptable = equivalence_table(binary_image.shape[0] * binary_image.shape[1])
  for x in range(labled_image.shape[0]):
    for y in range(labled_image.shape[1]):
      if binary_image[x, y] > 0:
        if labled_image[x-1, y-1] > 0:
          labled_image[x, y] = labled_image[x-1, y-1]
        elif labled_image[x, y-1] > 0 and labled_image[x-1, y] == 0:
          labled_image[x, y] = labled_image[x, y-1]
        elif labled_image[x-1, y] > 0 and labled_image[x, y-1] == 0:
          labled_image[x, y] = labled_image[x-1, y]
        elif labled_image[x-1, y] > 0 and labled_image[x, y-1] > 0:
          labled_image[x, y] = labled_image[x-1, y]
          lookuptable.addequivalence(labled_image[x-1, y], labled_image[x, y-1])
        else:
          labled_image[x, y] = num_lable
          num_lable += 1

  for x in range(labled_image.shape[0]):
    for y in range(labled_image.shape[1]):
      if labled_image[x, y] > 0:
        labled_image[x, y] = lookuptable.traceroot(labled_image[x, y])
  num_lable = np.unique(labled_image)
  for i in range(1, len(num_lable)):
    labled_image[labled_image == num_lable[i]] = 255 * (i / len(num_lable))
  return labled_image

def get_attribute(labled_image):
  attribute_list = []
  num_lable = np.unique(labled_image)
  for i in range(1, len(num_lable)):
    position_list = np.argwhere(labled_image == num_lable[i])
    position_list[:, 0] = labled_image.shape[0] - position_list[:, 0] - 1
    x_sum, y_sum = 0, 0
    a_sum, b_sum, c_sum = 0, 0, 0
    for point in position_list:
      x_sum += point[1]
      y_sum += point[0]
    x_mean = x_sum / len(position_list)
    y_mean = y_sum / len(position_list)
    for point in position_list:
      a_sum += (point[1] - x_mean) * (point[1] - x_mean)
      b_sum += 2 * (point[1] - x_mean) * (point[0] - y_mean)
      c_sum += (point[0] - y_mean) * (point[0] - y_mean)
    theta_1 = math.atan2(b_sum, a_sum - c_sum) / 2
    theta_2 = theta_1 + math.pi / 2
    E_min = a_sum * math.sin(theta_1) * math.sin(theta_1) - b_sum * math.sin(theta_1) * math.cos(theta_1) + c_sum * math.cos(theta_1) * math.cos(theta_1)
    E_max = a_sum * math.sin(theta_2) * math.sin(theta_2) - b_sum * math.sin(theta_2) * math.cos(theta_2) + c_sum * math.cos(theta_2) * math.cos(theta_2)
    roundness = E_min / E_max
    attribute_list.append({'position': {'x': x_mean, 'y': y_mean}, 'orientation': theta_1, 'roundness': roundness})
  return attribute_list

def main(argv):
  img_name = argv[0]
  thresh_val = int(argv[1])
  img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  binary_image = binarize(gray_image, thresh_val=thresh_val)
  labled_image = lable(binary_image)
  attribute_list = get_attribute(labled_image)

  cv2.imwrite('output/' + img_name + "_gray.png", gray_image)
  cv2.imwrite('output/' + img_name + "_binary.png", binary_image)
  cv2.imwrite('output/' + img_name + "_labled.png", labled_image)
  print(attribute_list)


if __name__ == '__main__':
  main(sys.argv[1:])
