#!/usr/bin/env python3
# chmod u+x
# -*- coding: utf-8 -*-  

import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
from math import ceil
import random
import imageio

def read_images(img_path):
  
  #image_path = '../data_set/Faces/s1/1.pgm'
  #img = misc.imread(image_path)#, flatten=True)
  try:
    img = np.array(imageio.imread(img_path), dtype=np.uint8)
  except:
    print("Img " + str(img_path) + " do not exist")
    sys.exit(1)

  return img

def display_img(img):
  # Plot the input
  plt.subplot(223)
  plt.imshow(img, cmap=plt.cm.gray)
  plt.axis('off')
  plt.show()

def save_img(img, path_to='../data_set/out.pgm'):
  imageio.imwrite(path_to, img[:, :])
def get_width_height(img):
  return [len(img[0]), len(img)]

def convolutional(img, width,height, filter):
  output_img_ReLu = np.zeros((height,width))
  #output_img2 = np.zeros((height,width))

  index_height = -1
  for line in img:
    index_width = 1
    index_height +=1
    for index in range(len(line)-2):
      # calculate the convolutional, from filter, using ReLu
      output_img_ReLu[index_height][index_width] = max(0,( (line[index] * filter[0]) + (line[index +1] * filter[1]) ))
      #output_img2[index_height][index_width] = abs(( (line[index] * filter[0]) + (line[index +1] * filter[1]) ))
      index_width+=1

  return output_img_ReLu

def main():
  
  img_path = '../data_set/Faces/s1/1.pgm'
  if len(sys.argv) >= 2:
    img_path = sys.argv[1] 

  img = read_images(img_path)

  display_img(img)

  width,height = get_width_height(img)

  output_img_ReLu = convolutional(img,width,height,[-1,1])

  display_img(output_img_ReLu)

  #save_img(output_img_ReLu)


if __name__ == '__main__':
  main()


