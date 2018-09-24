import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
from math import ceil
import random
import imageio

def read_images():
  
  image_path = '../data_set/Faces/s10/1.pgm'
  #img = misc.imread(image_path)#, flatten=True)
  img = np.array(imageio.imread(image_path), dtype=np.uint8)

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

def pad_img(img):
  width,height = get_width_height(img)

  
def convolutional(img, width,height, filter):
  output_img = np.zeros((height,width))

  index_height = -1
  for line in img:
    index_width = 1
    index_height +=1
    for index in range(len(line)-2):
      # calculate the convolutional, from filter, using ReLu
      output_img[index_height][index_width] = max(0,( (line[index] * filter[0]) + (line[index +1] * filter[1]) ))
      index_width+=1

  return output_img

def main():
  img = read_images()

  display_img(img)

  padded_img = pad_img(img)
  width,height = get_width_height(padded_img)

  output_img = convolutional(padded_img,width,height,[[-1,-1,-1], [-1,4,-1], [-1,-1,-1]])

  display_img(output_img)

  #save_img(img)


if __name__ == '__main__':
  main()


