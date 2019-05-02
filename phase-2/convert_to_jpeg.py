#!/usr/bin/env python
import cv2 
import os

IMAGES_DIR = 'Images/001/'
root, dirs, images = os.walk(IMAGES_DIR).next()
counter = 1

for j in images:
  img = cv2.imread(IMAGES_DIR + j)
  cv2.imwrite(IMAGES_DIR + str(counter) + '.JPEG', img)
  os.remove(IMAGES_DIR + j)
  counter = counter + 1
