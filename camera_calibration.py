# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 21:38:50 2019

@author: Kenkun Liu
"""

import cv2
import numpy as np
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
# the size of the chessboard 10x7
w = 9   # horizontal corner number
h = 6   # vertical corner number
# spatial coordinates of every corner
objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
objp = objp*20  # the length of every square is 20 mm

objpoints = [] # spatial coordinates
imgpoints = [] # image coordinates

images = glob.glob('./chessboard/*.jpg')  # path of the images for calibration

i = 1
for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # RGB to GRAY transform
    ret, corners = cv2.findChessboardCorners(gray, (w,h),None)  # find the corners
    # if all corners are detected,go into the next step
    if ret == True:
        i = i+1
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        # show the corners in the image
        cv2.drawChessboardCorners(img, (w,h), corners, ret)
        cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('findCorners', 810, 405)
        cv2.imshow('findCorners',img)
        cv2.waitKey(1)
cv2.destroyAllWindows()
# calibrating
_, mtx, dist, _, _= \
    cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print('Total number of images:', i)
print("mtx:\n",mtx)        # intrinsics 
print("dist:\n",dist   )   # distortion cofficients = (k_1,k_2,p_1,p_2,k_3)  