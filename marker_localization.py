# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:26:35 2019

@author: Kenkun Liu
"""

import numpy as np
import cv2
import cv2.aruco as aruco
from expm import *

# camera intrinsics
mtx = np.array([[599.31904925, 0, 336.78467099],
                [  0, 601.83038136, 352.52431021],
                [  0, 0, 1],])
# distortion coefficients
dist = np.array( [ 0.00474415, 0.06832274, -0.01973907, -0.00862282, -0.31042975] )

img = cv2.imread('./marker_localization/im1.jpg') # image path
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # RGB to gray transformation
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)  
parameters =  aruco.DetectorParameters_create()  
     
#lists of ids and the corners beloning to each id  
corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)
  
#    if ids != None: 
if ids is not None:
    # Estimate pose of each marker and return the values rvet(rotation vector) and tvec(translation vector)
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.048, mtx, dist)  

    (rvec-tvec).any() # get rid of that nasty numpy value array error  
     
    ###### DRAW ID ##### 
    for i in range(rvec.shape[0]):
        aruco.drawAxis(img, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
        aruco.drawDetectedMarkers(img, corners, ids)
 
  

# Display the resulting frame  
cv2.imshow('markers',img)  

key = cv2.waitKey(0)

# push esc to exit
if key == 27:
    cv2.destroyAllWindows()

# set Marker #55 as frame A
Wa = rvec[ids == 55][0]
Pa = tvec[ids == 55][0]

# set Marker #10 as frame B
Wb = rvec[ids == 10][0]
Pb = tvec[ids == 10][0]

# calculate the configuration of frame B w.r.t frame A
gca = g_ab(expm(Wa),Pa) #gca:the configuration of frame A w.r.t camera frame
gcb = g_ab(expm(Wb),Pb) #gcb:the configuration of frame B w.r.t camera frame

gab = np.dot(np.linalg.inv(gca), gcb) #gab = gac*gcb, gac = gca^(-1).  gab is the configuration of the frame B w.r.t frame A

print(gab) 

