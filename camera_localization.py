# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:26:35 2019

@author: Kenkun Liu
"""
# import packages required
import numpy as np
import cv2
import cv2.aruco as aruco
# import some functions defined by ourselves to calculate matrix exponentials
from expm import * 

# camera intrinsics
mtx = np.array([[599.31904925, 0, 336.78467099],
                [  0, 601.83038136, 352.52431021],
                [  0, 0, 1],])

# distortion coefficients
dist = np.array( [ 0.00474415, 0.06832274, -0.01973907, -0.00862282, -0.31042975] )

img = cv2.imread('./camera_localization/im4.jpg')   # the path of the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # convert the image into gray image
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250) # set the dict of markers
parameters =  aruco.DetectorParameters_create()

#lists of ids and the corners beloning to each id  
corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)

#    if ids != None: 
if ids is not None:
    # Estimate pose of each marker and return the values rvet(rotation vector) and tvec(translation vector)
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.048, mtx, dist)  #the length of each marker is 48mm 
  
    (rvec-tvec).any() # get rid of that nasty numpy value array error  
        
    ###### DRAW ID #####
    for i in range(rvec.shape[0]):
        aruco.drawAxis(img, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
        aruco.drawDetectedMarkers(img, corners, ids)

# Display the resulting frame  
cv2.imshow('markers',img)  

# push key esc to exit
key = cv2.waitKey(0)
if key == 27:    
    cv2.destroyAllWindows()
    

#---------------------------------#
# the actual configuration of the camera
actual_gsc = np.array([[1,0,0,0.074],[0,-0.5,0.866,-0.01],[0,-0.866,-0.5,0.15],[0,0,0,1]])   # configuration w.r.t frame S
actual_exp_coor = exp_coor(actual_gsc)

# set Marker #10 as frame A
gsa = np.array([[0,1,0,0],[-1,0,0,0.173],[0,0,1,0],[0,0,0,1]])    #configuration w.r.t frame S

# set Marker #99 as frame B
gsb = np.array([[1,0,0,0.141],[0,1,0,0.197],[0,0,1,0.007],[0,0,0,1]]) #configuration w.r.t frame S

# set Marker #55 as frame D
gsd = np.array([[-1,0,0,0],[0,-1,0,0.303],[0,0,1,0],[0,0,0,1]])   #configuration w.r.t frame S
#---------------------------------#

# configuration of frame A w.r.t the camera
Wa = rvec[ids == 10][0]
Pa = tvec[ids == 10][0]
gca = g_ab(expm(Wa),Pa)

# configuration of frame B w.r.t the camera
Wb = rvec[ids == 99][0]
Pb = tvec[ids == 99][0]
gcb = g_ab(expm(Wb),Pb)

# configuration of frame D w.r.t the camera
Wd = rvec[ids == 55][0]
Pd = tvec[ids == 55][0]
gcd = g_ab(expm(Wd),Pd)

# estimate the configuration of the camera by only one marker 10 (frame A)
gsc_a = np.dot(gsa,np.linalg.inv(gca))   # estimated camera configuration
exp_coor_sc_a = exp_coor(gsc_a) # estimated exp. coor. of camera

# estimate the configuration of the camera by only one marker 99 (frame B)
gsc_b = np.dot(gsb,np.linalg.inv(gcb))   # estimated camera configuration
exp_coor_sc_b = exp_coor(gsc_b) # estimated exp. coor. of camera

# estimate the configuration of the camera by only one marker 55 (frame D)
gsc_d = np.dot(gsd,np.linalg.inv(gcd))   # estimated camera configuration
exp_coor_sc_d = exp_coor(gsc_d) # estimated exp. coor. of camera

#---accuracy = norm(actual-estimated)/norm(actual)---#
accuracy_oneMarker = 1 - np.linalg.norm(actual_exp_coor-exp_coor_sc_b)/np.linalg.norm(actual_exp_coor)  
print('one marker:\n\n actual:\n',actual_exp_coor)
print('estimated:\n',exp_coor_sc_b)
print('accuracy: ',accuracy_oneMarker) 

avg_expcoor2 = (exp_coor_sc_a+exp_coor_sc_b)/2
accuracy_twoMarker = 1 - np.linalg.norm(actual_exp_coor-avg_expcoor2)/np.linalg.norm(actual_exp_coor)
print('two marker:\n\n actual:\n',actual_exp_coor)
print('estimated:\n',avg_expcoor2)
print('accuracy: ',accuracy_twoMarker) 

avg_expcoor3 = (exp_coor_sc_a+exp_coor_sc_b+exp_coor_sc_d)/3
accuracy_threeMarker = 1 - np.linalg.norm(actual_exp_coor-avg_expcoor3)/np.linalg.norm(actual_exp_coor)
print('three marker:\n\n actual:\n',actual_exp_coor)
print('estimated:\n',avg_expcoor3)
print('accuracy: ',accuracy_threeMarker) 
#---------------------------------------------------#
