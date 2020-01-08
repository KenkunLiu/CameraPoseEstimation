# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 12:34:51 2019

@author: Kenkun Liu
"""

import numpy as np
import cv2
import cv2.aruco as aruco
 
# set the marker dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
img = np.random.random((200,200))
img=aruco.drawMarker(aruco_dict, 55, 200, img, 1)     #In total,we created 4 markers numbered as 10,23,55,99
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()