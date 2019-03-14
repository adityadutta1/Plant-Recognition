# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 13:37:15 2018

@author: dell
"""

import numpy as np
import cv2


cap=cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame=cap.read()
    if ret==True:
        #g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame=cv2.flip(frame,1)
    #writing the flipped frame
        
    #DISPLAYING THE RESULTING FRAME
        cv2.imshow('ImageCreators',frame)
        cv2.imwrite('C:/Users/Adity/Desktop/projectleaf/dataset/test/image12.jpg',frame)
        if cv2.waitKey(10)&0xFF==ord('q'):
            break
    
cap.release()
#out.release()
cv2.destroyAllWindows()
import ltest.py
ltest.py