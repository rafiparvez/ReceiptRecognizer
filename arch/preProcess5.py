# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 09:33:09 2017

@author: rafip
"""

import cv2
def preProcess(ImgProcPath,imgPath):
    img = cv2.imread(imgPath)
    mser = cv2.MSER_create()
    
    #Resize the image so that MSER can work better
    img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = img.copy()
    
    regions = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
    cv2.polylines(vis, hulls, 1, (0,255,0))
    
    cv2.namedWindow('img', 0)
    cv2.imshow('img', vis)
    while(cv2.waitKey()!=ord('q')):
        continue
    cv2.destroyAllWindows()