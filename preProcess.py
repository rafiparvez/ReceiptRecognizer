# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 20:40:40 2017

@author: rafip
"""
import cv2
import numpy as np
from cv2 import boundingRect, countNonZero, cvtColor, drawContours, findContours, getStructuringElement, morphologyEx, rectangle, threshold

def resize(image, width=None, height=None):
    '''    
    returns image resized acoording to given width or height

    '''
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def imgShow(caption,image):
    '''    
    helps display images for troubleshooting
	displays images in window until some key is pressed
    '''
    cv2.imshow(caption, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def preProcess(ImgProcPath,imgPath):
    image = cv2.imread(imgPath)
    #origImg = image.copy()
    
    #ratio = image.shape[0] / 2000.0
	
	#Resize original images to reduce size
    rgb = resize(image, height=2000)
    
	#Convert into grayscale
    gray = cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    
	# Binarize image using adaptive Threshold and perform Morphological Opening and Closing 
	# to correct missing pixels from texts. Closed Image will be used to crop text blocks from
    filtered = cv2.adaptiveThreshold(gray.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 29, 8)
    kernel = np.ones((1,1), np.uint8) 
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
    
    # morphological gradient
    morph_kernel = getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = morphologyEx(gray, cv2.MORPH_GRADIENT, morph_kernel)
	
    # binarize image. This binary image is used for detecting text blocks
    _, bw = threshold(src=grad, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    morph_kernel = getStructuringElement(cv2.MORPH_RECT, (9, 1))
	
    # connect horizontally oriented regions
    connected = morphologyEx(bw, cv2.MORPH_CLOSE, morph_kernel)
    mask = np.zeros(bw.shape, np.uint8)
	
    # find contours
    im2, contours, hierarchy = findContours(connected, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	
	#mask to crop text blocks from binary image "clo
    mask2 = np.zeros(gray.shape, np.uint8)
	
    # filter contours
    for idx in range(0, len(hierarchy[0])):
        x, y, w, h = boundingRect(contours[idx])
        # fill the contour
        mask = drawContours(mask, contours, idx, (255, 255, 255), cv2.FILLED)
		
        # ratio of non-zero pixels in the filled region.
        # Condition are put to exclude very small contours containing noise
        # and non-text containg contours
        r = float(countNonZero(mask)) / (w * h)
        if r > 0.45 and h > 8 and w > 8:
            rgb = rectangle(rgb, (x, y+h), (x+w, y), (0,255,0),3)
			
			#Applying mask2 to extract binary text blocks
            mask2 = rectangle(mask2, (x, y+h), (x+w, y), (255, 255, 255),-1)
            bwgray = cv2.bitwise_not(closing)   
            imgCropped = cv2.bitwise_and(bwgray,bwgray,mask = mask2)
            imgCropped = cv2.bitwise_not(imgCropped)
 
           
    #imgShow('asa',resize(imgCropped, height=700))   
    #print(ImgProcPath)
	
	#Save the cropped image as processed image
    cv2.imwrite(ImgProcPath, imgCropped)
    