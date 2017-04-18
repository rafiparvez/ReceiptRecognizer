# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 22:13:05 2017

@author: rafip
"""

import yaml
from os import listdir
from os.path import isfile, join
from pytesseract import image_to_string
from PIL import Image
from preProcess import preProcess
from classifyReceipt import classifyReceipt

def main():
    #read the config file
    config = yaml.safe_load(open("config.yml"))
    
    #fetch image filenames from image directory
    img_dir = config['img_path']
    image_files = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
    image_files = [f for f in image_files if not f.startswith('.')]
    
    #Directory to store processed images
    img_processed_dir = config['img_processed_path']
    
    #Directory to store processed text files
    img_text_dir = config['text_path']
	
    for imgFile in image_files:
        imgPath = join(img_dir, imgFile)
        ImgProcPath = join(img_processed_dir, imgFile)
        
		#text output file path
        textOP = join(img_text_dir,imgFile.split('.')[0]+'.txt')
        if not isfile(textOP):
            print('Begin Processing ' + imgFile)
			
			#Image pre-processing
            preProcess(ImgProcPath,imgPath)
			
			# Extract text from pre-processed images
            extracted_str = image_to_string(Image.open(ImgProcPath), lang="eng", config="-psm 1")
            extracted_str = extracted_str.lower()
                      
            
            print("OUTPUT:"+textOP)
            with open(textOP, 'w' , encoding='utf-8') as text_file:
                text_file.write(extracted_str)
        else:
            print(" Already exists:"+textOP)
        #print(extracted_str)
    
    #Call method to classify receipts
    classifyReceipt()

if __name__ == '__main__':
    main()