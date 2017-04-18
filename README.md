# Receipt Classifier in Python  
- - - - 
 
This application recognizes whether a receipt belongs to a specific store (Walmart). It pre-processes image using OpenCV python library cv2, extracts the text from processed image and then uses SVC (Support Vector Classifier) to classify test receipts. 
 
- - - - 
 
## Usage ## 
 
1. **Configuration:** Paths of input image, pre-processed image, extracted text files, traning and test csvs are configured in config.yml 
2. **Execution:** Execute main.py to run the application.
 
- - - - 
 
## Image Pre-Processesing ## 
 
Image pre-processing is being done to improve receipt image quality before extracting text using tesseract. In this process, blocks of texts are identified and cropped to generated processed image. The approach includes following steps.  

1. **Rescaling:** Input images give are of larges sizes. They are rescaled to smaller dimension in order to speed up image pre-processing.
2. **Binarization:** Images are converted to Grayscale and then binarized using adaptive Threshold. 
3. **Denoising and Morphological Transformation:** Morphological Opening is done to remove noises and then Closing is done in order to close small holes in texts in the binary image. 
4. **Finding Countours of Text Blocks:** Morphological Gradient followed by Closing is performed to find contours of text blocks. 
5. **Generating Mask from Text Block Contours:** Mask from contours of text blocks is generated, which is used for cropping required text blocks.
6. **Cropping text blocks from binary image using mask:** The output is being saved as processed image which is to be parsed by tesseract. 
 
- - - - 
 
## Extracting Text from Pre-Processesed Images ## 

pytesseract, Python wrapper over Tesseract OCR utility is being used to extract text from images using following command. 
    extracted_str = image_to_string(Image.open(ImgProcPath), lang="eng", config="-psm 1") 
 
- - - - 
 
## Building Classification Model ## 