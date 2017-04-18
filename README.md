# ReceiptRecognizer 
- - - - 
 
This application recognizes whether a receipt belongs to a specific store (Walmart). It pre-processes image using OpenCV python library cv2, extracts the text from processed image and then uses SVC (Support Vector Classifier) to classify test receipts. 
 
- - - - 
 
## Usage ## 
 
1. **Configuration:** Paths of input image, pre-processed image, extracted text files, traning and test csvs are configured in config.yml 
2. **Execution:** Execute main.py to run the application.
 
- - - - 
 
## Image Pre-Processesing ## 
 
Image pre-processing is being done to improve receipt image quality before extracting text using tesseract. 

