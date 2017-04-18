# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 08:39:07 2017

@author: rafip
"""

import pandas as pd
import numpy as np
import yaml
from os.path import isfile, join
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report as clsr
from sklearn.cross_validation import train_test_split as tts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC

def buildnEvaluateModel(X, y):
    '''
    The function takes training data and splits it further into
    Training and Cross-validate sets. And returns the model.
    '''
    # Split the traning data input to get 20% cross-validation data set
    # for model evaluation
    X_train, X_cv, y_train, y_cv = tts(X, y, test_size=0.2)
    
    #convert dataframe with float valaues into bool
    y_train = [bool(int(i)) for i in y_train]
    y_cv = [bool(int(i)) for i in y_cv]
    
    #output classification labels
    labels = LabelEncoder()
    labels.fit_transform(y_train)

    # define classification model
    text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SVC(kernel='linear',probability=True)),])
    
    #Traning the model
    text_clf = text_clf.fit(X_train, y_train)
    
    '''
    Following section evaluates the model performance
    '''
    predicted = text_clf.predict(X_cv) 
    print("Model Accuracy = "+str(np.mean(predicted == y_cv)))
    print(clsr(y_cv, predicted, target_names=[str(i) for i in labels.classes_]))
    
    return text_clf


def classifyReceipt():
    config = yaml.safe_load(open("config.yml"))
    #path of traning csv
    train_file = config['train_csv']
    test_file = config['test_csv']
    
    #path of text files
    img_text_dir = config['text_path']
    
    df_trainFiles = pd.read_csv(train_file)
    
    #Training dataframe
    df_train = pd.DataFrame(columns = ['fileID', 'fileData' , 'isWalmart'])
    for index, row in df_trainFiles.iterrows():
        fileID =  row['EXT_ID']
        isWalmart =  row['IsWalmart']
        text_ip = join(img_text_dir,fileID+'.txt')      
        #Process text file and save in fileData column of traning dataframe
        if isfile(text_ip):
            with open(text_ip, 'r', encoding='utf-8') as myfile:
                extracted_str=myfile.read().replace('\n', ' ')
            df_train.loc[len(df_train.index)] = [fileID, extracted_str, isWalmart]
    X_train = df_train['fileData']
    y_train = df_train['isWalmart']
    
    df_testFiles = pd.read_csv(test_file)   
    #Test Dataframe   
    df_test = pd.DataFrame(columns = ['EXT_ID', 'WalmartReceipt' , 'fileData', 'PredictionScore'])
    for index, row in df_testFiles.iterrows():
        fileID =  row['EXT_ID']
        text_ip = join(img_text_dir,fileID+'.txt')    
        #Process text file for test images and save in fileData column of test dataframe
        if isfile(text_ip):
            with open(text_ip, 'r', encoding='utf-8') as myfile:
                extracted_str=myfile.read().replace('\n', ' ')
            df_test.loc[len(df_test.index)] = [fileID, '' , extracted_str,'']
     
    #get classification model
    model = buildnEvaluateModel(X_train, y_train)
    
    X_test = df_train['fileData']    
    Y_test = model.predict(X_test)
    
    #Calculating class probablity for expected output = 1.0
    predict_score = model.predict_proba(X_test)     
    df_test['PredictionScore'] = pd.Series(predict_score[:,1])  
    df_test['WalmartReceipt'] = pd.Series(Y_test)
    
    #Saving results in output.csv
    header = ["EXT_ID", "WalmartReceipt", "PredictionScore"]
    df_test.to_csv('output.csv', columns = header)
    
    print(df_test.head(n=15))

if __name__ == '__main__':
    classifyReceipt()