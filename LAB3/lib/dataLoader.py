import numpy as np
import sys
from string import punctuation



"""
This function is used to extract the data from the dataset to be able to use our designed model
args:
    dataset: The dataset that we want to split
returns:
    A tupple of 4 elements:
        1) The data for training
        2) The data for testing
        3) Training labels
        4) Testing labels
"""
def splitDataset(dataset):
    return ([dataset['train'][i]['text'] for i in range(25000)], [dataset['test'][i]['text'] for i in range(25000)], dataset['train']['label'], dataset['test']['label'])



"""

"""
def datasetToFasttextFile(dataset):
    (X_train_text, X_test_text, y_train, y_test) = splitDataset(dataset)
    labels = ["negative", "positive"]

    with open('texts/validation.txt', 'w') as f:
        for index in range(5000):
            f.write("__label__" + labels[y_train[index]] + " " + X_train_text[index] + "\n")
 
    with open('texts/train.txt', 'w') as f:
        for index in range(5000, len(X_train_text)):
            f.write("__label__" + labels[y_train[index]] + " " + X_train_text[index] + "\n")

    with open('texts/test.txt', 'w') as f:
        for index in range(len(X_test_text)):
            f.write("__label__" + labels[y_test[index]] + " " + X_test_text[index] + "\n")



"""
Remove ponctuation and lowercased the first str of a string
Args:
    textLst: The list with 1 str that we want to lowercase and without ponctuation
return:
    The textLst List with her first str lowercased and without ponctuation
"""
def removePonct(textLst):
    textLst[0] = textLst[0].translate(str.maketrans('', '', punctuation)).lower()
    return textLst



"""
Remove ponctuation and lowercased the element 'text' of a dict
Args:
    row: The dict that we want to lowercase and remove ponctuation of the element 'text'
return:
    The row dict with his 'text' element lowercased and without ponctuation
"""
def removePonctTransform(row):
    if 'text' in row :
        row['text'] = removePonct(row['text'])

    return row

