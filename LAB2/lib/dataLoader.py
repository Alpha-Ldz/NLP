from typing import List, Tuple

import nltk
import re
import numpy as np

from string import punctuation

from datasets import DatasetDict

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

stop = set(stopwords.words())
stop.add("br")

nltk.download('punkt')
re_word = re.compile(r"^\w+$")

"""
Remove ponctuation and lowercased the first str of a string
Args:
    textLst: The list with 1 str that we want to lowercase and without ponctuation
return:
    The textLst List with her first str lowercased and without ponctuation
"""
def removePonct(
    textLst: List[str]
) -> List[str]:
    textLst[0] = textLst[0].translate(str.maketrans('', '', punctuation)).lower()
    return textLst



"""
Remove ponctuation and lowercased the element 'text' of a dict
Args:
    row: The dict that we want to lowercase and remove ponctuation of the element 'text'
return:
    The row dict with his 'text' element lowercased and without ponctuation
"""
def removePonctTransform(
    row: dict
) -> dict:
    if 'text' in row :
        row['text'] = removePonct(row['text'])

    return row



"""
Remove ponctuation and lowercased the first str of a string
Args:
    textLst: The list with 1 str that we want to lowercase and without ponctuation
return:
    The textLst List with her first str lowercased and without ponctuation
"""
def removeStopWords(
    textLst: List[str]
) -> List[str]:
    text = textLst[0]

    stemmer = SnowballStemmer("english")
    stemmed = [stemmer.stem(word) for word in word_tokenize(text.lower()) if re_word.match(word) and not word in stop]
    newText = " ".join(stemmed)

    textLst[0] = newText
    return textLst



"""
Remove ponctuation and lowercased the element 'text' of a dict
Args:
    row: The dict that we want to lowercase and remove ponctuation of the element 'text'
return:
    The row dict with his 'text' element lowercased and without ponctuation
"""
def removeStopWordsTransform(
    row: dict
) -> dict:
    if 'text' in row :
        row['text'] = removeStopWords(row['text'])

    return row



"""
Return a dict of all words (limited to 10000 here to not crash your laptop memory :) ) of the dataset and their occurence sorted
args:
    data: The dataset that we want to count the number of occurence of each words
return:
    The dict with words occurences sorted
"""
def listSortRevelantWord(
    data : List[str]
) -> dict :
    words = dict()

    for row in data :
        for e in row.split(' ') :
            if not e in words:
                words[e] = 1
            else :
                words[e] += 1

    words = {k: v for k, v in sorted(words.items(), key=lambda item: item[1], reverse=True)}
    words = dict(zip(list(words.keys())[:10000], list(words.values())[:10000]))

    return dict(zip(words, range(len(words))))



"""
Return the same dict with the n first elements
args:
    words: The dict that we want to reduce
    n: The length wanted
return:
    The n first elements of the words dict
"""
def getNFirstWords(
    words: dict, 
    n: int
) -> dict:
    return dict(zip(list(words.keys())[:n], list(words.values())[:n]))



"""
Return a dict of the n more occurence words in the data dataset
args:
    data: The dataset that we want to count the number of occurence of each words
    n: The length wanted
return:
    The dict with the n firsts words occurences sorted
"""
def listMostRevelantWord(
    data: List[str], 
    n: int
) -> dict:
    words = dict()

    for row in data :
        for e in row.split(' ') :
            if not e in words:
                words[e] = 1
            else :
                words[e] += 1

    words = {k: v for k, v in sorted(words.items(), key=lambda item: item[1], reverse=True)}
    words = dict(zip(list(words.keys())[:n], list(words.values())[:n]))

    return dict(zip(words, range(n)))



"""
Encode the text from the data dataset on the words dict
args: 
    data: the dataset that we want to encode
    words: the words dict that we want to take reference to encode the texts
returns:
    the data dataset encoded
"""
def textToRevelantMatrice(
    data: List[str], 
    words: dict
) -> np.ndarray :
    N = len(data)
    n = len(words)

    rtn = np.zeros((N, n))

    for index in range(N):
        for word in data[index].split(' ') :
            if word in words:
                rtn[index][words[word]] += 1

    return rtn

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
def splitDataset(
    dataset: DatasetDict
) -> Tuple[List[str], List[str], List[int], List[int]]:
    return ([dataset['train'][i]['text'] for i in range(25000)], 
            [dataset['test'][i]['text'] for i in range(25000)], 
            dataset['train']['label'], 
            dataset['test']['label'])
