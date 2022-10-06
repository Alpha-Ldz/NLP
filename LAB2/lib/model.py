from typing import Callable, List, Tuple

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import seaborn as sns

from numpy import ndarray

def createModel() -> MultinomialNB :
    return MultinomialNB()

def trainModel(
    model: MultinomialNB, 
    data: ndarray, 
    labels: List[int]
) -> MultinomialNB:
    return model.fit(data, labels)

def countDif(
    model: MultinomialNB,
    testData: List[int],
    testLabel: List[int]
):
    pred = model.predict(testData)
    rtn = (testLabel != pred).sum()
    return rtn

def testModel(
    model: MultinomialNB, 
    testData: List[int], 
    testLabel: List[int], 
    showHeatmap: bool = False
):
    pred = model.predict(testData)

    num_fail = (testLabel != pred).sum()

    print("Number of mislabeled points out of a total " + str(testData.shape[0]) + " points : " + str(num_fail) + ", the accuracy is: " + str((len(testData) - num_fail) / len(testData) ))

    if showHeatmap:
        conf_mat = confusion_matrix(testLabel, pred)
        sns.heatmap(conf_mat, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
