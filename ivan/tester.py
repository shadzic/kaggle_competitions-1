# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 21:07:43 2014

@author: ivanux
"""

from sklearn.metrics import accuracy_score


class Tester():
    """
    This class will test different algorithms. It's like a supervisor to 
    manage the prediction/classification tasks and comparison between algorithms. 
    """
    
    def __init__(self, classifier, digits):
        self._classifier = classifier
        self.digits = digits


    def score(self):
        self._classifier.fit(self.digits.X_train, self.digits.y_train)
        predictions = self._classifier.predict(self.digits.X_test)
        return accuracy_score(self.digits.y_test, predictions, normalize=True) # score = np.sum(knn_predictions == d.y_test)/d.y_test.shape[0]


