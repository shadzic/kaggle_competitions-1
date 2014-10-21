# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 17:25:37 2014

@author: ivan lepoutre
"""


from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.grid_search import GridSearchCV


def features_to_select(X, y, mode = 'slow'):
    """
    Function using GridSearchCV returning the best number of features to select. 
    """
    if mode == 'slow':    
        # SVC has difficulty to scale to dataset with more than a couple of 10000 samples
        Xbis = X.iloc[0:1000,:]
        ybis = y.iloc[0:1000]    
        
        svc = SVC(kernel="linear", C=1) 
        parameters = {'n_features_to_select' : range(20,401)}
        rfe = RFE(svc, n_features_to_select = parameters, step = 20)
        clf = GridSearchCV(rfe, parameters)
        clf.fit(Xbis, ybis)
        n_features_to_select = clf.best_params_ 
        return n_features_to_select
    else:
        # the ideal number of features to select is 207
        return 207 