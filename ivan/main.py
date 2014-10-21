# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 20:01:04 2014

@author: ivanux
"""

from data import Digits
from tester import Tester

TEST_SIZE = 0.25

d = Digits()

d.plot_some_digits() # plotting the nine first hand written digits

d.train_test_split(test_size = TEST_SIZE)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 10)
t = Tester(knn, d)
knn_score = t.score() # 0.94758297258297264


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf_score= Tester(rf, d).score()

