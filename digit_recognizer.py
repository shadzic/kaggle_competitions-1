# -*- coding: utf-8 -*-
"""
Created on Mon Jul 07 19:46:37 2014

@author: ivan lepoutre
"""

import pandas as pd
import os
# Standard scientific Python imports
import pylab as pl
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

# si windows , si linux etc..
path = os.getcwd() +'\\train.csv'
data = pd.read_csv(path,delimiter=',')
n = data.shape[0]
p = data.shape[1]

y = data['label'] # labels 
X = data.iloc[:,1:p] # features


#-------------------------------- 0 INTRODUCTION ---------------------------------------

# 0-1 plot the 9 first digit 
nine_first_digit = X.iloc[0:9,:]
for i in range(nine_first_digit.shape[0]):  
    pl.subplot(3, 3, i + 1)
    digit = nine_first_digit.iloc[i,:].values.reshape((28,28))
    pl.imshow(digit, cmap=pl.cm.gray_r, interpolation='nearest')


#-------------------------------- I DATA ANALYSIS ---------------------------------------

# I/1 covariance between features

# I/2 feature elimination 

svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, y)
ranking = rfe.ranking_.reshape(digits.images[0].shape)

# Plot pixel ranking
import pylab as pl
pl.matshow(ranking)
pl.colorbar()
pl.title("Ranking of pixels with RFE")
pl.show()



#-------------------------------- II PREDICTION -----------------------------------------
classifier = svm.SVC(gamma=0.001)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# We learn the digits on the first half of the digits
classifier.fit(X_train, y_train)
# Now predict the value of the digit on the second half:
y_hat = classifier.predict(X_test)



