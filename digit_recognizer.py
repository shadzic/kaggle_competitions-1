# -*- coding: utf-8 -*-
"""
Created on Mon Jul 07 19:46:37 2014

@authors:   ivan lepoutre
            selma hadzic
"""
import pdb              # debug tool
import pandas as pd     # for read_csv function
import numpy as np
#import plyplot as plyplot
import os
import platform
# Standard scientific Python imports

if platform.system() == 'Windows':
    if os.path.exists(os.getcwd() +'\\train.csv'):
        path = os.getcwd() +'\\train.csv'
    else:
        path = 'E:\\perso\\github\\kaggle_competitions-1\\digit_recognizer\\train.csv'
elif platform.system() == 'Linux':
    path = os.getcwd() + '/train.csv'

data = pd.read_csv(path,delimiter=',')
n = data.shape[0]       # 42000
p = data.shape[1]       # 785 = 1+ 28*28
y = data['label']       # labels 
X = data.iloc[:,1:p]    # features

#%%
#-------------------------------- 0 INTRODUCTION ---------------------------------------
import pylab as pl

# 0/1 plot the 9 first digit 
nine_first_digit = X.iloc[0:9,:]
for i in range(nine_first_digit.shape[0]):
    pl.subplot(3, 3, i + 1)
    digit = nine_first_digit.iloc[i,:].values.reshape((28,28))
    pl.imshow(digit, cmap=pl.cm.summer, interpolation='none') #gray_r nearest
#%%
#-------------------------------- I DATA ANALYSIS ---------------------------------------
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.grid_search import GridSearchCV

# I/ outlier detection 

# I/1 covariance between features

# I/2 dimension reduction
# I/2/a Recursive Feature Elimination (RFE)

# SVC has difficulty to scale to dataset with more than a couple of 10000 samples
# RFE est fitte sur 1000 digits pour etre plus rapide ds un premier temps
Xbis = X.iloc[0:1000,:]
ybis = y.iloc[0:1000]

svc = SVC(kernel="linear", C=1) 
# GridSearchCV to determine the best number of features to select
#parameters = {'n_features_to_select':range(20,401)}
#rfe = RFE(svc, n_features_to_select = parameters, step = 20)
#clf = GridSearchCV(rfe, parameters)
#clf.fit(Xbis,ybis)        
# 
#n_features_to_select = clf.best_params_ 
# --> the ideal number of features to select is 207

rfe = RFE(svc, n_features_to_select = 207, step = 20)
rfe.fit(Xbis,ybis)
ranking = rfe.ranking_.reshape((28,28))
pl.matshow(ranking)
pl.colorbar()
pl.title("Ranking of pixels with RFE")
pl.show()

# I/3 feature elimination
features_to_keep = np.asarray(np.where(rfe.ranking_ == 1)).reshape(207)
Xbis = X.iloc[:,features_to_keep]

# I/4 feature selection using PCA
#pca = np.PCA(n_components=81) 
#X_new = pca.fit_transform(X)
#print(pca.explained_variance_ratio_) # sum is equal to 1
#plot(pca.explained_variance_ratio_)
#
#nine_first_digit_new = X_new[0:9,:] # X_new.iloc[0:9,:]
#for i in range(nine_first_digit_new.shape[0]):  
#    pl.subplot(3, 3, i + 1)
#    digit = nine_first_digit_new[i,:].reshape((9,9))
#    pl.imshow(digit, cmap=pl.cm.gray_r, interpolation='nearest')



#%%
#-------------------------------- II PREDICTION -----------------------------------------
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn as sklearn

X_train, X_test, y_train, y_test = train_test_split(Xbis.values, y.values, test_size=0.66, random_state=42)

# II/1 prediction using support vector machine classification --> classification binaire : inadaptée ici
#classifier = svm.SVC(gamma = 0.0001)
#X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.33, random_state=42)
#print type(X_train)
#classifier.fit(X_train, y_train) # We learn the digits on the training set 
#y_hat = classifier.predict(X_test) # Now predict the value of the digit on the test set


# II/2 prediction using support k neighborhood classification : k odd number is best
# Using grid.search to determine the best k
#parameters = {'n_neighbors':[5,7,9,11,13]}
#neigh = KNeighborsClassifier()
#clf = GridSearchCV(neigh, parameters)
#clf.fit(X_train, y_train)        
# 
#k = clf.best_params_ #voir comment récupérer automatiquement k
#clf.best_score_
#clf.score(X_test, y_test)

# KNN with best value for k: 5
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train) 
yhat_neigh = neigh.predict(X_test)
# Error rate
   # confusion_matrix(y_test, yhat_neigh, labels=None)
E_neigh = accuracy_score(y_test, yhat_neigh, normalize=True)
print( E_neigh )
# On obtient E_neigh ~ 0.95

# II/3 prediction using Linear Discriminant Analysis (LDA)
LDA = LDA()
LDA.fit(X_train, y_train)
yhat_LDA = LDA.predict(X_test)
E_LDA = accuracy_score(y_test, yhat_LDA, normalize=True)
print( E_LDA )
# KNN better than LDA (E_LDA ~ 0.86)

# II/3 prediction using Quadratic Discriminant Analysis (QDA)
QDA = QDA()
QDA.fit(X_train, y_train)
yhat_QDA = QDA.predict(X_test)
E_QDA = accuracy_score(y_test, yhat_QDA, normalize=True)
print( E_QDA )
# voir pb colinéarité des variables

# II/4 prediction using Random Forests (RF)
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
yhat_RF = RF.predict(X_test)
E_RF = accuracy_score(y_test, yhat_RF, normalize=True)
print( E_RF )
# On obtient E_RF ~ 0.91
# Analysons plus finement le paramétrage des RF pour améliorer ce résultat
''' N.B. : pour analyser vraiment finement les résultats, avoir des résultats 
robustes, il faudrait faire des tirages aléatoires et répéter les étapes
au moins 200 fois et en faire une moyenne'''
RF2 = RandomForestClassifier(n_estimators = 1000)
RF2.fit(X_train, y_train)
yhat_RF2 = RF2.predict(X_test)
E_RF2 = accuracy_score(y_test, yhat_RF2, normalize=True)
print( E_RF2 )
''' Analyse n°1 : augmenter le nombre d'arbres donne de meilleurs résultats 
mais cette relation est non linéaire :
n_estimators = 10 --> Error = 0.908
n_estimators = 100 --> Error = 0.9470
n_estimators = 500 --> Error = 0.9503
n_estimators = 1000 --> Error = 0.9509
Il serait judicieux de prendre un bon compromis complexité/résultat'''

RF3 = RandomForestClassifier(min_samples_leaf = 3)
RF3.fit(X_train, y_train)
yhat_RF3 = RF3.predict(X_test)
E_RF3 = accuracy_score(y_test, yhat_RF3, normalize=True)
print( E_RF3 )
''' Analyse n°2 : augmenter fortement le nombre minimum d'observations dans 
les feuilles diminue la performance du classifieur :
min_samples_leaf = 4 --> Error = 0.9142
min_samples_leaf = 10 --> Error = 0.908
min_samples_leaf = 100 --> Error = 0.83
min_samples_leaf = 1000 --> Error = 0.66
On peut donc l'augmenter jusqu'à 3 mais cela risque de diminuer la performance
des random forests, et cela dépend du nombre d'arbres créés.
Ainsi pour 10 arbres créés, 3 observations minimum dans les feuilles est optimal.
Mais pour 100 arbres créés, seulement 2
Ce paramètre a peu d'impact et peu diminuer la précision de la prédiction 
on garde la valeur par défaut '''

RF4 = RandomForestClassifier(n_estimators = 1000, max_depth = 50)
RF4.fit(X_train, y_train)
yhat_RF4 = RF4.predict(X_test)
E_RF4 = accuracy_score(y_test, yhat_RF4, normalize=True)
print( E_RF4 )
''' Analyse n°3 : limiter la taille de l'arbre peut améliorer la prédiction'''

# Remarque : bizarrement Knn plus performant que RF (!)

# Multi-class logistic regression (one vs all)
regLog = LogisticRegression()
regLog.fit(X_train, y_train)
yhat_regLog = regLog.predict(X_test)
E_regLog = accuracy_score(y_test, yhat_regLog, normalize=True)
print( E_regLog )   # 0.896103896104


#%%
#-------------------------------- III DEEP LEARNING --------------------------------------
from theano import *

# III/1 LeNet5 neural network (https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/convolutional_mlp.py)

#%%
#-------------------------------- IV TESTING ---------------------------------------------


print(__doc__)

# Standard scientific Python imports
import pylab as pl

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# The digits dataset
digits = datasets.load_digits()

# The data that we are interested in is made of 8x8 images of digits,
# let's have a look at the first 3 images, stored in the `images`
# attribute of the dataset. If we were working from image files, we
# could load them using pylab.imread. For these images know which
# digit they represent: it is given in the 'target' of the dataset.
for index, (image, label) in enumerate(zip(digits.images, digits.target)[:4]):
    pl.subplot(2, 4, index + 1)
    pl.axis('off')
    pl.imshow(image, cmap=pl.cm.gray_r, interpolation='nearest')
    pl.title('Training: %i' % label)

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples / 2:]
predicted = classifier.predict(data[n_samples / 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

for index, (image, prediction) in enumerate(
        zip(digits.images[n_samples / 2:], predicted)[:4]):
    pl.subplot(2, 4, index + 5)
    pl.axis('off')
    pl.imshow(image, cmap=pl.cm.gray_r, interpolation='nearest')
    pl.title('Prediction: %i' % prediction)

pl.show()






