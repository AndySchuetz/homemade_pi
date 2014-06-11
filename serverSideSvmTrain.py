#
#
#	Server side image processing
#
#
from __future__ import print_function

from time import time
import os
import logging

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from sklearn.externals import joblib

import numpy as np



if __name__ == "__main__":

	# Dataset location and file structure
    dataDir = '/Users/andy/Documents/Software/imageProcessing/'

    dataFile = 'X.csv'
    labelFile = 'y.csv'
    testDataFile = 'Xtest.csv'
    testLabelFile =  'ytest.csv'
    testNameFile = 'NamesTest.csv'

    modelName = 'svmImageClassifier.pkl'

	############################################################################

    X = np.genfromtxt(dataDir+dataFile, delimiter=',')
    # X = X[:,0:40000] # TODO Fix nan column
    X = X[:,0:3200] # TODO Fix nan column
    y = np.genfromtxt(dataDir+labelFile, delimiter=',')
    n_samples,n_features = X.shape

	############################################################################
	# Train a SVM classification model

    print("Fitting the classifier to the training set")
    t0 = time()

	# param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
	#    'gamma': [0.0005, 0.001, 0.005, 0.01, 0.1], }
    # param_grid = {'C': [1e2,5e2,1e3, 5e3, 1e4],
    #    'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    param_grid = {'C': [1e2,5e2,1e3, 5e3, 1e4],
        'gamma': [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005], }
    
    
    clf = GridSearchCV(SVC(kernel='linear', class_weight='auto'), param_grid) # 13 errors in 107 test set
    # clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
    
    clf = clf.fit(X, y)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
        
	# Save model to disk
    clf = clf.best_estimator_
    joblib.dump(clf, dataDir+'imageSvmClassifier.pkl')

    y_pred = clf.predict(X)
    print(classification_report(y, y_pred, target_names=list(str(y))))

	############################################################################
	# Quantitative evaluation of the model quality on the test set

    Xtest = np.genfromtxt(dataDir+testDataFile, delimiter=',')
    # Xtest = Xtest[:,0:40000] # TODO Fix nan column
    Xtest = Xtest[:,0:3200]
    ytest = np.genfromtxt(dataDir+testLabelFile, delimiter=',')

    nameListTest = []
    fName = open(dataDir+testNameFile)
    nl = fName.readline()
    while nl<>'':
        nameListTest.append(nl)
        nl = fName.readline()
    fName.close
    # print(nameListTest)

    print("Predicting presence of people in the test set")
    t0 = time()
    y_pred = clf.predict(Xtest)
    print("done in %0.3fs" % (time() - t0))

    # print(classification_report(ytest, y_pred, target_names=list(strytest)))

    print(y_pred)

    nn = ytest.shape[0]
    errorCount = 0
    for i in range(ytest.shape[0]):
        print('For '+nameListTest[i].strip()+' '+'Actual: '+str(ytest[i])+
            ' Predicted: '+str(y_pred[i]))
        if (ytest[i]<>y_pred[i]):
            errorCount += 1
    print(str(nn)+' test set elements')
    print(str(errorCount)+' incorrectly classified')

	# print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
