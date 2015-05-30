#
#
#     Server side image processing
#   Adding PCA dimensionality reduction
#
from __future__ import print_function

from time import time

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.decomposition import RandomizedPCA

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
    X = X[:,0:3200] # TODO Fix nan column
    y = np.genfromtxt(dataDir+labelFile, delimiter=',')
    n_samples,n_features = X.shape

    ############################################################################
    # PCA for dimensionality reduction
    ############################################################################
    n_components = 25

    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X)
    joblib.dump(pca, dataDir+'transform.pkl')
    eigenpeople = pca.components_.reshape((n_components, 80, 40))        # TODO: automatically get h and w

    X_train_pca = pca.transform(X)

    ############################################################################
    # Train a SVM classification model
    ############################################################################

    print("Fitting the classifier to the training set")
    t0 = time()

    param_grid = {'C': [1e2, 5e2, 1e3, 5e3, 1e4],
        'gamma': [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005], }

    clf = GridSearchCV(SVC(kernel='linear', class_weight='auto'), param_grid) # 13 errors in 107 test set
    # clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
    
    clf = clf.fit(X_train_pca, y)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
        
     # Save model to disk
    clf = clf.best_estimator_
    joblib.dump(clf, dataDir+'imageSvmClassifier.pkl')

    y_pred = clf.predict(X_train_pca)
    print(classification_report(y, y_pred, target_names=list(str(y))))

    ############################################################################
    # Quantitative evaluation of the model quality on the test set
    ############################################################################

    Xtest = np.genfromtxt(dataDir+testDataFile, delimiter=',')
    Xtest = Xtest[:, 0:3200]
    ytest = np.genfromtxt(dataDir+testLabelFile, delimiter=',')

    nameListTest = []
    # fName = open(dataDir+testNameFile)
    # nl = fName.readline()
    # while nl<>'':
    #     nameListTest.append(nl)
    #     nl = fName.readline()

    with open(dataDir+testNameFile) as fName:
        for line in fName:
            nameListTest.append(line)

    print("Predicting presence of people in the test set")
    t0 = time()
    X_test_pca = pca.transform(Xtest)
    y_pred = clf.predict(X_test_pca)
    print("done in %0.3fs" % (time() - t0))

    # print(classification_report(ytest, y_pred, target_names=list(strytest)))

    print(y_pred)

    nn = ytest.shape[0]
    errorCount = 0

    for i in range(ytest.shape[0]):
        flag = ''
        if (ytest[i]<>y_pred[i]):
            errorCount += 1
            flag = '---- error ---'
        print('For '+nameListTest[i].strip()+' '+'Actual: '+str(ytest[i])+
            ' Predicted: '+str(y_pred[i])+flag)

    print(str(nn)+' test set elements')
    print(str(errorCount)+' incorrectly classified')

     # print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
