#
#
#	Server Side Modules image manipulation
#
#
from __future__ import print_function

import os
import math
import logging

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

import numpy as np
import Image


def singleImage2nparray(imagePath,imageSize):
    '''
       
        Convert one image to nparray
        
    '''

    imageShape = (1, imageSize[0]*imageSize[1] )
    i = Image.open(imagePath)
    # Downsize
    isub = i.resize(imageSize,Image.ANTIALIAS)
    
    # Rotate image if from pi mounted to roof 
    if (os.path.basename(imagePath)[0:5]=='piPic'):
        isub = isub.rotate(180)
    
    # Make black and white
    ii = isub.convert('L')
    # Make 32-bit integer pixels
    iii = ii.convert('I')
    # Convert to ndarray
    npiii = np.array(iii).flatten()
    
    return np.reshape(npiii,imageShape)
    


def images2nparray(imageDir,imageSize,yFlag):
    '''
        
        Convert all images in directory to nparrays
        
    '''

    print(imageDir)
    imageShape = (1, imageSize[0]*imageSize[1] )

    # Get target dir contents and drop '.DS_Store'
    contsAll = os.listdir(imageDir)
    contsClean = [ c for c in contsAll if not (c.startswith('.'))]
    print('Clean Contents '+str(contsClean))

    X = np.empty((0,imageSize[0]*imageSize[1]))
    y = np.empty((0,))
    
    for i in contsClean:

        print('Image: '+ str(i)+ ' is coded ' + str(yFlag) )
    
        Xtmp = singleImage2nparray(imageDir+i,imageSize)
    
        X = np.append(X, Xtmp ,axis=0)
        y = np.append(y,yFlag)

    return (X,y,contsClean)


def createTrainingSetByWindow(imagePath,
                                windowSize=(40,80),
                                overlap=(0.5,0.5),
                                cropBox=(80,40,160,200),
                                invertImage=False,
                                debug=False
                                ):
    '''
        
        Move a window across an image, and create a set of cropped images 
        to be used as training set elements
        
    '''
    
    leftCornerHor = 0
    leftCornerVer = 0
    w = windowSize[0]
    h = windowSize[1]
    ovlapHor = overlap[0]
    ovlapVer = overlap[1]
    
    i = Image.open(imagePath)
    # Downsize
    
    ## Rotate if pi mounted upside down
    if invertImage:
        i = i.rotate(180)
    
    ## Crop
    isub = i.crop(cropBox)
    
    imageWidth,imageHeight = isub.size
    step = 1
    y_pred = -1
    
    # Scan window over rows first...
    while (leftCornerVer+h <= imageHeight):
        # Then columns
        while (leftCornerHor+w <= imageWidth):
            
            # print('Step '+str(step)+' '+str(leftCornerHor)+' '+str(leftCornerVer),' ',str(w),str(h))
            step = step + 1
            ic = isub.crop( (int(leftCornerHor),int(leftCornerVer),
                             int(leftCornerHor+w),
                             int(leftCornerVer+h)) )
            
            # if y_pred == 1:
            if debug:
                ic.show()

            ic.convert('RGB').save((imagePath+'_'+str(step)+'.jpg'))
            
            # Move window
            leftCornerHor += math.ceil(float(w)*(1.0-ovlapHor))
        
        # After completing a row move down one row and repeat
        leftCornerHor = 0.
        leftCornerVer += math.ceil(float(h)*(1.0-ovlapVer))
    
    return step


def writeDataSets(X,y,xFileName='X.csv',yFileName='y.csv',imageNameList=None,
                   nameFileName='Names.csv'):
	
    # TODO: Use with syntax to avoid an open file in case of error.
    
    xFile = open(xFileName,'w')

    for i in X:
        s = ''
        for j in i:
            s += str(j)+', '
        xFile.write(s+'\n')
    xFile.close()

    yFile = open(yFileName,'w')
    for i in y:
        yFile.write(str(i)+'\n')
    yFile.close()

    # Optionally, write file with list of image file names
    if (imageNameList<>None):
        nameFile = open(nameFileName,'w')
        for i in imageNameList:
            nameFile.write(i+'\n')
        nameFile.close()

    return


if __name__ == "__main__":

    # Dataset location and file structure
    trainingDir = '/Users/andy/Documents/Software/imageProcessing/TrainingSet/'
    testDir = '/Users/andy/Documents/Software/imageProcessing/TestSet/'
    positiveSubDir = 'positive/'
    negativeSubDir = 'negative/'

    # Final image size fed to algorithm
    # imageSize = (200,200)
    imageSize = (40,80)

    ####################################
    ####################################

    ## Create training set
    Xp,yp,imageNamesp = images2nparray(trainingDir+positiveSubDir,imageSize,1)
    Xn,yn,imageNamesn = images2nparray(trainingDir+negativeSubDir,imageSize,-1)

    X = np.append(Xp,Xn, axis=0)
    y = np.append(yp,yn, axis=0)
    fileNames = imageNamesp+imageNamesn
        
    writeDataSets(X,y,imageNameList=fileNames)
    print('Success, wrote X.csv, Y.csv, and Names.csv')

    ## Create test set
    Xp_test,yp_test,imageNamesp_test = images2nparray(testDir+positiveSubDir,
                                                      imageSize,1)
    Xn_test,yn_test,imageNamesn_test = images2nparray(testDir+negativeSubDir,
                                                      imageSize,-1)
    print('Xp_test shape = '+str(Xp_test.shape))
    
    X_test = np.append(Xp_test,Xn_test, axis=0)
    y_test = np.append(yp_test,yn_test, axis=0)
    fileNames_test = imageNamesp_test + imageNamesn_test

    writeDataSets(X_test,y_test,'Xtest.csv','yTest.csv',
                  imageNameList=fileNames_test,nameFileName='NamesTest.csv')
    
