#
#
#   This module provides server runtime image classiciation functions
#
#
import time
import math

import numpy as np
# import Image
from PIL import Image
import sklearn


def runtimeImageProcessByWindow(modelPath,imagePath,
                                windowSize=(40,80),
                                overlap=(0.5,0.5),
                                cropBox=(80,40,160,200),
                                invertImage=False,
                                debug=False
                                ):
    """
        
        Classify an image as containing or not containing a threat.  The image
        is classified by moving a window across the portion of the image 
        remaining after the specified cropBox, and applying the classification 
        algorithm to each crop defined by the window area.
        
    """
    
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

    # Make black and white
    isub = isub.convert('L')
    # Make 32-bit integer pixels
    isub = isub.convert('I')

    imageWidth,imageHeight = isub.size
    step = 1
    y_pred = -1
    

    # Scan window over rows first...
    while ((leftCornerVer+h <= imageHeight) & (y_pred==-1) ):
        # Then columns
        while ((leftCornerHor+w <= imageWidth) & (y_pred==-1) ):
            
            step = step + 1
            ic = isub.crop( (int(leftCornerHor),int(leftCornerVer),
                             int(leftCornerHor+w),
                             int(leftCornerVer+h)) )
            
            # Convert to ndarray
            X = np.array(ic).flatten()
            # print(X.shape)
            
            imageShape = (1, w*h )
            np.reshape(X,imageShape)
            
            # Make prediction
            clf = sklearn.externals.joblib.load(modelPath)
            y_pred = clf.predict(X)
        
            # ic.show()
            
            if y_pred == 1:
                if debug:
                    ic.show()
                    # ic.thumbnail((80,160))
                    ic.convert('RGB').save((imagePath+'threatImage'+
                                            str(time.time())+'.jpg'))
                return y_pred
            
            # Move window
            leftCornerHor += math.ceil(float(w)*(1.0-ovlapHor))
        
        # After completing a row move down one row and repeat
        leftCornerHor = 0.
        leftCornerVer += math.ceil(float(h)*(1.0-ovlapVer))
    
    return y_pred

############################################################################
############################################################################

if __name__ == "__main__":
    
    
    IMAGE_DIR = '/Users/andy/Documents/Software/imageProcessing/'
    MODEL_NAME = './model/imageSvmClassifier.pkl'
    TEST_IMAGE = 'windowTest.jpg'

    t0 = time.time()
    y_pred = runtimeImageProcessByWindow(IMAGE_DIR+MODEL_NAME,
            IMAGE_DIR+TEST_IMAGE,
            (80,160),(40,80),(.75,.75))

    print("Prediction done in %0.3fs" % (time.time() - t0))
    print('Prediction for '+ testImage +' is '+str(y_pred))
    