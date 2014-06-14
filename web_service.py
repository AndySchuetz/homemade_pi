#
#
#   This module defines the web service that recieves images from worker pis,  
#   analyzes the images, and communicates results
#
#   If an image from a pi is classified as a threat, it is uploaded to
#   dropbox and an sms is sent to the designated notification numbers
#   TODO: finish dropbox and sms
#

import os
import json
import time
import logging

import flask
from werkzeug.utils import secure_filename
# import dropboxClient

from serverImageProcRuntime import runtimeImageProcessByWindow

#####################################################################
### Configuration Parameters
#####################################################################

UPLOAD_FOLDER = '/Users/andy/Documents/Software/imageProcessing/uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
MAX_UPLOAD_SIZE = 16 * 1024 * 1024

IMAGE_SIZE = (200,200)
MODEL_NAME = ('/Users/andy/Documents/Software/imageProcessing/model/'+
                'imageSvmClassifier.pkl')
POSITIVE_LABEL = 1
NEGATIVE_LABEL = -1

MAX_IMAGES_KEPT = 5000

LOG_FILE = 'imageServer.log'

NOTIFY = True
THREAT_TO_DROPBOX = True
REGULAR_TO_DROPBOX = True
ACCESS_TOKEN = 'YzZRHsL5ss8AAAAAAAAAAQU7MgIbj7lQmQzeorqzaEoWL691PBQsVmRYGxGBJGY6'

#####################################################################
### Set-up
#####################################################################

logging.basicConfig(filename=LOG_FILE,level=logging.DEBUG)

app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE

# Schema is [('node','date','imageName')]
imageDict = { 'threats':[], 'nonthreats':[]}
nodeId = 1

#####################################################################
### Internal functions 
#####################################################################


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def manage_list(imageDict,upLoadDir,maxImageCount):
    """
       
        Run-time maintenance function to keep the number of stored
        images in imageDict <= max/(image classes)
        The method removes entries from end of list, assumes newest are 
        inserted at front of list.
        
    """
    k = imageDict.keys()
    lenK = len(k)
    
    for i in k: # For each class of image
        while len(imageDict[i])>int(maxImageCount/lenK):
            f = imageDict[i].pop()
            logging.debug('Attempting to remove file: '+str(f[2]))
            try:
                print(f[2])
                os.remove(upLoadDir+'/'+f[2])
            # except TODO: add appropriate error catching

    
            finally:
                logging.debug(('Image '+str(f[2])+
                    ' not found in '+ upLoadDir))

    return imageDict


#####################################################################
### Web-app definition and supported URLS
#####################################################################


@app.route('/uploads', methods=['GET', 'POST']) # TODO are we really supporting get?
def upload_file():
    # TODO: stop using a global variable!
    global imageDict
    
    if flask.request.method == 'POST':
        file = flask.request.files['file']
        if file and allowed_file(file.filename):

            # Handle Upload
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # Scan window over image an apply ML classifier to each crop
            y_pred = runtimeImageProcessByWindow(MODEL_NAME,
                        os.path.join(app.config['UPLOAD_FOLDER'], filename),
                        (40,80),(.75,.75),
                        (40,40,90,160),
                        invertImage=False,
                        debug=True
                        )
            
            ### If threat
            if y_pred[0] == POSITIVE_LABEL:
                i = {"nodeId": 1, "threat": True, "filename":filename}
                imageDict['threats'].insert( 0, (nodeId,time.time(),filename) )
            
            # if THREAT_TO_DROPBOX:
            # dropboxClient.upload(ACCESS_TOKEN,filename)
            
            ### If not threat
            elif y_pred[0] == NEGATIVE_LABEL:
                i = {"nodeId": 1, "threat": False, "filename":filename}
                imageDict['nonthreats'].insert( 0,
                                               (nodeId,time.time(),filename) )
            
            ## Failed to classify
            else:
                i = {"nodeId": 1, "threat": None, "filename":filename}
        
            imageDict = manage_list(imageDict,UPLOAD_FOLDER,MAX_IMAGES_KEPT)
            logging.debug('Processed: '+str(i))

            return json.dumps(i)


    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return flask.send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/getimages')
def get_images():
    return json.dumps(imageDict)


@app.route('/lastthreat')
def get_threat():
    return json.dumps(imageDict['threats'][0][2])


from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})


#####################################################################
#####################################################################


if __name__ == '__main__':
    # app.run(debug=True)
    app.debug = True
    app.run(host='0.0.0.0')

    if app.debug:
        logging.warn('Running server in debug mode!!!')
