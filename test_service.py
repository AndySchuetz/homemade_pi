#
# Unit tests for web_service.py, using nose framework
#
# tests are designed to use ~/testing/ as a safe place to create, manip, and delete files
#
#
import web_service
import random
import time

UPLOAD_FOLDER = '/Users/andy/Documents/Software/imageProcessing/testing/'
MAX_IMAGES_KEPT = 5

class TestWebService:
  
  iimageDict = { 'threats':[], 'nonthreats':[]}

  def setUp(self):
    # Create fake images and corresponding image dictionary
    imageDict = { 'threats':[], 'nonthreats':[]}

    for i in range(10):
        tmpName = 'fakeImage'+str(i)+'.jpg'
        tmpFile = open(UPLOAD_FOLDER+tmpName,'w')
        tmpFile.close()
        if random.random() > 0.5:
          tmpThreat = True
          imageDict['threats'].insert( 0, (1,time.time(),tmpName) )
        else:
          tmpThreat = False
          imageDict['nonthreats'].insert( 0, (1,time.time(),tmpName) )

    print('Number of fake files created is: '+str(len(imageDict['threats'])+len(imageDict['nonthreats'] )))

    self.iimageDict = imageDict


  def tearDown(self):
    pass


  def test_manage_list(self):

    # Call manage_list to remove excess files in UPLOAD_FOLDER
    imageDictUpdate = web_service.manage_list(self.iimageDict,UPLOAD_FOLDER,MAX_IMAGES_KEPT)

    result = len(imageDictUpdate['threats'])+len(imageDictUpdate['nonthreats'])

    print('Number of fake files created is: '+str(result))

    # Assert that our cleanup of UPLOAD_FOLDER worked
    assert result <= MAX_IMAGES_KEPT
    