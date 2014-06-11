#
#
#   This is the image processing module that runs on the py
#
#
#
#
#
import datetime
import os
import time

import requests
import json
import picamera

###############################################################################
# Configuration
###############################################################################

IMAGE_DIR = '/home/pi/imageClient/'
SERVER_URL = 'http://192.168.1.6:5000/uploads'

IMAGE_NAME = 'piPic'
IMAGE_SUFFIX = '.jpg'
IMAGE_RES = (160,160)
WARMUP_TIME = 2    # Seconds allowed for camera to wake up
IFREQUENCY_TIME = 5    # Seconds between camera cycles

###############################################################################
# Get image
###############################################################################

while True:
    with picamera.PiCamera() as camera:
        camera.resolution = IMAGE_RES
        camera.start_preview()
        # Camera warm-up time
        time.sleep(2)
        dateStamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        fName = IMAGE_DIR+IMAGE_NAME+dateStamp+IMAGE_SUFFIX
        camera.capture(fName,quality=50)
	print('Captured image '+fName)

###############################################################################
# Push to server
###############################################################################
        files = {'file':open(fName,'rb')}
        r = requests.post(SERVER_URL, files=files)
	if r.status_code == 200:
		print('Posted image to server with response '+str(r.json()))
	else:
		print('Post status for image '+fName+' is '+str(r.status_code))

	# delete image
	os.remove(fName)

###############################################################################
# Wait
###############################################################################
    time.sleep(IFREQUENCY_TIME)
