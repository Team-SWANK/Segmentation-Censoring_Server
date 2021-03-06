import io
import numpy as np
import redis
import settings
import bson
import time
import base64
from PIL import Image
from skimage.io import imread
from algorithms import guassian_blur, pixelization, pixel_sort, fill_in, black_bar, metadata_erase
import json
import os
import atexit

# connect to Redis server
db = redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)

def get_response_image(byte_arr):
	encoded_image = base64.encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
	return encoded_image

def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    return imread(io.BytesIO(imgdata))

def main():
    print("starting loop to check CENSOR_IMAGE_QUEUE")
    while True:
        queue = db.lrange(settings.CENSOR_IMAGE_QUEUE, 0, 1)
        if len(queue) > 0:
            q = bson.loads(queue[0])
            image = stringToRGB(q["image"])
            mask = stringToRGB((q["mask"]))[:,:,:1].astype(np.float)
            options = q["options"]
            options = options.strip('][').split(',')
            print("* Censoring image with shape: {}".format(image.shape))
            print("* Censoring with options: {}".format(options))

            # runs through options available in a specific order
            if('black_bar' in options):
                image = black_bar(Image.fromarray(image), mask)
            if('pixelization' in options):
                image = pixelization(image, mask)
            if('pixel_sort' in options):
                image = pixel_sort(image, mask)
            if('fill_in' in options):
                image = fill_in(image, mask)
            if('gaussian' in options):
                if('pixel_sort' in options):
                    image = guassian_blur(image, mask, 3)
                else:
                    image = guassian_blur(image, mask, 7)

            # runs through metadata scrubber if there is anything
            print("* Erasing metadata")
            image = Image.fromarray(image)
            imgByteArr = io.BytesIO()
            image.save(imgByteArr, "JPEG")
            imgByteArr = imgByteArr.getvalue()
            image = metadata_erase(imgByteArr, q["exif"], q["tags"])

            # encodes image in base64 before sending
            encoded_image = get_response_image(image)
            db.set(q["id"], json.dumps(encoded_image))
            db.ltrim(settings.CENSOR_IMAGE_QUEUE, 1, -1)

        time.sleep(settings.SERVER_SLEEP)
        
def restartProcesses():
    os.system("workon photosense_api")
    os.system("nohup bash commands.sh")

atexit.register(restartProcesses)

if __name__ == "__main__":
    main()