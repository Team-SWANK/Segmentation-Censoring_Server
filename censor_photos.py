import os
import io
import numpy as np
import redis
import settings
import helpers
import bson
import time
import base64
from PIL import Image
from skimage.io import imread
from algorithms import guassian_blur, pixelization, pixel_sort, fill_in, metadata_erase
import json

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
            print("found image in queue")
            q = bson.loads(queue[0])
            image = stringToRGB(q["image"])
            mask = stringToRGB((q["mask"]))[:,:,:1].astype(np.float)
            options = q["options"]
            options = options.strip('][').split(', ')

            # runs through options available in a specific order
            if('pixelization' in options):
                image = pixelization(image, mask)
                print("finished pixelization")
            if('pixel_sort' in options):
                image = pixel_sort(image, mask)
                print("finished pixel sort")
            if('fill_in' in options):
                image = fill_in(image, mask)
                print("finished fill in")
            if('gaussian' in options):
                image = guassian_blur(image, mask, 10)
                print("finished gaussian")

            # runs through metadata scrubber if there is anything
            image = Image.fromarray(image)
            imgByteArr = io.BytesIO()
            image.save(imgByteArr, format=q["format"])
            imgByteArr = imgByteArr.getvalue()
            image = metadata_erase(imgByteArr, q["exif"], q["tags"])

            # encodes image in base64 before sending
            encoded_image = get_response_image(image)
            db.set(q["id"], json.dumps(encoded_image))
            db.ltrim(settings.CENSOR_IMAGE_QUEUE, 1, -1)

        time.sleep(settings.SERVER_SLEEP)
        


if __name__ == "__main__":
    main()