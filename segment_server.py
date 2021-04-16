import os
import io
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
import math
import redis
import numpy as np
import redis
import uuid
import time
import settings
import helpers
import json
import datetime
import base64
from skimage.io import imread
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image


app = Flask(__name__)
db = redis.StrictRedis(host="localhost", port=6379, db=0)
api = Api(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
db = redis.StrictRedis(host=settings.REDIS_HOST,
	port=settings.REDIS_PORT, db=settings.REDIS_DB)
db.flushall()


def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")
	# resize the input image and preprocess it
	# image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	# return the processed image
	return image

def get_response_image(image_path):
	pil_image = Image.fromarray(np.uint8(image_path)) # reads the PIL image
	byte_arr = io.BytesIO()
	pil_image.save(byte_arr, format='JPEG') # convert the PIL image to byte array
	encoded_image = base64.encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
	return encoded_image

class Censor(Resource):
    def get(self):
        return {'hello': 'world'}
    def post(self):
        data = {"success": False}
        options=request.args.get('options', None)
        # reads file streams and inputs them in correct array structure
        files = request.files.to_dict()
        if files.get("image") and files.get("mask"):
            img = imread(io.BytesIO(files['image'].read()))[:,:,:3]
            mask_img = imread(io.BytesIO(files['mask'].read()))[:,:,:3]
            # img = prepare_image(img, (settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT))
            # img = img.copy(order="C")
            # mask_img = mask_img.copy(order="C")
            k = str(uuid.uuid4())
            image_shape = img.shape
            # img = helpers.base64_encode_image(img)
            # mask_img = helpers.base64_encode_image(mask_img)
            img = get_response_image(img)
            mask_img = get_response_image(mask_img)
            d = {"id": k, "image": img, "mask": mask_img, "shape": image_shape, "options": options}
            db.rpush(settings.CENSOR_IMAGE_QUEUE, json.dumps(d))
            timeDelta = datetime.datetime.now() + datetime.timedelta(minutes=3)

            while True:
                output = db.get(k)
                if output is not None:
                    output = output.decode("utf-8")
                    data["ImageBytes"] = json.loads(output)
                    db.delete(k)
                    break
                time.sleep(settings.CLIENT_SLEEP)
                if datetime.datetime.now() > timeDelta:
                    print('timeout')
                    break

            if "ImageBytes" in data:
                data["success"] = True
        return jsonify(data)

class Segment(Resource):
    def get(self):
        return {'hello': 'world'}

    def post(self):
        data = {"success": False}
        files = request.files.to_dict()
        if files.get("image"):
            image =Image.open(io.BytesIO(files['image'].read()))
            image = prepare_image(image, (settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT))
            # ensure our NumPy array is C-contiguous as well,
            # otherwise we won't be able to serialize it
            image = image.copy(order="C")
            # generate an ID for the classification then add the
            # classification ID + image to the queue
            k = str(uuid.uuid4())
            image_shape = image.shape
            image = helpers.base64_encode_image(image)
            d = {"id": k, "image": image, "shape": image_shape}
            db.rpush(settings.SEGMENT_IMAGE_QUEUE, json.dumps(d))
            timeDelta = datetime.datetime.now() + datetime.timedelta(minutes=3)

            # keep looping until our model server returns the output
            # predictions
            while True:
                # attempt to grab the output predictions
                output = db.get(k)
                # check to see if our model has classified the input
                # image
                if output is not None:
                    # add the output predictions to our data
                    # dictionary so we can return it to the client
                    output = output.decode("utf-8")
                    data["predictions"] = json.loads(output)

                    # delete the result from the database and break
                    # from the polling loop
                    db.delete(k)
                    break

                # sleep for a small amount to give the model a chance
                # to classify the input image
                time.sleep(settings.CLIENT_SLEEP)
                if datetime.datetime.now() > timeDelta:
                    print('timeout')
                    break

            # indicate that the request was a success
            if "predictions" in data:
                data["success"] = True

        # return the data dictionary as a JSON response
        return jsonify(data)


api.add_resource(Segment, '/Segment')
api.add_resource(Censor, '/Censor')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8000)
