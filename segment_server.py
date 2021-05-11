import io
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
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
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import piexif
import bson


app = Flask(__name__)
db = redis.StrictRedis(host="localhost", port=6379, db=0)
api = Api(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
db = redis.StrictRedis(host=settings.REDIS_HOST,
	port=settings.REDIS_PORT, db=settings.REDIS_DB)
db.flushall()


def prepare_image(image):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")
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
        options=request.args.get('options', '[]')
        # reads file streams and inputs them in correct array structure
        files = request.files.to_dict()
        if files.get("image") and files.get("mask"):
            im=io.BytesIO(files['image'].read())
            # img = imread(im)
            mask_img = imread(io.BytesIO(files['mask'].read()))
            # need Pillow to get exif data from image
            im=Image.open(im)
            if im.mode != "RGB":
                im = im.convert("RGB")

            exif= im.info["exif"] if "exif" in im.info else piexif.dump({"0th":{}, "Exif":{}, "GPS":{}, "1st":{}, "thumbnail":None})
            tags=request.args.get('metadata', '[]')
            tags=tags.strip('][').split(',')
            
            k = str(uuid.uuid4())
            img = np.array(im)
            image_shape = img.shape
            # swap axes in case axes do not correspond correctly
            if(image_shape[0] == mask_img.shape[1] and image_shape[1] == mask_img.shape[0]):
                mask_img = np.swapaxes(mask_img, 0,1)
            # resize mask in case it isn't the same size
            if(image_shape[0] != mask_img.shape[0] and image_shape[1] != mask_img.shape[1]):
                mask_img = resize(mask_img, (image_shape[0], image_shape[1]), mode='constant', preserve_range=True)

            img = get_response_image(img)
            mask_img = get_response_image(mask_img)
            
            d = {"id": k, "image": img, "mask": mask_img, "shape": image_shape, "options": options, "exif": exif, "tags": tags}
            db.rpush(settings.CENSOR_IMAGE_QUEUE, bson.dumps(d))
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
            image = prepare_image(image)
            # get confidence from query or set to 0.7
            confidence = float(request.args.get('confidence', 0.7))
            # ensure our NumPy array is C-contiguous as well,
            # otherwise we won't be able to serialize it
            image = image.copy(order="C")
            # generate an ID for the classification then add the
            # classification ID + image to the queue
            k = str(uuid.uuid4())
            image_shape = image.shape
            image = helpers.base64_encode_image(image)
            d = {"id": k, "image": image, "shape": image_shape, 'confidence': confidence}
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


