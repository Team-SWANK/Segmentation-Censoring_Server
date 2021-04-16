import os
import math
import cv2
from keras.utils.generic_utils import to_list
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
import settings
import helpers
import redis
import time
import json

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# connect to Redis server
db = redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)

# IMG_WIDTH = 128 # for faster computing
# IMG_HEIGHT = 128 # for faster computing
# IMG_CHANNELS = 3

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def get_unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    """Function to define the UNET Model"""
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

def main():
    print("* Loading model...")
    input_img = Input((settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, settings.IMAGE_CHANS), name='img')
    model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    model.load_weights('face-segmentation.h5')
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print("* Model loaded")

    while True:
        # attempt to grab a batch of images from the database, then
        # initialize the image IDs and batch of images themselves
        queue = db.lrange(settings.SEGMENT_IMAGE_QUEUE, 0, settings.BATCH_SIZE - 1)
        imageIDs = []
        batch = None
        # loop over the queue
        for q in queue:
            # deserialize the object and obtain the input image
            q = json.loads(q.decode("utf-8"))
            image = helpers.base64_decode_image(q["image"], settings.IMAGE_DTYPE, q["shape"])
            # check to see if the batch list is None
            if batch is None:
                batch = image
            # otherwise, stack the data
            else:
                batch = np.vstack([batch, image])
            # update the list of image IDs
            imageIDs.append(q["id"])
        
        # check to see if we need to process the batch
        if len(imageIDs) > 0:
            print('* ImageIDs: {}'.format(imageIDs))
            faces = {}
            for i in range(len(batch)):
                print("* Batch size: {}".format(batch[i].shape))
                image_cv2 = np.array(batch[i], dtype="uint8")
                face = faceCascade.detectMultiScale(
                    image_cv2,
                    scaleFactor=1.3,
                    minNeighbors=3,
                    minSize=(30, 30)
                )
                # print("Found {0} Faces!".format(len(faces)))
                if len(face) == 0:
                    faces[i] = [[0, 0, batch[i].shape[0], batch[i].shape[1]]]
                else:
                    faces[i] = face
            # print(faces)
            numFaces = 0
            for i in range(len(faces)):
                for _ in faces[i]:
                    numFaces += 1
            X = np.zeros((numFaces, 128, 128, 3), dtype=np.float32)
            X_positions = []
            index=0
            for imageNum, faceList in faces.items():
                image = batch[imageNum]
                for(x, y, w, h) in faceList:
                    transpose_x, transpose_y = w * 0.75, h * 0.75
                    x_img = math.floor(x-transpose_x) if math.floor(x-transpose_x) >= 0 else 0
                    y_img = math.floor(y-transpose_y) if math.floor(y-transpose_y) >= 0 else 0
                    w_img = math.floor(w+2*transpose_x) if math.floor(w+2*transpose_x) <= image.shape[0] else image.shape[0]
                    h_img = math.floor(h+2*transpose_y) if math.floor(w+2*transpose_y) <= image.shape[1] else image.shape[1]
                    X_positions.append([x_img, y_img, w_img, h_img])
                    roi_color = image[y_img:h_img+y_img, x_img:w_img+x_img]
                    X[index] = resize(roi_color, (settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH), mode='constant', preserve_range=True)
                    index +=1
            preds_test = (model.predict(X, verbose=1) > 0.9).astype(np.uint8)
            index = 0
            masks = []
            for imageNum, faceList in faces.items():
                upsampled_mask = np.zeros((batch[imageNum].shape[0], batch[imageNum].shape[1]), dtype=np.uint8)
                for i in range(index, index+len(faceList)):
                    coords = X_positions[i]
                    if coords[2] == upsampled_mask.shape[0] and coords[3] == upsampled_mask.shape[1]:
                        section = resize(np.squeeze(preds_test[i]), (upsampled_mask.shape[0], upsampled_mask.shape[1]), preserve_range=True)
                        upsampled_mask[coords[0]:coords[2]+coords[0], coords[1]:coords[3]+coords[1]] += section.astype(np.uint8)
                    else:
                        section = resize(np.squeeze(preds_test[i]), (coords[3], coords[2]), preserve_range=True)
                        upsampled_mask[coords[1]:coords[3]+coords[1], coords[0]:coords[2]+coords[0]] += section.astype(np.uint8)
                masks.append(upsampled_mask)
                index += len(faceList)
            
            for imageID, i in zip(imageIDs, range(len(imageIDs))):
                db.set(imageID, json.dumps(masks[i].tolist()))
            # remove the set of images from our queue
            db.ltrim(settings.SEGMENT_IMAGE_QUEUE, len(imageIDs), -1)
        # sleep for a small amount
        time.sleep(settings.SERVER_SLEEP)

if __name__ == "__main__":
    main()