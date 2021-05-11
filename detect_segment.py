import os
import math
import cv2
import numpy as np
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
    print("* Loading models...")
    input_img = Input((settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, settings.IMAGE_CHANS), name='img')
    model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    model.load_weights('face-segmentation.h5')
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
    print("* Models loaded")

    while True:
        # attempt to grab a batch of images from the database, then
        # initialize the image IDs and batch of images themselves
        queue = db.lrange(settings.SEGMENT_IMAGE_QUEUE, 0, settings.BATCH_SIZE - 1)
        imageIDs = []
        batch = []
        confidences = []
        # loop over the queue
        for q in queue:
            # deserialize the object and obtain the input image
            q = json.loads(q.decode("utf-8"))
            image = helpers.base64_decode_image(q["image"], settings.IMAGE_DTYPE, q["shape"])
            # check to see if the batch list is None
            batch.append(image)
            # update the list of image IDs
            imageIDs.append(q["id"])
            # add to list of confidence attributes
            confidences.append(q["confidence"])
        
        # check to see if we need to process the batch
        if len(imageIDs) > 0:
            print('* ImageIDs: {}'.format(imageIDs))
            faces = {}
            for i in range(len(batch)):
                print("* Batch size: {}".format(batch[i][0].shape))
                image = np.array(batch[i][0], dtype="uint8")
                (h,w) = image.shape[:2]
                if h > 1000 or w > 1000:
                    blob = cv2.dnn.blobFromImage(image, mean=(104.0, 117.0, 123.0), swapRB=True)
                    net.setInput(blob)
                    detections = net.forward()
                else:
                    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0), swapRB=True)
                    net.setInput(blob)
                    detections = net.forward()

                if detections.shape[2] == 0:
                    faces[i] = np.array([[[[0., 1., 1., 0., 0., 1., 1.]]]])
                else:
                    faces[i] = detections

            X = []
            X_positions = []
            numFaces = []
            for imageNum, detections in faces.items():
                image = batch[imageNum][0]
                confidence = confidences[imageNum]
                (h,w) = image.shape[:2]
                index=0
                # loop over the detections
                for i in range(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with the prediction
                    conf = detections[0, 0, i, 2]
                    # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
                    if conf > confidence:
                        # compute the (x, y)-coordinates of the bounding box for the object
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        if startX > image.shape[1]-10 or startY > image.shape[0]-10:
                            continue
                        w_box = endX - startX
                        h_box = endY - startY
                        transpose_x, transpose_y = w_box * 0.75, h_box * 0.75
                        x_img = math.floor(startX-transpose_x) if math.floor(startX-transpose_x) >= 0 else 0
                        y_img = math.floor(startY-transpose_y) if math.floor(startY-transpose_y) >= 0 else 0
                        w_img = math.floor(w_box+2*transpose_x) if x_img + math.floor(w_box+2*transpose_x) <= image.shape[1] else image.shape[1] - x_img
                        h_img = math.floor(h_box+2*transpose_y) if y_img + math.floor(h_box+2*transpose_y) <= image.shape[0] else image.shape[0] - y_img
                        X_positions.append([x_img, y_img, w_img, h_img])
                        # 2d array of cropped image
                        roi_color = image[y_img:h_img+y_img, x_img:w_img+x_img]
                        X.append(resize(roi_color, (settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH), mode='constant', preserve_range=True))
                        index +=1
                if(index == 0):
                    index = 1
                    box = [0., 0., 1., 1.] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    w_box = endX - startX
                    h_box = endY - startY
                    transpose_x, transpose_y = w_box * 0.75, h_box * 0.75
                    x_img = math.floor(startX-transpose_x) if math.floor(startX-transpose_x) >= 0 else 0
                    y_img = math.floor(startY-transpose_y) if math.floor(startY-transpose_y) >= 0 else 0
                    w_img = math.floor(w_box+2*transpose_x) if x_img + math.floor(w_box+2*transpose_x) <= image.shape[1] else image.shape[1] - x_img
                    h_img = math.floor(h_box+2*transpose_y) if y_img + math.floor(h_box+2*transpose_y) <= image.shape[0] else image.shape[0] - y_img
                    X_positions.append([x_img, y_img, w_img, h_img])
                    # 2d array of cropped image
                    roi_color = image[y_img:h_img+y_img, x_img:w_img+x_img]
                    X.append(resize(roi_color, (settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH), mode='constant', preserve_range=True))
                numFaces.append(index)
                
            X = np.array(X).astype(np.float32)
            preds_test = (model.predict(X, verbose=1) > 0.8).astype(np.uint8)
            index = 0
            masks = []
            for imageNum, detections in faces.items():
                upsampled_mask = np.zeros((batch[imageNum][0].shape[0], batch[imageNum][0].shape[1]), dtype=np.uint8)
                for i in range(index, index+numFaces[imageNum]):
                    coords = X_positions[i]
                    section = resize(np.squeeze(preds_test[i]), (coords[3], coords[2]), mode='constant', preserve_range=True, order=0)
                    upsampled_mask[coords[1]:coords[3]+coords[1], coords[0]:coords[2]+coords[0]] += section.astype(np.uint8)
                masks.append(upsampled_mask)
                index += numFaces[imageNum]
            
            for imageID, i in zip(imageIDs, range(len(imageIDs))):
                db.set(imageID, json.dumps(masks[i].tolist()))
            # remove the set of images from our queue
            db.ltrim(settings.SEGMENT_IMAGE_QUEUE, len(imageIDs), -1)
        # sleep for a small amount
        time.sleep(settings.SERVER_SLEEP)

if __name__ == "__main__":
    main()