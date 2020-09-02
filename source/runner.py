import sys
if sys.version_info[0] < 3:
    raise Exception("ERROR: python 3 or a more recent version is required.")

import os
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import random
import cv2
import numpy as np
from pathlib import Path
import shutil

input_image_width = 150
input_image_height = 150

opencv_data_dir = 'opencv/data/haarcascade/'
classifier_xml_file = opencv_data_dir + 'haarcascade_frontalface_default.xml'

if not os.path.exists(classifier_xml_file):
    raise Exception("opencv classifier file not exists.")

face_cascade_classifier = cv2.CascadeClassifier(classifier_xml_file)

def createModel(frame_w = input_image_width, frame_h = input_image_height):
    tf_model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Conv2D(100, (3,3), activation='relu', input_shape=(frame_w, frame_h, 3)),
        tensorflow.keras.layers.MaxPooling2D(2,2),
        tensorflow.keras.layers.Conv2D(100, (3,3), activation='relu'),
        tensorflow.keras.layers.MaxPooling2D(2,2),

        tensorflow.keras.layers.Flatten(),
        tensorflow.keras.layers.Dropout(0.5),
        tensorflow.keras.layers.Dense(50, activation='relu'),
        tensorflow.keras.layers.Dense(2, activation='softmax')
        ])
    tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return tf_model

def calc_rects(ImagePtr):
    image=cv2.flip(ImagePtr,1,1)
    gray = cv2.cvtColor(ImagePtr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade_classifier.detectMultiScale(gray, 1.1, 5)
    return faces

def getFaceCoord(filename):
    image = cv2.imread(filename)
    return calc_rects(image)

def getFaceImages(filename, dest, frame_w = input_image_width, frame_h = input_image_height):
    name = os.path.splitext(os.path.basename(filename))[0] 
    ext = os.path.splitext(os.path.basename(filename))[1]
    image = cv2.imread(filename)
    faces = calc_rects(image)
    i = 0
    for (x,y,w,h) in faces:
        dest_file = dest + '/' + name + '_' + str(i) + ext
        i = i + 1
        print (dest_file)
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_color = image[y:y+h, x:x+w]
        resized = cv2.resize(roi_color,(frame_w,frame_h))
        cv2.imwrite(dest_file, resized)
        #cv2.imshow('image',resized)
        #cv2.waitKey(5000)
    return

def create_data_dir(src_dir, dirname, train, test):
    test_0 = dirname + '/test/0'
    test_1 = dirname + '/test/1'
    train_0 = dirname + '/train/0'
    train_1 = dirname + '/train/1'

    dirpath = Path(dirname)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    os.makedirs(test_0)
    os.makedirs(test_1)
    os.makedirs(train_0)
    os.makedirs(train_1)

    for file in train:
        if file.startswith('Y_'):
            getFaceImages(src_dir + file, train_1)
        else:
            getFaceImages(src_dir + file, train_0)

    for file in test:
        if file.startswith('Y_'):
            getFaceImages(src_dir + file, test_1)
        else:
            getFaceImages(src_dir + file, test_0)

    return



def preprocess_data(dir_path, train_dest_dir, test_data_size = 0.2, filetypes = ['.jpg', '.jpeg', '.png', '.bmp']):
    files = []
    # search for supported extension types
    for file in os.listdir(dir_path):
        for ext in filetypes:
            if file.endswith(ext):
                files.append(file)
                break

    total_files = len(files)
    if (total_files < 10):
        raise Exception("ERROR: not enough training data.")

    #shuffle list of files and split the list in to train-test data
    random.shuffle(files)
    x_test, x_train = np.split(files, [int(total_files * test_data_size)])
    create_data_dir(dir_path, train_dest_dir, x_train, x_test)
    return

def train_model(mode, train_dirm chkpoint_path):
    #prepare training data
    training_data = ImageDataGenerator(rescale=1.0/255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    training_generator = training_data.flow_from_directory(train_dir, 
                                                           batch_size=10, 
                                                           target_size=(input_image_width, input_image_height))
    #generate checkpoints
    checkpoint = ModelCheckpoint(chkpoint_path,monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
    history = model.fit_generator(training_generator,
                                  epochs=32,
                                  validation_data=test_generator,
                                  callbacks=[checkpoint])
    model.save_weights(weights_file)

#----------------------------------------------------Start Here----------------------------------------------------------

processed_data_dir = 'tmp'
processed_train_data_dir = processed_data_dir + '/train'
processed_test_data_dir = processed_data_dir + '/test'
#preprocess_data('../data/training/', processed_data_dir)
cache_dir = 'cache/'
checkpoint_path = cache_dir + 'model-{epoch:03d}.model'

model = createModel()
test_data = ImageDataGenerator(rescale=1.0/255)
test_generator = test_data.flow_from_directory(processed_test_data_dir, 
                                                batch_size=10, 
                                                target_size=(input_image_width, input_image_height))

#see if we have trained model
latest = tensorflow.train.latest_checkpoint(cache_dir)
if latest != None:
    model.load_weights(latest)
    loss, acc = model.evaluate_generator(test_generator, verbose=2)
    print("found trained model, accuracy: {:5.2f}%".format(100*acc))
else:
    #prepare training data
    training_data = ImageDataGenerator(rescale=1.0/255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    training_generator = training_data.flow_from_directory(processed_train_data_dir, 
                                                           batch_size=10, 
                                                           target_size=(input_image_width, input_image_height))
    #generate checkpoints
    checkpoint = ModelCheckpoint(checkpoint_path,monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
    history = model.fit_generator(training_generator,
                                  epochs=32,
                                  validation_data=test_generator,
                                  callbacks=[checkpoint])
    loss, acc = model.evaluate_generator(test_generator, verbose=2)
    print("trained model accuracy: {:5.2f}%".format(100*acc))
    model.save_weights(weights_file)