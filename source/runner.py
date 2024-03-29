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
import matplotlib.pyplot as plt

input_image_width = 150
input_image_height = 150

curr_dirname = os.path.dirname(__file__)
opencv_data_dir = curr_dirname + '/opencv/data/haarcascade/'
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
    #gray = cv2.cvtColor(ImagePtr, cv2.COLOR_BGR2GRAY)
    gray = ImagePtr
    faces = face_cascade_classifier.detectMultiScale(gray, 1.1, 5)
    return faces

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
            shutil.copy2(src_dir + file, train_1)
            #getFaceImages(src_dir + file, train_1)
        else:
            shutil.copy2(src_dir + file, train_0)
            #getFaceImages(src_dir + file, train_0)

    for file in test:
        if file.startswith('Y_'):
            shutil.copy2(src_dir + file, test_1)
            #getFaceImages(src_dir + file, test_1)
        else:
            shutil.copy2(src_dir + file, test_0)
            #getFaceImages(src_dir + file, test_0)

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


def train_model(model, train_dir, cache_dir, test_generator):
    #prepare training data
    model = createModel()
    checkpoint_path = cache_dir + 'model-{epoch:03d}.model'
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
    cp_callback = ModelCheckpoint(checkpoint_path,
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')
    history = model.fit(training_generator,
              epochs=30,
              validation_data=test_generator,
              callbacks=[cp_callback])
    np.save('my_history.npy',history.history)
    
def get_best_model(test_generator, cache_dir, scan = False):
    if not os.path.exists(cache_dir):
        return None
    #get the latest model
    if not scan:
        subdirs = os.listdir(cache_dir)
        if len(subdirs) > 0:
            subdirs.sort(key = (lambda x : x[6:9]), reverse = False)
            return tensorflow.keras.models.load_model(cache_dir + subdirs[0])
        else:
            return None
    else:
        if not os.path.isdir(cache_dir):
            return None
        #scan thorugh every model and get best.
        cost = 0.0
        res_model = None
        for entry in os.listdir(cache_dir):
            model_path = os.path.join(cache_dir,entry)
            if os.path.isdir(model_path):
                model = tensorflow.keras.models.load_model(model_path)
                #model.summary()
                loss, acc = model.evaluate(test_generator, verbose=2)
                if (acc + (1-loss) > cost):
                    cost = acc + (1-loss)
                    res_model = model
        return res_model

#----------------------------------------------------Start Here----------------------------------------------------------

processed_data_dir = curr_dirname + '/tmp'
processed_train_data_dir = processed_data_dir + '/train'
processed_test_data_dir = processed_data_dir + '/test'
cache_dir = curr_dirname + '/cache/'

#uncomment this to recreate cropped images
#preprocess_data('../data/training/', processed_data_dir)

test_data = ImageDataGenerator(rescale=1.0/255)
test_generator = test_data.flow_from_directory(processed_test_data_dir, 
                                                batch_size=10, 
                                                target_size=(input_image_width, input_image_height))

#see if we have trained model or need to train it
#model = get_best_model(test_generator, cache_dir)
model = None

if model == None:
    print ('training model...')
    train_model(model, processed_train_data_dir, cache_dir, test_generator)
    model = get_best_model(test_generator, cache_dir)

if model == None:
    raise Exception("ERROR: failed to create the model.")

loss, acc = model.evaluate(test_generator, verbose=2)
print("model accuracy: {:5.2f}%".format(100*acc))


# now start the webcam face-mask detection

# we use this to scale down the image to detect faces faster and
# then expand back the detected faces.
scale = 4
face_labels = {0: 'no_mask', 1: ' '}
face_color = {0: (0,0,255), 1: (0,255,0)}

webcam = cv2.VideoCapture(0)
while True:
    ( _, frame) = webcam.read()
    frame = cv2.flip(frame, 1, 1)
    image = cv2.resize(frame, (frame.shape[1] // scale, frame.shape[0] // scale))
    faces = face_cascade_classifier.detectMultiScale(image)
    for face in faces:
        # we expand-back the faces
        (x, y, w, h) = [v * scale for v in face]
        face_im = frame[y:y+h, x:x+w]
        face_resized=cv2.resize(face_im,(input_image_width, input_image_height))
        face_normalized=face_resized/255.0
        face_reshaped=np.reshape(face_normalized,(1, input_image_width, input_image_height, 3))
        face_reshaped = np.vstack([face_reshaped])
        result=model.predict(face_reshaped)
        #print(result)        
        result=np.argmax(result,axis=1)[0]
        cv2.rectangle(frame,(x,y),(x+w,y+h),face_color[result],2)
        cv2.rectangle(frame,(x,y-20),(x+w,y),face_color[result],-1)
        cv2.putText(frame, face_labels[result],(x, y-10),cv2.FONT_HERSHEY_PLAIN,0.8,(255,255,255),2)

    cv2.imshow('Press ESC to exit...', frame)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()