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


history=np.load('my_history.npy',allow_pickle='TRUE').item()

loss_train = history['acc']
loss_val = history['val_acc']
epochs = range(1,31)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()