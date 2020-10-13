"""MobileNetV2_10_06 (Version 2) + OD

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nmmLFuKdKpbw0zMaOuyr4XykIyLOXk6S
"""


# For running inference on the TF-Hub module.
import tensorflow as tf

import tensorflow_hub as hub
from tensorflow.keras import datasets, layers, models

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring the inference time.
import time
from progress.bar import Bar

# Print Tensorflow version
print(tf.__version__)

# Image imports
from zipfile import ZipFile
import matplotlib.pyplot as plt
import pandas as pd

# Check available GPU devices.
print("The following GPU devices are available: %s" % tf.test.gpu_device_name())

df = pd.read_csv('tranch_master.csv')
df = df.dropna(subset=['primary_posture'])
df = df.reset_index(drop=True)
df = df[df['primary_posture'] != 'Unknown']
categories = {'Sitting': 0, 'Standing': 1, 'Lying': 2}
df['primary_posture_n'] = df['primary_posture'].map(categories)

"""Model"""

import cv2
from PIL import Image

train_ims = []
train_labels = []
test_ims = []
test_labels = []
split = int(len(df) * 0.8)

print('\nloading train images')
bar = Bar('Countdown', max = split)
for i in range(split):
    im = cv2.imread(df.iloc[i].file_path)
    im = cv2.resize(im, (224, 224))
    train_ims.append(im)
    train_labels.append(df.iloc[i].primary_posture_n)
    bar.next()
    
bar.finish()

print('\nloading test images')
bar = Bar('Countdown', max = len(df) - split)
for i in range(split, len(df)):
    im = cv2.imread(df.iloc[i].file_path)
    im = cv2.resize(im, (224, 224))
    test_ims.append(im)
    test_labels.append(df.iloc[i].primary_posture_n)
    bar.next()

bar.finish()

print('\nfinishing loading images')

train_ims = np.array( train_ims ) / 255
train_labels = np.array( train_labels )
test_ims = np.array( test_ims ) / 255
test_labels = np.array( test_labels )

train_labels = tf.keras.utils.to_categorical( train_labels , num_classes=3 )
test_labels = tf.keras.utils.to_categorical( test_labels , num_classes=3 )

print('\ncreated np arrays')

from keras.applications import MobileNetV2
from keras import layers, optimizers

print('\loading model')

base_model = MobileNetV2(weights='imagenet',include_top=False, input_shape=(224, 224, 3)) # 224, 224
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024,activation='relu')(x) 
x = layers.Dense(1024,activation='relu')(x) 
x = layers.Dense(512,activation='relu')(x) 
preds = layers.Dense(3, activation='softmax')(x) 

model = models.Model(inputs=base_model.input,outputs=preds)

for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

print('\ntraining')

model.fit(
    train_ims, train_labels,
    epochs=10,
    batch_size=10,
    callbacks=None,
)

# image data generator 
# flow from dataframe and flow from directory
# 2000 images at a time

print('\ntesting')
model.evaluate(test_ims, test_labels)

"""- what size to resize to
- training over all 3 tranches
- min/max pixel size
- preprocessed/untouched test sets
- model architecture
"""

