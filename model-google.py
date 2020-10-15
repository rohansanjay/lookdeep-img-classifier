# salloc --gres=gpu:p100 --time=0:40:00 --mem=96GB

# For running inference on the TF-Hub module.
import tensorflow as tf

import tensorflow_hub as hub
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split

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

all_ims = []
all_labels = []

print('\nloading images')

bar = Bar('Countdown', max = len(df))

for i in range(len(df)):
    im = cv2.imread(df.iloc[i].file_path)
    im = cv2.resize(im, (224, 224))
    all_ims.append(im)
    all_labels.append(df.iloc[i].primary_posture_n)
    bar.next()
    
bar.finish()

print('\nsplitting images')
train_ims, test_ims, train_labels, test_labels = train_test_split(all_ims, all_labels, test_size=.20)

print('\nfinishing loading images')
print('len of train images:', len(train_ims))
print('len of train images:', len(test_ims))

train_ims = np.array( train_ims ) / 255
train_labels = np.array( train_labels )
test_ims = np.array( test_ims ) / 255
test_labels = np.array( test_labels )

train_labels = tf.keras.utils.to_categorical( train_labels , num_classes=3 )
test_labels = tf.keras.utils.to_categorical( test_labels , num_classes=3 )

print('\ncreated np arrays')


print('\loading model')
    
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

initial_epochs = 10

print('\ntraining')

history = model.fit(
    train_ims, train_labels,
    epochs=initial_epochs,
    batch_size=10,
    validation_split=0.3
)

base_model.trainable = True
# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

print('\nfine tuning')

history_fine = model.fit(
    train_dataset,
    epochs=total_epochs,
    batch_size=10,
    initial_epoch=history.epoch[-1],
    validation_split=0.3
)

print('\ntesting')
results = model.evaluate(test_ims, test_labels)
print("test loss, test acc:", results)
print('Test accuracy :', accuracy)