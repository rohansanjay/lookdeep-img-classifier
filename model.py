# salloc --gres=gpu:p100 --time=0:40:00 --mem=96GB

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
train_max = 20000
test_max = 23000

print('\nloading train images')

bar = Bar('Countdown', max = split)
# bar = Bar('Countdown', max = train_max)

# for i in range(train_max):
for i in range(split):
    im = cv2.imread(df.iloc[i].file_path)
    # height, width, channels = im.shape
    # 
    # r = height / width
    # if r > 2 or r < 0.5:
    #     continue

    im = cv2.resize(im, (224, 224))
    train_ims.append(im)
    train_labels.append(df.iloc[i].primary_posture_n)
    bar.next()
    
bar.finish()


print('\nloading test images')

bar = Bar('Countdown', max = len(df) - split)
# bar = Bar('Countdown', max = test_max - train_max)

# for i in range(train_max, test_max):
for i in range(split, len(df)):
    im = cv2.imread(df.iloc[i].file_path)
    # height, width, channels = im.shape
    # 
    # r = height / width
    # if r > 2 or r < 0.5:
    #     continue
    #     
    im = cv2.resize(im, (224, 224))
    test_ims.append(im)
    test_labels.append(df.iloc[i].primary_posture_n)
    bar.next()

bar.finish()


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

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=["categorical_accuracy"])

print('\ntraining')

history = model.fit(
    train_ims, train_labels,
    epochs=50,
    batch_size=32,
    callbacks=None,
)

# image data generator 
# flow from dataframe and flow from directory
# 2000 images at a time

print('\ntesting')
results = model.evaluate(test_ims, test_labels)
print("test loss, test acc:", results)

print(history.history.keys())
print(history.history['categorical_accuracy'])
print(history.history['loss'])

plt.plot(history.history['categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show(block=True)
plt.savefig('accuracy.pdf')

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train',], loc='upper left')
plt.show(block=True)
plt.savefig('loss.pdf')
