# For running inference on the TF-Hub module.
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from keras.applications import MobileNetV2
from keras import layers, optimizers

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO
from zipfile import ZipFile
import pandas as pd

# For image manipulation.
import numpy as np
from PIL import Image
import cv2

# For measuring the inference time.
import time
from progress.bar import Bar

#BORDER_REPLICATE blurs background of image, creates padding, and makes image square. 
def replicate_border(im, height, width):
    if height != width: 
        padding = abs(int((height - width) / 2))
        if height > width: 
            im = cv2.copyMakeBorder(im, 0, 0, padding, padding, cv2.BORDER_REPLICATE)
        elif width < height: 
            im = cv2.copyMakeBorder(im, padding, padding, 0, 0, cv2.BORDER_REPLICATE)
    return im
    

# Print Tensorflow version
print(tf.__version__)

# Check available GPU devices.
print("The following GPU devices are available: %s" % tf.test.gpu_device_name())

"""Load in df of images (3 traches) created by download_data.py"""

df = pd.read_csv('tranch_master.csv')
df = df.dropna(subset=['primary_posture'])
df = df.reset_index(drop=True)
df = df[df['primary_posture'] != 'Unknown']

# numerical mappings for labels
categories = {'Sitting': 0, 'Standing': 1, 'Lying': 2}
df['primary_posture_n'] = df['primary_posture'].map(categories)


"""Model"""


all_ims = []
all_labels = []

print('\nloading images')

# progress bar for loading images
bar = Bar('Countdown', max = len(df))


# take all images and resize it to create border
for i in range(len(df)):
    im = cv2.imread(df.iloc[i].file_path)
    # created to determine what the height/width of the image is
    height, width, _ = im.shape
    im = replicate_border(im, height, width)
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

# create train and test matrixes
train_ims = np.array( train_ims ) / 255
train_labels = np.array( train_labels )
test_ims = np.array( test_ims ) / 255
test_labels = np.array( test_labels )

train_labels = tf.keras.utils.to_categorical( train_labels , num_classes=3 )
test_labels = tf.keras.utils.to_categorical( test_labels , num_classes=3 )


print('\loading model')
base_model = MobileNetV2(weights='imagenet',include_top=False, input_shape=(224, 224, 3)) # 224, 224
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024,activation='relu')(x) 
x = layers.Dense(1024,activation='relu')(x) 
x = layers.Dense(512,activation='relu')(x) 
preds = layers.Dense(3, activation='softmax')(x) 
model = models.Model(inputs=base_model.input,outputs=preds)


# fine tuning on top of base MobileNet model
fine_tune_at = 100
for layer in model.layers[:fine_tune_at]:
    layer.trainable=False
for layer in model.layers[fine_tune_at:]:
    layer.trainable=True


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=["accuracy"])

print('\ntraining')
print()

history = model.fit(
    train_ims, train_labels,
    epochs=20,
    batch_size=18,
    callbacks=None,
    validation_split=0.3
)


print('\ntesting')
results = model.evaluate(test_ims, test_labels)

print("test loss, test acc:", results)
print()


# confusion matrix
predictions = model.predict(test_ims)
pl = [np.argmax(x) for x in predictions]
tl = [np.argmax(x) for x in test_labels]

print('prediction shape', pl)
print('test labels', tl)
print(tf.math.confusion_matrix(tl, pl))


# accuray graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.pdf')

plt.close()

# loss graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.pdf')