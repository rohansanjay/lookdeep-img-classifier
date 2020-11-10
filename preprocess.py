# this preprocess script is used to process all the three tranches and divde them into a directory 
# that is separated by sitting, standing, and lying
# in keras there is a functino that allows the model to infer the label from this subdirectory strucuture,
# (https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory)
# though it ended up being incompatible with the MobileNet v2
# we still wanted to include this script in case it may be compabible with a different model architecture

import config
import os,sys
import tensorflow as tf

# import these modules to load and organize the images from a zip file
from zipfile import ZipFile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# to confirm that GPU is being utilized to make more efficient
print("The following GPU devices are available: %s" % tf.test.gpu_device_name())

# gives user the option to preserve whatever files may have already been inside the specified sub-directory 
print(f'"Delete data in {config.PROCESSED}? y/n"')
ip = input()

if ip=='y':
    os.system(f'rm -rf {config.PROCESSED}/*')
else:
    exit(0)

# create the sub-directories of sitting, standing, and lying
os.system(f'mkdir -p {config.PROCESSED}/Sitting')
os.system(f'mkdir -p {config.PROCESSED}/Standing')
os.system(f'mkdir -p {config.PROCESSED}/Lying')

# create empty dataframe that we will add onto as we process the tranches
df = pd.DataFrame()

# loop through all tranch names
for tranch in range (1, 4): 
    labels_path = f'{config.RAW}/tranch'+str(tranch)+'_labels.csv'
    pictures_path = f'{config.RAW}/persons-posture-tranch'+str(tranch)+'.zip'

    labels = pd.read_csv(labels_path)
    # in tranch 1, it is called file_name, but in other tranches it is not. we rename for consistency
    if tranch != 1:
        labels = labels.rename(columns={'final_url' : 'file_name'})

    # we load and keep all the respective zip_file objects for retrieval later in the img_load function
    if tranch == 1:
        zip_file1 = ZipFile(pictures_path)
        file_list = [obj.filename for obj in zip_file1.infolist()]
    elif tranch == 2:
        zip_file2 = ZipFile(pictures_path)
        file_list = [obj.filename for obj in zip_file2.infolist()]
    else: 
        zip_file3 = ZipFile(pictures_path)
        file_list = [obj.filename for obj in zip_file3.infolist()]

    file_list_simple = [name.split('/')[-1] for name in file_list]

    # create a dataframe with just the file path and file name
    names = pd.DataFrame({'file_path': file_list, 'file_name': file_list_simple})

    # create a dateframe that combines the file paths and label names
    # also added a tranch column to reference later when loading images from zip
    tranch_df = pd.merge(names, labels, on = 'file_name')
    tranch_df.insert(loc=len(tranch_df.columns), column='tranch', value=tranch)
    tranch_df = tranch_df.dropna(subset=['primary_posture'])
    tranch_df = tranch_df[tranch_df['primary_posture'] != 'Unknown']

    df = df.append(tranch_df)   
    
categories = {'Sitting': 0, 'Standing': 1, 'Lying': 2}
df['label'] = df['primary_posture'].map(categories)

###################### begin preprocessing + cleaning

# drops images with more than one person and drop unknown for occluded
one_index = df[df['how_many'] != 'One'].index 
df.drop(one_index, axis = 0, inplace = True)

# drops occluded unknown
occluded_index = df[df['primary_occluded'] == 'Unknown'].index
df.drop(occluded_index, axis = 0, inplace = True)

df = df.reset_index(drop=True)

# in addition, since our focus is posture we drop the role column + exception_case
del df['exception_case']
del df['staff_patient_other']

# numerical mappings for labels
categories = {'Sitting': 0, 'Standing': 1, 'Lying': 2}
df['label'] = df['primary_posture'].map(categories)

# for resizing/preprocessing images to load
from skimage import transform

# subject to change based off model architecture
IMG_SIZE = (224, 224)

# give tranch as argument to find correct zip file 
# load image and resize based off desired shape
def img_load(img_path, tranch):
    if tranch == 1: 
        zip = zip_file1
    elif tranch == 2: 
        zip = zip_file2
    else: 
        zip = zip_file3
    img = plt.imread(zip.open(img_path))
    img = transform.resize(img, IMG_SIZE)
    return img


# based off idx in dataframe, load image
def loadimg(idx):
    row = df.iloc[idx]
    # just making sure that this specific image is indeed categorized as either Sitting, Stnanding, or Lying
    if row['primary_posture'] in ['Sitting', 'Standing', 'Lying']: pass
    else: return None, None, None
    img = img_load(row.file_path, row.tranch)
    # returning the img, the label, and the filename for reference
    return img, row['primary_posture'], row.file_name



# total is used for loading bar as we see how many of the images from all the tranches are being processed
total = len(df)
for idx, row in df.iterrows():
    print('\r               \r',end='')
    print(f'{idx}/{total}',end='') # this is just to load the number of images that have been processed
    sys.stdout.flush()    
    img, _class, filename = loadimg(idx)
    # check to see if a filename was indeed returned, and then save that image to directory
    if filename:
        plt.imsave(f'{config.PROCESSED}/{row["primary_posture"]}/{row["file_name"]}', img)
        #plt.imsave("/project/tiruvilu_529/MATH499/PROCESSED/team-3-all-tranches/" + row["file_name"])

# ds means dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(config.PROCESSED, image_size=(224, 224),
    seed=100, labels='inferred', subset='training', color_mode='rgb',validation_split=0.8)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(config.PROCESSED,  image_size=(224, 224),
    seed=100, labels='inferred',subset='validation', color_mode='rgb',validation_split=0.2)

# Enables parallel loading of images and model training
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.shuffle(700).cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)