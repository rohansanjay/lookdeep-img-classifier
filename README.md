# LookDeep
An image classification model to predict patient posture in hospital rooms for the Bay Area based startup [LookDeep Health](https://www.lookdeep.health/).

### Table of Contents
- [How to run code on cluster & set up environment](#How-to-run-code-on–cluster–&–set–up–environment)
- [Downloading the image data and Data Preprocessing](#Downloading-the-image-data-and-Data-Preprocessing)
- [Data Visualization](#Data-Visualization)
- [Data Augmentation](#Data-Augmentation)
- [Border Replication](#Border-Replication)
- [Model Overview](#Model-Overview)
- [Conclusion](#Conclusion)


### How to run code on cluster & set up environment

Training the model on USC's High Performance Computing Cluster enables faster computations and GPU usage. The following commands login to a compute node on the cluster with the correct specifications to train the model over the full dataset:

``` bash
salloc --gres=gpu:p100 --time=0:40:00 --mem=96GB

module load gcc
module load cuda/10.1.243
module load cudnn/7.6.5.32-10.1
module load python
```
Note: these commands can be added to a job.sh file and called in one step from the command line but sometimes, the GPU may not load properly. Running each command one at a time ensures that this issue will not arise. 
Once these commands are entered, the compute node will be active for the period of time specified. Python scripts can then be run as follows: 
``` bash
python yourScript.py
```  

### Downloading the image data and Data Preprocessing

The full data set can be downloaded using the following script:
``` bash
python download_data.py
```
This downloads all 3 tranches of images and their corresponding labels. 
  
3 folders with the images are created: home/, tranch2/ and tranch3/.  
A csv file with all image file paths and corresponding labels is created: tranch_master.csv. 

This script contains an alternate method of loading and preprocessing images designed to increase efficiency. It implements basic preprocessing that excludes the following images with the intent of improving accuracy:

    1. Images containing more than one person (an "easier" subset for the model to work on)
    2. Images that are marked as occluded = unknown (these images tend to be confusing to the model)
    3. Images that have a blank "primary_posture" label

We load the images into a directory with three subfolders ("Sitting," "Standing," "Lying"), and then convert those images into a tf.data.Dataset object that infers labels from the directory structure (hence the subfolders). This creates a training set that allows us to use a prefetch function to load images in batches and push them to the model while simultaneously loading then next batch of images. This speeds up the model training process significantly. 

However, we were unable to get this to work with our MobileNetV2 model, since the model seems to require a different input. However, this preprocessing script could be implemented with other models, and serves as a useful guide to preprocessing images and loading them into a model efficiently. 


### Data Visualization

The table below gives a better understanding of our dataset based on various measures and parameters of exploration and their distribution based on the counts of their labels, namely number of person in the image (person_count), occlusion (primary_occluded), person's posture (primary_posture) and type of person (person_type). The vertical axes correspond to the number (in thousands) to each label in the horizontal axes. This represents some of the key features of the dataset and we explored certain scopes of inquiry within this label-based framework.

![individual_tables2](https://user-images.githubusercontent.com/31398970/98973874-23a56380-24c9-11eb-9379-16288a40634e.png)

Given the scope of our analysis centering around investigating around sorting data as "sitting", "standing" or "lying", the graphs below illustrate the distribution of given dataset when comparing the person_type (red), primary_occluded (green) and person_count (blue) against primary_posture. More information and visualization data can be found in the datasets folder.

![variables_against_posture_normalized](https://user-images.githubusercontent.com/31398970/98973917-3029bc00-24c9-11eb-8d2a-d865d23b8af9.png)

The same data above, with the vertical axes standardized

![variables_against_posture2](https://user-images.githubusercontent.com/31398970/98973887-27d18100-24c9-11eb-98c8-d0d7bab14792.png)

These data tables give a better understanding of the omissions made in the pre-processing stage, and ultimately we focused our model around the median/centroid of the training dataset - a further scope to improve our model's accuracy would have to involve exploring the exclusions and training data to better account for them as well.


### Data Augmentation
Data Augmentation preprocesses as normal with the only difference being that every image labeled as lying in every tranch is replicated through copying existing images already present. That alongside including CV2 instead of other image manipulation tools changed augmented accuracy from high 70s to high 80s. Although there is an issue with downloading the model to train on the augmented data set, the preprocessing element is functional. It helps because of the imbalanced data set. The sitting class is over represented and the lying class is inherently difficult for the model to train on. We discovered that the model was struggling most with classifying lying data, so for that reason we decided to augment that subset of the data.


### Border Replication
The purpose of border replication is to avoid the skew that occurs due to extreme ratios when resizing. It blues the edges instead to maintain a humanoid figure that is not disproportionately skewed.


### Model Overview
The model is trained using the following command: 
``` bash
python model.py
```  
The model uses the MobileNetV2 base model from Keras applications. It was trained across all three tranches (~36,000 images with known labels) using the Imagenet weights. After the base model, a pooling layer and three dense layers were added, with 100 layers frozen for the purposes of fine tuning. The model reached an accuracy of 87% across all three tranches. No preprocessing of the images was done when training this model. 

MobileNetV2 is light and fast -- it mainly uses depthwise convolution for an architecture that uses fewer parameters and less computational complexity while still achieving good results. When we first started working on this model (before we got access to USC's computing cluster), we were strongly limited by memory and latency limitations, and this model was a great option for getting into image processing and still achieving a high accuracy without an heavy and complicated model architecture. 


### Conclusion
We were able to reach a ~90% accuracy level even without preprocessing the data. If we were to continue this project, we would like to continue to build on our model accuracy and explore how preprocessing of data could improve our final results. Furthermore, we would like to begin working on classifying video footage. Overall, our group is extremely happy and proud about our progress with this project. We have developed a reliable machine learning model with little to no prior exposure. We hope you find relevant information in this Github Repository and please do not hesitate to contact us with any questions or suggestions.
