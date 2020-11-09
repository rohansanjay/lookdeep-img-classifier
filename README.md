# LookDeep

### Model Used
MobileNetV2 is light and fast -- it mainly uses depthwise convolution for an architecture that uses fewer parameters and less computational complexity while still achieving good results. When we first started working on this model (before we got access to USC's computing cluster), we were strongly limited by memory and latency limitations, and this model was a great option for getting into image processing and still achieving a high accuracy without an heavy and complicated model architecture. 

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

### Downloading the image data

The full data set can be downloaded using the following script:
``` bash
python download_data.py
```
This downloads all 3 tranches of images and their corresponding labels. 
  
3 folders with the images are created: home/, tranch2/ and tranch3/.  
A csv file with all image file paths and corresponding labels is created: tranch_master.csv. 


### Model Overview
The model is trained using the following command: 
``` bash
python model.py
```  
The model uses the MobileNetV2 base model from Keras applications. It was trained across all three tranches (~36,000 images with known labels) using the Imagenet weights. After the base model, a pooling layer and three dense layers were added, with 100 layers frozen for the purposes of fine tuning. The model reached an accuracy of 87% across all three tranches. No preprocessing of the images was done when training this model. 

  
### Data Preprocessing
This script contains an alternate method of loading and preprocessing images designed to increase efficiency. It implements basic preprocessing that excludes the following images with the intent of improving accuracy:

    Images containing more than one person (an "easier" subset for the model to work on)
    Images that are marked as occluded = unknown (these images tend to be confusing to the model)
    Images that have a blank "primary_posture" label

We load the images into a directory with three subfolders ("Sitting," "Standing," "Lying"), and then convert those images into a tf.data.Dataset object that infers labels from the directory structure (hence the subfolders). This creates a training set that allows us to use a prefetch function to load images in batches and push them to the model while simultaneously loading then next batch of images. This speeds up the model training process significantly. 

However, we were unable to get this to work with our MobileNetV2 model, since the model seems to require a different input. However, this preprocessing script could be implemented with other models, and serves as a useful guide to preprocessing images and loading them into a model efficiently. 


### Data Augmentation
- Rohan

### Data Visualization
- Kartik /Rohan/ Chris

![individual_tables](https://user-images.githubusercontent.com/31398970/98498936-03da1b00-21fd-11eb-98c8-2f47413134ca.png)


### Conclusion
We were able to reach a 90% accuracy level even without preprocessing the data. If we were to continue this project, we would have liked to continue to build on our model accuracy and explore how preprocessing of data could improve our final results. Overall, our group is extremely happy and proud about our progress with this project. We have developed a reliable machine learning model with little to no prior exposure. We hope you find relevant information in this Github Repository and please do not hesitate to contact us with any questions or suggestions.
