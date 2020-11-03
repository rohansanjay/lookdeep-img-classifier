# LookDeep

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
python3 yourScript.py
```



### Downloading the image data
   - Rohan
USCoders Final Deliverable
- How to run code on cluster & set up environment
   - Rohan
- Downloading the image data
   - Rohan
- Model overview & assessment
   - Derek: Why set up this way, freezing layers, etc
- Object detection 
   - Kartik/Rohan ask other groups about it
- Conclusion
   - Kelly & Carolina

To do: 
- Kartik to talk to other groups -> cluster stuff
- Derek to overview model & add pictures
- Rohan to talk to groups & address parts above
- Carolina & Kelly to make sure everything looks cute & nice


Model Overview

The model uses the MobileNetV2 base model from Keras applications. It was trained across all three tranches using the Imagenet weights. After the base model, a pooling layer and three dense layers were added, with 100 layers frozen for the purposes of fine tuning. The model reached an accuracy of 87% across all three tranches.
