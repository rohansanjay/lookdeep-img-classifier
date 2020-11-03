# LookDeep

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
