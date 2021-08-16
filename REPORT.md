# GSoC 2021 at TensorFlow : Final Report
 
## **Implement RegNet-Y in TensorFlow 2**
<br>


## Objectives

- To create script(s) instantiating the RegNet-Y architecture.
- To train RegNet-Y  on ImageNet-1k and add to TF Hub.
<br>
<br>

## Acknowledgement

I thank Google Summer of Code and TensorFlow for granting me this opportunity. I am grateful to my mentors Sayak Paul and Morgan Roff for their continuous guidance and encouragement. Without them this project would not have been possible. I also thank TPU Research Cloud ([TRC](https://sites.research.google/trc/)) for providing high performance TPUs for model training. Lastly, I thank TensorFlow Hub for making the models widely available.  
<br>

## Overview

RegNets, and RegNetY in particular is widely used for self-supervised learning and were previously not available in TensorFlow. During GSoC 2021, I implemented and trained RegNetY on the ImageNet-1k dataset. I trained four variants of RegNetY and uploaded them to TFHub. Project links and references are mentioned at the end of this file.   
My time was largely devoted to three tasks: creating an efficient input pieline, coding the model architecture, training and fine-tuning the models. I also wrote readable and reusable code and gained insights about training on Google Cloud TPUs.   
<br>

## Important milestones

### Input pipeline

Having an efficient input pipeline is a necessity for training on powerful hardware like TPUs. Thus, I spent a lot of time creating and optimizing the input pipeline. `tf.data` is a boon for such tasks and it took away most of the painpoints. However, the challenge was to augment the data as fast as possible. It all came down to the subtleties of the preprocessing functions. Which augmentation should be aaplied first? What parts can be added to the model itself? Is I/O the most time-consuming? The answers to all of these questions came with increased performance. It is noteworthy that `tf.image` and `tfa` made it a breeze to experiment with different setups.  
<br>
### Model implementation

RegNetY is originally implemented in PyTorch by Facebook AI Research in their repository `pycls`. It was crucial that the model was as close to the original implementation. This would greatly affect the  accuracy of the models. Thus, we implemented the models with extreme care to ensure that the architecture was as close to the original one.    
<br>

### Training
Training and evaluation of the model was the most interesting part of the project. It included finding the perfect setup which suited the model and gave the best accuracy. During the training, we fixated on the hyperparameters which resulted in the best performance and started improving from there. Iteratively and progressively, we were able to get substantial gains. It was common to go back to the model implementation and input pipeline to check their correctness and improve them. We left no stone unturned.   

Validation accuracies of models:
| **Model** | **Accuracy** |
|-----------|--------------|
| 200MF     |       67.54% |
| 400MF     |       70.19% |
| 600MF     |       73.18% |
| 800MF     |       73.94% |
<br>   

## TPU Research Cloud

This project was supported by TPU Research Group (TRC). They provided high performance TPUs which sped up my training by an order of magnitude. I thank Jaeyoun Kim for giving access to Google Cloud and 
Cloud TPUs. This made the process experimentation and training substantially fluid and easy.  
<br>

## Parting thoughts

It has been an extraordinary journey of three months. During GSoC I got a chance to work with the best in the field. I express my heartfelt gratitude to my mentors for guiding me. I learnt a lot about TensorFlow and its vibrant community during this time. I intend to continue working on this project as well as contribute to TensorFlow to the best of my abilities. 
<br>

## Important links
1. [Designing Network design spaces by Radosavovic et al.](https://arxiv.org/abs/2003.13678)
2. [AdityaKane2001/regnety](https://github.com/AdityaKane2001/regnety)
3. [Google Summer of Code project page](https://summerofcode.withgoogle.com/projects/#4760303897673728)
4. [TensorFlow Hub collection page]() 
