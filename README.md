# RegNetY

<a href="https://colab.research.google.com/github/AdityaKane2001/regnety/blob/main/RegNetY_models_in_TF_2_5.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

TensorFlow 2.x implementation of RegNet-Y from the paper "[Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)". 

<img src="https://raw.githubusercontent.com/AdityaKane2001/archive/main/regnetys.png?token=APLNNKW6U75LR3OZUUGE6DTBFXFZC"> 

## About this repository

This implementation of RegNet-Y is my project for Google Summer of Code 2021 with TensorFlow<sup>[1]</sup>. 


### Salient features:

- Four variants of RegNet-Y implemented - 200MF, 400MF, 600MF, 800MF.
- TPU compatible: All models are trainable on TPUs out-of-the box.
- Inference speed: All models boast blazing fast inference speeds.
- Fine-tuning: Models can be fine-tuned using gradual unfreezing and offer more granularity by using the released checkpoints.
- Modular and reusable: Every part of the architecture is implemented by subclassing `tf.keras.Model`. This makes the individual parts of the models highly reusable. 
- Trained on ImageNet-1k: Models in this repository are trained on ImageNet-1k and can be used for inference out-of-the box. These pretrained models are available on [TFHub](https://tfhub.dev/adityakane2001/collections/regnety/1) and thus can be easily used with `hub.KerasLayer` and `hub.load`.

## Table of contents

1. [Introduction to RegNets](https://github.com/AdityaKane2001/regnety#introduction-to-regnets)
2. [Model statistics](https://github.com/AdityaKane2001/regnety#model-statistics)
3. [Usage](https://github.com/AdityaKane2001/regnety#usage)
4. [Known caveats](https://github.com/AdityaKane2001/regnety#known-caveats)
5. [What's next?](https://github.com/AdityaKane2001/regnety#whats-next)
6. [Acknowledgement](https://github.com/AdityaKane2001/regnety#acknowledgemnt)

## Introduction to RegNets

### About the paper

The paper "[Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)" aims to systematically deduce the best model population starting from a model space that has no constraints. The paper also aims to find a best model population, as opposed to finding a singular best model as in works like [NASNet](https://arxiv.org/abs/1707.07012). 

The outcome of this experiment is a family of networks which comprises of models with various computational costs. The user can choose a particular architecture based upon the need. 

### About the models

Every model of the RegNet family consists of four Stages. Each Stage consists of numerous Blocks. The architecture of this Block is fixed, and three major variants of this Block are available: X Block, Y Block, Z Block<sup>[2]</sup>. Other variants can be seen in the paper, and the authors state that the model deduction method is robust and RegNets generalize well to these block types.    
The number of Blocks and their channel width in each Stage is determined by a simple quantization rule put forth in the paper. More on that in this [blog](https://medium.com/visionwizard/simple-powerful-and-fast-regnet-architecture-from-facebook-ai-research-6bbc8818fb44).
RegNets have been the network of choice for self supervised methods like [SEER](https://arxiv.org/pdf/2103.01988.pdf) due to their remarkable scaling abilities. 

RegNet architecture:   
<img src="https://raw.githubusercontent.com/AdityaKane2001/archive/main/regnety_architecture.png?token=APLNNKU47VKUQENVQPB5DB3BFSGJ2" width="588" height="250" > 

Y Block:   
<img src="https://raw.githubusercontent.com/AdityaKane2001/archive/main/YBlock.jpg?token=APLNNKTOBNXVBHPYTBICS3TBFSK7U" width="588" height="250" />



## Model statistics

<table>
    <tr>
        <td rowspan="2">Model Name</td>
        <td rowspan="2">Accuracy (Ours/ Paper) </td>
        <td colspan="2">Inference speed (images per second)</td>
        <td rowspan="2">FLOPs (Number of parameters)</td>
        <td rowspan="2">Link</td>
    </tr>
    <tr>
        <td>K80</td>
        <td>V100</td>
    </tr>
    <tr>
        <td>RegNetY 200MF</td>
        <td>67.54% / 70.3%</td>
        <td>656.626</td>
        <td>591.734</td>
        <td>200MF (3.23 million)</td>
        <td> <a href="https://tfhub.dev/adityakane2001/regnety200mf_classification/1">Classifier</a>, <a href="https://tfhub.dev/adityakane2001/regnety200mf_feature_extractor/1"> Feature Extractor </a></td>
    </tr>
    <tr>
        <td>RegNetY 400MF</td>
        <td>70.19% / 74.1%</td>
        <td>433.874</td>
        <td>703.797</td>
        <td>400MF (4.05 million)</td>
        <td> <a href="https://tfhub.dev/adityakane2001/regnety400mf_classification/1">Classifier</a>, <a href="https://tfhub.dev/adityakane2001/regnety400mf_feature_extractor/1"> Feature Extractor </a></td>
    </tr>
    <tr>
        <td>RegNetY 600MF</td>
        <td>73.18% / 75.5%</td>
        <td>359.797</td>
        <td>921.56</td>
        <td>600MF (6.21 million)</td>
        <td> <a href="https://tfhub.dev/adityakane2001/regnety600mf_classification/1">Classifier</a>, <a href="https://tfhub.dev/adityakane2001/regnety600mf_feature_extractor/1"> Feature Extractor </a></td>
    </tr>
    <tr>
        <td>RegNetY 800MF</td>
        <td>73.94% / 76.3%</td>
        <td>306.27</td>
        <td>907.439</td>
        <td>800MF (6.5 million)</td>
        <td> <a href="https://tfhub.dev/adityakane2001/regnety800mf_classification/1">Classifier</a>, <a href="https://tfhub.dev/adityakane2001/regnety800mf_feature_extractor/1"> Feature Extractor </a></td>
    </tr>
</table>

MF signifies million floating point operations. Reported accuracies are measured on ImageNet-1k validation dataset.

## Usage

<a href="https://colab.research.google.com/github/AdityaKane2001/regnety/blob/main/RegNetY_models_in_TF_2_5.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### 1. Using TFHub

This is the preferred way to use the models developed in this repository. Pretrained models are uploaded to [TFHub]().  See the [colab](https://colab.research.google.com/github/AdityaKane2001/regnety/blob/temp_readme/RegNetY_models_in_TF_2_5.ipynb) for detailed explaination and demo. Basic usage:

```python
model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/adityakane2001/regnety200mf_feature_extractor/1", training=True), # Can be False
    tf.keras.layers.GlobalAveragePooling(),
    tf.keras.layers.Dense(num_classes)
])

model.compile(...)
model.fit(...)
```

### 2. Reloading checkpoints from training

One can easily load checkpoints for fine-grained control.  Here's an example:

```python
!pip install -q git+https://github.com/AdityaKane2001/regnety@main

from regnety.models import RegNetY

# One function call and it's done!
model = RegNetY(flops="200mf", load_checkpoint=True) #Loads pretrained checkpoint

model.compile(...)
model.fit(...)
```


## Known caveats

- The models achieve lower accuracies than mentioned in the paper. After substantial scrutiny we conclude that this is due to the difference in internal working of TensorFlow and PyTorch. We are still trying to increase the accuracy of these models using different training methods.
- These models cannot be trained on a CPU. One can run inference on the models by strictly using batch_size=1. The models function as expected on GPUs and TPUs. This is because grouped convolutions are not supported for training by TensorFlow (as of 2.6.0).  

## What's next?

This project is being actively worked on and developed. If you have any suggestions, feel free to open an issue or start a PR.  Following is the list of things to be done in near future. 


- [ ] Improve accuracy of the models.
- [ ] Convert existing models to TFLite.
- [ ] Implement and train more variants of RegNetY.
- [ ] Training models using noisy student method.
- [ ] Implement and train RegNet-{X, Z}


## Acknowledgement

This repository is my project for Google Summer of Code 2021 at TensorFlow. Project report is available [here](https://adityakane2001.github.io/opensource/gsoc2021report). 
<br>

### Mentors:
- Sayak Paul ([@sayakpaul](https://github.com/sayakpaul))
- Morgan Roff ([@MorganR](https://github.com/MorganR))

I thank Google Summer of Code and TensorFlow for granting me this opportunity. I am grateful to my mentors Sayak Paul and Morgan Roff for their continuous guidance and encouragement. Without them this project would not have been possible. I also thank TPU Research Cloud ([TRC](https://sites.research.google/trc/)) for providing high performance TPUs for model training. Lastly, I thank TensorFlow Hub for making the models widely available.  

## References

[1] Project report [here](https://adityakane2001.github.io/opensource/gsoc2021report).    
[2] Z Block is proposed in the paper "[Fast and Accurate Model Scaling](https://arxiv.org/abs/2103.06877)".