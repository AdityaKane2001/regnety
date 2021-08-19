# RegNetY

<a href="https://colab.research.google.com/github/AdityaKane2001/regnety/blob/temp_readme/RegNetY_models_in_TF_2_5.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Introduction 

This repository was developed during Google Summer of Code 2021 at TensorFlow. Project report is available [here](https://adityakane2001.github.io/opensource/gsoc2021report). 
<br>

### Mentors:
- Sayak Paul ([@sayakpaul](https://github.com/sayakpaul))
- Morgan Roff ([@MorganR](https://github.com/MorganR))

This repository contains the TensorFlow 2.x implementation of RegNet-Y from the paper "[Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)".

Model stats are as follows:


<table>
    <tr>
        <td rowspan="2">Model Name</td>
        <td rowspan="2">Accuracy</td>
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
        <td>67.54%</td>
        <td>656.626</td>
        <td>591.734</td>
        <td>200MF (3.23 million)</td>
        <td> <a href="https://tfhub.dev/adityakane2001/regnety200mf_classification/1">Classifier</a>, <a href="https://tfhub.dev/adityakane2001/regnety200mf_feature_extractor/1"> Feature Extractor </a></td>
    </tr>
    <tr>
        <td>RegNetY 400MF</td>
        <td>70.19%</td>
        <td>433.874</td>
        <td>703.797</td>
        <td>400MF (4.05 million)</td>
        <td> <a href="https://tfhub.dev/adityakane2001/regnety400mf_classification/1">Classifier</a>, <a href="https://tfhub.dev/adityakane2001/regnety400mf_feature_extractor/1"> Feature Extractor </a></td>
    </tr>
    <tr>
        <td>RegNetY 600MF</td>
        <td>73.18%</td>
        <td>359.797</td>
        <td>921.56</td>
        <td>600MF (6.21 million)</td>
        <td> <a href="https://tfhub.dev/adityakane2001/regnety600mf_classification/1">Classifier</a>, <a href="https://tfhub.dev/adityakane2001/regnety600mf_feature_extractor/1"> Feature Extractor </a></td>
    </tr>
    <tr>
        <td>RegNetY 800MF</td>
        <td>73.94%</td>
        <td>306.27</td>
        <td>907.439</td>
        <td>800MF (6.5 million)</td>
        <td> <a href="https://tfhub.dev/adityakane2001/regnety800mf_classification/1">Classifier</a>, <a href="https://tfhub.dev/adityakane2001/regnety800mf_feature_extractor/1"> Feature Extractor </a></td>
    </tr>
</table>

MF signifies million floating point operations.

Reported accuracies are measured on ImageNet-1k validation dataset.

## Usage

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

One can easily load checkpoints from any epoch during training for fine-grained control. For reference, all runs are available on [WandB](). Here's an example:

```python
!pip install -q git+https://github.com/AdityaKane2001/regnety@main

from regnety.models import RegNetY

# One function call and it's done!
model = RegNetY(flops="200mf", load_checkpoint="42") #Loads weights from the 42nd epoch of training.

model.compile(...)
model.fit(...)
```


## Acknowledgement

I thank Google Summer of Code and TensorFlow for granting me this opportunity. I am grateful to my mentors Sayak Paul and Morgan Roff for their continuous guidance and encouragement. Without them this project would not have been possible. I also thank TPU Research Cloud ([TRC](https://sites.research.google/trc/)) for providing high performance TPUs for model training. Lastly, I thank TensorFlow Hub for making the models widely available.  