# Visual Saliency Prediction Using a Mixture of Deep Neural Networks
This folder provides reference code for the paper "Visual Saliency Prediction Using a Mixture of Deep Neural Networks".

## Installation

Before running the code you will need to download the following pre-trained networks into the models folder:

* Weights VGG-16: [vgg16_weights.h5](https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing)
* Weights ML-Net: [mlnet_salicon.h5](https://drive.google.com/file/d/0B2HAjarkcu-wclBpLVFnUjBoYTg/view?usp=sharing)

Install the required libraries using pip:
```
pip install -r requirements.txt
```

You will also need to download the [CAT2000 dataset](http://saliency.mit.edu/results_cat2000.html)

## Usage

First set the path to the dataset in the param.py file (parameter BASEPATH). Then run the code using:
```
python train.py
```

To test the model run
```
python test.py
```

We also include the code to fine-tune the ML-net model. This code can be run using the "baseline_train.py" file.

