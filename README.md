# On the connection between model robustness and saliency map interpretability

Repository for the paper "On the connection between model robustness and saliency map interpretability," submitted at The 18th European Conference on Computer Vision ECCV 2024.

This paper provides our implementation of building several robust models for improving the quality of saliency maps using Vanilla Gradient method. You will need to download ImageNette dataset from [here](https://github.com/fastai/imagenette). 

## Requirement
- kornia (for applying filters to feature-maps)
- cleverhans (for adversarial training)
- captum (for saliency maps)
- quantus (for computing several metrics for saliency maps)


The directory consists of the following file and folders:

    FMNIST: This folder consists of a notebook that demonstrates training of several robust and non-robust FMNIST models. 
    CIFAR-10: This folder consists of python files for training robust and non-robust CIFAR-10 models.
    ImageNette: This folder consists of python files for training robust and non-robust ImageNette models.
    metrics.py: This program file consists of python functions for evaluating saliency maps.
