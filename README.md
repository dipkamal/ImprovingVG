# Towards improving saliency map interpretability using feature map smoothing

Repository for the paper "Towards stable and sparse saliency maps via feature map smoothing," submitted on Transactions on Machine Learning Research (TMLR).

In this work, we examine how adversarial training, known to enhance model robustness, affects explanation quality, and propose a lightweight feature-map smoothing mechanism that can be integrated during adversarial training. Through empirical studies on FMNIST, CIFAR-10, and ImageNette, we find that local smoothing filters (e.g., mean, median) improve the stability and human-perceived clarity of saliency maps, while retaining the sparsity benefits of adversarial training.


## Requirement
- kornia (for applying filters to feature-maps)
- cleverhans (for adversarial training)
- captum (for saliency maps)
- quantus (for computing several metrics for saliency maps)
- ImageNette dataset ([download link](https://github.com/fastai/imagenette). )


The directory consists of the following file and folders:

    FMNIST: This folder consists of a notebook that demonstrates training of several robust and non-robust FMNIST models. 
    CIFAR-10: This folder consists of python files for training robust and non-robust CIFAR-10 models.
    ImageNette: This folder consists of python files for training robust and non-robust ImageNette models.
    metrics.py: This program file consists of python functions for evaluating saliency maps.
