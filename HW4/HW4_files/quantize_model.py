import argparse
import numpy as np
import torch
import random

parser = argparse.ArgumentParser(description='ECE361E HW4 Quantization')
# TODO add argument for ONNX model
args = parser.parse_args()

# Each experiment you will do will have slightly different results due to the randomness
# of 1. the initialization value for the weights of the model, 2. sampling batches of training data
# 3. numerical algorithms for computation (in CUDA.) In order to have reproducible results,
# we have fixed a random seed to a specific value such that we "control" the randomness.
random_seed = 1
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g = torch.Generator()
g.manual_seed(random_seed) # for data loader shuffling

# TODO add CIFAR10 train dataset

# TODO add CIFAR10 Calibration Data Reader

# TODO Preprocess model for quantization

# TODO Use 1,000 images from the CIFAR10 Calibration Data Reader

# TODO Perform static quantization

