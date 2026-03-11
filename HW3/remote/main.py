from vgg11 import VGG11
from vgg16 import VGG16
from mobilenet import MobileNetv1
import torch
import torch.nn as nn
import time
from thop import profile
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import argparse
import random
import numpy as np

# Argument parser
parser = argparse.ArgumentParser(description='ECE361E HW3 - Starter PyTorch code')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
# TODO: Add argument for choosing the model
parser.add_argument('--model', type=str, default='vgg16', help='Model to use')
args = parser.parse_args()

# Always make assignments to local variables from your args at the beginning of your code for better
# control and adaptability
num_epochs = args.epochs
batch_size = args.batch_size
model = args.model

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

# CIFAR10 Dataset (Images and Labels)
train_dataset = dsets.CIFAR10(root='data', train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]), download=True)

test_dataset = dsets.CIFAR10(root='data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]))

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,generator=g)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# TODO: Get VGG11 model
if model == "vgg11":
    model = VGG11()
elif model == "vgg16":
    model = VGG16()
elif model == "mobilenet":
    model = MobileNetv1()
else:
    raise ValueError('Invalid model choice. Please choose from "vgg11", "vgg16", or "mobilenet".')

# TODO: Put the model on the GPU
model.to(torch.device('cuda'))

# Minimal stats: trainable params, FLOPs, timing, GPU memory
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# thop expects model in eval mode for FLOPs counting
model.eval()
dummy_input = torch.randn(1, 3, 32, 32).to(torch.device('cuda'))
macs, thop_params = profile(model, inputs=(dummy_input,), verbose=False)
flops = float(macs) * 2.0

start_time = time.time()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for epoch in range(num_epochs):
    # Training phase
    train_correct = 0
    train_total = 0
    train_loss = 0
    # Sets the model in training mode.
    model = model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        # TODO: Put the images and labels on the GPU
        images = images.to(torch.device('cuda'))
        labels = labels.to(torch.device('cuda'))

        # Sets the gradients to zero
        optimizer.zero_grad()
        # The actual inference
        outputs = model(images)
        # Compute the loss between the predictions (outputs) and the ground-truth labels
        loss = criterion(outputs, labels)
        # Do backpropagation to update the parameters of your model
        loss.backward()
        # Performs a single optimization step (parameter update)
        optimizer.step()
        train_loss += loss.item()
        # The outputs are one-hot labels, we need to find the actual predicted
        # labels which have the highest output confidence
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        # Print every 100 steps the following information
        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                             len(train_dataset) // batch_size,
                                                                             train_loss / (batch_idx + 1),
                                                                             100. * train_correct / train_total))
    # Testing phase
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # TODO: Put the images and labels on the GPU
            images = images.to(torch.device('cuda'))
            labels = labels.to(torch.device('cuda'))

            # Perform the actual inference
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    print('Test loss: %.4f Test accuracy: %.2f %%' % (test_loss / (batch_idx + 1),100. * test_correct / test_total))

    # TODO: Save the PyTorch model in .pt format
    torch.save(model.state_dict(), '{}.pt'.format(model.__class__.__name__))

# Training finished - final summary
end_time = time.time()
total_training_time_s = end_time - start_time

gpu_peak_mem_mb = None
if torch.cuda.is_available():
    peak_bytes = torch.cuda.max_memory_allocated()
    gpu_peak_mem_mb = peak_bytes / (1024.0 ** 2)

print('\n----- Final Summary -----')
print('Training accuracy [%%]: %.2f' % (100. * train_correct / train_total if train_total>0 else 0.0))
print('Test accuracy [%%]: %.2f' % (100. * test_correct / test_total if test_total>0 else 0.0))
print('Total time for training [s]: %.2f' % total_training_time_s)
print('Number of trainable parameters: %d' % num_trainable_params)
print('FLOPs: %.2f' % flops)
if gpu_peak_mem_mb is not None:
    print('GPU memory during training [MB]: %.2f' % gpu_peak_mem_mb)