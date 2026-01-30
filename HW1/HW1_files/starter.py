import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import argparse
import random
import numpy as np
import os
import csv
import time
from torch.utils.data import DataLoader

# Argument parser
parser = argparse.ArgumentParser(description='ECE361E HW1 - Starter code')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=25, help='Number of epoch to train')
# Define the learning rate of your optimizer
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
args = parser.parse_args()

# The size of input features
input_size = 28 * 28
# The number of target classes, you have 10 digits to classify
num_classes = 10

# Always make assignments to local variables from your args at the beginning of your code for better
# control and adaptability
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='data', train=False, transform=transforms.ToTensor())

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, generator=g)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Define your model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    # Your model only contains a single linear layer
    def forward(self, x):
        out = self.linear(x)
        return out


model = LogisticRegression(input_size, num_classes)
model = model.to(torch.device('cuda'))

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Prepare measurements
total_train_time = 0.0
if device.type == 'cuda':
    # reset peak memory statistics for training measurement
    torch.cuda.reset_peak_memory_stats(device)

# Training loop
for epoch in range(num_epochs):
    # Training phase
    train_correct = 0
    train_total = 0
    train_loss = 0
    # Sets the model in training mode.
    model = model.train()
    # measure only the training phase time
    t_train_start = time.time()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(torch.device('cuda')) 
        labels = labels.to(torch.device('cuda'))
        # Here we vectorize the 28*28 images as several 784-dimensional inputs
        images = images.view(-1, input_size)
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
    t_train_end = time.time()
    total_train_time += (t_train_end - t_train_start)
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
            # Here we vectorize the 28*28 images as several 784-dimensional inputs
            images = images.view(-1, input_size).to(torch.device('cuda'))
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
    print('Test accuracy: %.2f %% Test loss: %.4f' % (100. * test_correct / test_total, test_loss / (batch_idx + 1)))
    # write minimal per-epoch metrics (one row per epoch)
    epoch_train_loss = train_loss / len(train_loader)
    epoch_test_loss = test_loss / len(test_loader)
    epoch_train_acc = 100. * train_correct / train_total if train_total > 0 else 0.0
    epoch_test_acc = 100. * test_correct / test_total if test_total > 0 else 0.0
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metrics.csv')
    write_header = not os.path.exists(csv_path)
    try:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['epoch', 'train_loss', 'test_loss', 'train_acc', 'test_acc'])
            writer.writerow([epoch+1, epoch_train_loss, epoch_test_loss, epoch_train_acc, epoch_test_acc])
    except Exception as e:
        print('Failed to write metrics CSV:', e)

# After training complete: collect training time and GPU memory usage
train_time_s = total_train_time
if device.type == 'cuda':
    peak_gpu_bytes = torch.cuda.max_memory_allocated(device)
    peak_gpu_mb = float(peak_gpu_bytes) / (1024.0 * 1024.0)
else:
    peak_gpu_mb = 0.0

# Now measure inference time once on the test dataset, with batch_size=1
test_single_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
model = model.eval()
total_infer_time = 0.0
total_images = 0
with torch.no_grad():
    for images, labels in test_single_loader:
        images = images.view(-1, input_size).to(device)
        labels = labels.to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        outputs = model(images)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        total_infer_time += (t1 - t0)
        total_images += images.size(0)

inference_time_s = total_infer_time
avg_inference_ms = (inference_time_s / total_images) * 1000.0 if total_images > 0 else 0.0

# Save Table 1 summary to file
summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'table1.csv')
try:
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['train_accuracy_pct', 'test_accuracy_pct', 'train_time_s', 'inference_time_s', 'avg_inference_ms', 'gpu_mem_mb'])
        writer.writerow([f"{epoch_train_acc:.4f}", f"{epoch_test_acc:.4f}", f"{train_time_s:.4f}", f"{inference_time_s:.4f}", f"{avg_inference_ms:.4f}", f"{peak_gpu_mb:.4f}"])
    print(f'Saved Table 1 summary to {summary_path}')
except Exception as e:
    print('Failed to save summary:', e)