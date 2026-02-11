import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import time
import random
import numpy as np
import os
import csv

# Argument parser
parser = argparse.ArgumentParser(description='ECE361E HW1 - myCNN')
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
parser.add_argument('--epochs', type=int, default=25, help='Number of epoch to train')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
# Always use normalized dataset for this problem
parser.add_argument('--normalize', action='store_true', default=True, help='Apply normalization to the MNIST datasets')
args = parser.parse_args()

# Model: smaller than SimpleCNN (fewer params)
class MyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MyCNN, self).__init__()
        # Two small conv layers + adaptive pooling + single linear layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.adapt = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.adapt(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# reproducibility
random_seed = 1
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g = torch.Generator()
g.manual_seed(random_seed)

# Dataset (normalized)
if args.normalize:
    mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
else:
    mnist_transform = transforms.ToTensor()

train_dataset = dsets.MNIST(root='data', train=True, transform=mnist_transform, download=True)
test_dataset = dsets.MNIST(root='data', train=False, transform=mnist_transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, generator=g)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

# For logging
metrics_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'mycnn_metrics_{"normalized" if args.normalize else "unnormalized"}.csv')
# remove existing metrics file for clean run
if os.path.exists(metrics_csv):
    os.remove(metrics_csv)

start_time = time.time()
for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
    # evaluate
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    epoch_train_loss = train_loss / len(train_loader)
    epoch_test_loss = test_loss / len(test_loader)
    epoch_train_acc = 100. * train_correct / train_total if train_total>0 else 0.0
    epoch_test_acc = 100. * test_correct / test_total if test_total>0 else 0.0
    print(f'Epoch {epoch+1}/{args.epochs} | Train Loss {epoch_train_loss:.4f} Acc {epoch_train_acc:.2f}% | Test Loss {epoch_test_loss:.4f} Acc {epoch_test_acc:.2f}%')
    # append to csv
    write_header = not os.path.exists(metrics_csv)
    try:
        with open(metrics_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['epoch', 'train_loss', 'test_loss', 'train_acc', 'test_acc'])
            writer.writerow([epoch+1, epoch_train_loss, epoch_test_loss, epoch_train_acc, epoch_test_acc])
    except Exception as e:
        print('Failed to write csv:', e)

end_time = time.time()
training_time = end_time - start_time
final_train_acc = epoch_train_acc
final_test_acc = epoch_test_acc
print(f'Finished training in {training_time:.2f}s | Final train acc {final_train_acc:.2f}% | Final test acc {final_test_acc:.2f}%')

# Save model files and compute sizes
state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mycnn_state.pth')
full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mycnn_full.pth')
torch.save(model.state_dict(), state_path)
torch.save(model, full_path)
size_state = os.path.getsize(state_path)/1024.0
size_full = os.path.getsize(full_path)/1024.0
# compute total params and bytes
total_params = sum(p.numel() for p in model.parameters())
param_bytes = sum(p.numel()*p.element_size() for p in model.parameters())/1024.0
print(f'Total params: {total_params:,} ({param_bytes:.2f} KB)')
print(f'State-dict saved to {state_path} -> {size_state:.2f} KB')
print(f'Full model saved to {full_path} -> {size_full:.2f} KB')

# Append results to table3.csv
table3_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'table3.csv')
write_header = not os.path.exists(table3_path)
try:
    with open(table3_path, 'a', newline='') as tf:
        tw = csv.writer(tf)
        if write_header:
            tw.writerow(['model','normalized','params','param_kb','state_kb','full_kb','final_train_acc','final_test_acc','train_time_s'])
        tw.writerow(['myCNN','yes', total_params, f'{param_bytes:.2f}', f'{size_state:.2f}', f'{size_full:.2f}', f'{final_train_acc:.4f}', f'{final_test_acc:.4f}', f'{training_time:.4f}'])
    print('Appended results to table3.csv')
except Exception as e:
    print('Failed to append table3:', e)

# Generate plots using existing plotting script
try:
    import subprocess
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_exec = os.path.join(script_dir, 'HW1_files', 'hw_one_venv', 'Scripts', 'python.exe')
    # fallback to current interpreter
    if not os.path.exists(python_exec):
        python_exec = None
    if python_exec:
        subprocess.run([python_exec, os.path.join(script_dir, 'HW1_files', 'plot_metrics.py'), metrics_csv], check=True)
    else:
        # run using current interpreter
        import plot_metrics as pm
        pm.main(metrics_csv)
    print('Saved plots (loss_plot.png, accuracy_plot.png)')
except Exception as e:
    print('Failed to run plot_metrics:', e)
