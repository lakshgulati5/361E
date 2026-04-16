import time
import torch
from models.get_model import get_model
from utils.dataset_utils import load_data
from tqdm import tqdm
from torch import nn
from copy import deepcopy


def cal_uniform_act(out, beta, device):
    zero_mat = torch.zeros(out.size()).to(device)
    softmax = nn.Softmax(dim=1)
    logsoftmax = nn.LogSoftmax(dim=1)
    kldiv = nn.KLDivLoss(reduce=True)
    cost = beta * kldiv(logsoftmax(out), softmax(zero_mat))
    return cost


def train(model, loss_func, dev_idx, model_path, cuda_name, optimizer, local_epochs, data_iid,
          loss_type="fedavg", mu=1.0, beta=1, verbose=False, seed=42):
    """
    Trains the provided model with the training data loaded from disk.

    Parameters:
        model (torch.nn.Module): The PyTorch model to train.
        loss_func (function): The loss function used for training.
        dev_idx (int): The index of the device used for loading data.
        model_path (str): The path where the trained model state will be saved.
        cuda_name (str): The name of the CUDA device to be used for training.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        local_epochs (int): The number of epochs to train.
        verbose (bool, optional): If True, displays status messages during execution. Defaults to False.
        data_iid (bool, optional): Indicates whether the data is independently and identically distributed.

    Returns:
        tuple: The final training loss and accuracy.
    """
    device = torch.device(cuda_name)
    data_loader = load_data(data_iid=data_iid, dev_idx=dev_idx, seed=seed)

    model.train()

    if loss_type == 'fedprox':
        w_glob = deepcopy(model.state_dict())
        l2_norm = nn.MSELoss()

    for epoch in range(local_epochs):
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(tqdm(data_loader)):
            images, labels = images.to(device), labels.to(device)
            if loss_type == 'fedmax':
                outputs, activations = model(images)
            else:
                outputs = model(images)

            loss = loss_func(outputs, labels)
            if loss_type == 'fedprox':
                reg_loss = 0
                for name, param in model.named_parameters():
                    reg_loss += l2_norm(param, w_glob[name])
                loss += mu / 2 * reg_loss
            elif loss_type == 'fedmax':
                loss += cal_uniform_act(activations, beta, device)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        if verbose:
            print(f"[+] Epoch {epoch}: Loss - {train_loss / (batch_idx + 1):.4f}, Accuracy - {100. * correct / total:.2f}%\n")
    torch.save(model.state_dict(), model_path)
    return train_loss / (batch_idx + 1), (100. * correct / total)


def test(model, loss_func, cuda_name, verbose=False, seed=42, loss_type='fedavg'):
    """
    Tests the provided model with the test data loaded from disk.

    Parameters:
        model (torch.nn.Module): The PyTorch model to test.
        loss_func (function): The loss function used for testing.
        cuda_name (str): The name of the CUDA device to be used for testing.
        verbose (bool, optional): If True, displays status messages during execution. Defaults to False.

    Returns:
        tuple: The final testing loss, accuracy, and the total number of test samples (optional).
    """
    device = torch.device(cuda_name)
    model = model.to(device)
    data_loader = load_data(test_global=True, seed=seed)

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(data_loader)):
            images, labels = images.to(device), labels.to(device)
            if loss_type == 'fedmax':
                outputs, activations = model(images)
            else:
                outputs = model(images)
            loss = loss_func(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    if verbose:
        print(f'[+] Test Loss: {test_loss / (batch_idx + 1)}, Test Accuracy: {100. * correct / total}%')
        
    return test_loss / (batch_idx + 1), 100. * correct / total


def local_training(model_name, loss_func, loss_type, model_path, cuda_name, learning_rate, local_epochs, dev_idx,
                   data_iid, verbose=False, seed=42, mu=1.0, beta=1):
    """
    Performs training of a local model for federated learning.

    Parameters:
        model_name (str): The name of the model.
        loss_func (function): The loss function for training.
        model_path (str): The path of the pre-trained model.
        cuda_name (str): The name of the CUDA device for model training.
        learning_rate (float): The learning rate for training.
        local_epochs (int): The number of epochs for local training.
        dev_idx (int): The index of the client device.
        verbose (bool, optional): If True, displays status messages during execution. Defaults to False.
        data_iid (bool, optional): If True, the data is considered IID. Defaults to True.

    Returns:
        tuple: Training loss, training accuracy, test loss, test accuracy, and training time.
    """

    device = torch.device(cuda_name)

    model = get_model(model_name=f"{model_name}", loss_type=loss_type).to(device)
    model.load_state_dict(torch.load(model_path))

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    if verbose:
        print(f"[+] Device {dev_idx} training...")

    start_time = time.time()

    train_loss, train_acc = train(
        model=model, loss_func=loss_func, loss_type=loss_type,
        model_path=model_path, cuda_name=cuda_name, optimizer=optimizer,
        local_epochs=local_epochs, verbose=verbose,
        data_iid=data_iid, dev_idx=dev_idx, seed=seed, mu=mu, beta=beta
    )

    train_time = time.time() - start_time

    print(f"[+] Train loss: {train_loss}, Train accuracy: {train_acc}%")

    return train_loss, train_acc, train_time
