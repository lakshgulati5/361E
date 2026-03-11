import argparse
import numpy as np
import torch
import random
import torchvision
import torchvision.transforms as transforms
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

parser = argparse.ArgumentParser(description='ECE361E HW4 Quantization')
# TODO add argument for ONNX model
parser.add_argument('--model', type=str, required=True, help="Path to ONNX model")
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
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

# TODO add CIFAR10 Calibration Data Reader
class CIFAR10DataReader(CalibrationDataReader):
    def __init__(self, dataloader, input_name):
        self.iterator = iter(dataloader)
        self.input_name = input_name

    def get_next(self):
        try:
            images, _ = next(self.iterator)
            return {self.input_name: images.numpy()}
        except StopIteration:
            return None

# TODO Preprocess model for quantization
preprocessed_model = args.model.replace('.onnx', '_infer.onnx')
quant_pre_process(args.model, preprocessed_model)

# TODO Use 1,000 images from the CIFAR10 Calibration Data Reader
subset = torch.utils.data.Subset(train_dataset, range(1000))
dataloader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)

session = ort.InferenceSession(preprocessed_model, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
data_reader = CIFAR10DataReader(dataloader, input_name)

# TODO Perform static quantization
quantized_model = args.model.replace('.onnx', '_quant.onnx')
quantize_static(
    model_input=preprocessed_model,
    model_output=quantized_model,
    calibration_data_reader=data_reader,
    quant_format=QuantFormat.QOperator,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8
)