import torch
from vgg11 import VGG11
from vgg16 import VGG16
from mobilenet import MobileNetv1

# model_one = torch.load('models/VGG11.pt', map_location='cpu')
# model_one.eval()

# obj = torch.load('../models/VGG11.pt', map_location='cpu')
# print(type(obj))



state_dict = torch.load('../models/MobileNetv1.pt', map_location='cpu')
state_dict = {k: v for k, v in state_dict.items() 
              if not k.endswith('total_ops') and not k.endswith('total_params')}


model_one = MobileNetv1()
model_one.load_state_dict(state_dict)
model_one.eval()
print(model_one)

dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(
    model_one,
    dummy_input,
    "MobileNetv1_RP3B.onnx",
    export_params=True,
    opset_version=16
)

"""
state_dict = torch.load('../models/VGG11.pt', map_location='cpu')

state_dict.pop('total_ops', None)
state_dict.pop('total_params', None)


model_one = VGG11()
model_one.load_state_dict(state_dict)
model_one.eval()

dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(
    model_one,                   # Your loaded model
    dummy_input,             # Dummy input tensor (correct shape!)
    "VGG11_MC1.onnx",         # Output file path
    export_params=True,      # Save trained weights in the file
    opset_version=13,        # ONNX opset (13 is a safe, widely supported choice)
)
"""