import csv
import os
import sys


script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from thop import profile
except Exception as e:
    print('thop not available:', e)
    profile = None

try:
    from p3_q1 import SimpleCNN
except Exception as e:
    print('Could not import SimpleCNN from p3_q1:', e)
    class SimpleCNN:
        def __init__(self, num_classes=10):
            raise RuntimeError('SimpleCNN import failed')

try:
    from p3_q2 import MyCNN
except Exception as e:
    print('Could not import MyCNN from p3_q2_mycnn:', e)
    class MyCNN:
        def __init__(self, num_classes=10):
            raise RuntimeError('MyCNN import failed')

results = {}

if profile is None:
    print('thop not installed; cannot compute MACs')
else:
    import torch
    device = torch.device('cpu')
    # SimpleCNN
    try:
        model = SimpleCNN(10)
        model.to(device)
        dummy = torch.randn(1,1,28,28).to(device)
        macs, params = profile(model, inputs=(dummy,), verbose=False)
        results['SimpleCNN'] = {'macs': int(macs), 'flops': int(2*macs)}
    except Exception as e:
        print('Failed to profile SimpleCNN:', e)
    # MyCNN
    try:
        model = MyCNN(10)
        model.to(device)
        dummy = torch.randn(1,1,28,28).to(device)
        macs, params = profile(model, inputs=(dummy,), verbose=False)
        results['myCNN'] = {'macs': int(macs), 'flops': int(2*macs)}
    except Exception as e:
        print('Failed to profile MyCNN:', e)

print('Profile results:', results)
