import os
import torch
import torch.nn as nn
from torchsummary import summary

try:
    from thop import profile
except Exception:
    profile = None

try:
    from ptflops import get_model_complexity_info
except Exception:
    get_model_complexity_info = None

# SimpleCNN definition (same as in p3_q1.py)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(7 * 7 * 64, num_classes)

    def forward(self, x):
        out = torch.relu(self.conv1(x))
        out = self.pool1(out)
        out = torch.relu(self.conv2(out))
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out


def bytes_to_human(n):
    # simple helper to format bytes
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(num_classes=10).to(device)

    # 1) Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Compute total parameter bytes (assumes current dtype element size)
    total_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Total params: {total_params:,} ({total_param_bytes/1024:.2f} KB)")
    print(f"Trainable params: {trainable_params:,}")

    # 2) torchsummary
    print('\ntorchsummary (layerwise)')
    try:
        summary(model, (1, 28, 28), device=str(device))
    except Exception as e:
        print('torchsummary failed:', e)
    # Explicit torchsummary-style params size (MB and KB)
    params_size_mb = total_param_bytes / (1024 ** 2)
    params_size_kb = total_param_bytes / 1024
    print(f"torchsummary -> Params size (MB): {params_size_mb:.2f} MB ({params_size_kb:.2f} KB)")
    # 3) MACs / FLOPs and params using thop or ptflops
    print('\nMACs / FLOPs')
    dummy_input = torch.randn(1, 1, 28, 28).to(device)

    if profile is not None:
        try:
            macs, params = profile(model, inputs=(dummy_input,), verbose=False)
            print(f"thop -> MACs: {macs:,}, params: {params:,}")
            print(f"thop -> FLOPs (approx) = 2 * MACs = {2*macs:,}")
        except Exception as e:
            print('thop.profile failed:', e)

    if get_model_complexity_info is not None:
        try:
            macs_str, params_str = get_model_complexity_info(model, (1, 28, 28), as_strings=True, print_per_layer_stat=False)
            print('ptflops ->', macs_str, params_str)
            # note: ptflops reports MACs by default (see library docs)
        except Exception as e:
            print('ptflops failed:', e)

    if profile is None and get_model_complexity_info is None:
        print('Install `thop` (pip install thop) or `ptflops` (pip install ptflops) to get MACs/FLOPs.')

    # 4) Saved model size
    print('\nSaved model size')
    state_path = 'simplecnn_state.pth'
    full_path = 'simplecnn_full.pth'
    torch.save(model.state_dict(), state_path)
    torch.save(model, full_path)
    size_state = os.path.getsize(state_path)
    size_full = os.path.getsize(full_path)
    print(f"State-dict saved to {state_path} -> {bytes_to_human(size_state)} ({size_state/1024:.2f} KB)")
    print(f"Full model saved to {full_path} -> {bytes_to_human(size_full)} ({size_full/1024:.2f} KB)")
    print('\nDone.')


if __name__ == '__main__':
    main()
