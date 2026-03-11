import argparse
import os
import torch
from mobilenet import MobileNetv1


def export_checkpoint_to_onnx(ckpt_path, out_name=None, opset_version=16, model_cls=MobileNetv1):
    """Load a checkpoint (state-dict or full model) and export to ONNX.

    Returns the output filename on success.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    raw = torch.load(ckpt_path, map_location='cpu')

    if isinstance(raw, torch.nn.Module):
        model = raw
    else:
        if isinstance(raw, dict):
            if 'state_dict' in raw:
                sd = raw['state_dict']
            elif 'model_state_dict' in raw:
                sd = raw['model_state_dict']
            else:
                sd = raw
        else:
            sd = raw

        if isinstance(sd, dict):
            sd = {k: v for k, v in sd.items() if not k.endswith('total_ops') and not k.endswith('total_params')}

        model = model_cls()
        try:
            model.load_state_dict(sd)
        except Exception:
            model.load_state_dict(sd, strict=False)

    model = model.cpu().eval()

    if out_name is None:
        base = os.path.splitext(os.path.basename(ckpt_path))[0]
        out_name = f'{base}.onnx'

    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(model, dummy_input, out_name, export_params=True, opset_version=opset_version,
                      input_names=['input'], output_names=['output'])
    return out_name


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch checkpoint to ONNX')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Path to checkpoint (if not provided will be constructed from prune params)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output ONNX filename')
    parser.add_argument('--opset', type=int, default=16, help='ONNX opset version')
    # pruning parameters (used to build the checkpoint name saved by prune_model.py)
    parser.add_argument('--prune_metric', type=str, default='magnitude', help='Prune metric used in prune_model')
    parser.add_argument('--prune_ratio', type=float, default=0.5, help='Prune ratio used in prune_model')
    parser.add_argument('--pruning_iter', type=int, default=5, help='Pruning iteration index used in prune_model')
    parser.add_argument('--finetuning_epochs', type=int, default=10, help='Finetuning epochs used in prune_model')
    args = parser.parse_args()

    # If checkpoint not provided, construct filename matching prune_model.py saves
    if args.checkpoint is None:
        ckpt = f"{args.prune_metric}_{args.prune_ratio}_{args.pruning_iter}_{args.finetuning_epochs}_MBNv1.pth"
    else:
        ckpt = args.checkpoint

    out_name = export_checkpoint_to_onnx(ckpt, out_name=args.output, opset_version=args.opset)
    print(f'Exported ONNX model to {out_name}')


if __name__ == '__main__':
    main()
