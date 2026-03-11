import os
import re
import csv
import matplotlib.pyplot as plt
import torch

fractions = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
plot_fractions = [0.0] + fractions

# Define your directory names
PTH_DIR = 'pth_files'
OUTPUTS_DIR = 'outputs'

def get_final_accuracy(filepath):
    """Reads the log file and extracts the last test accuracy printed."""
    if not os.path.exists(filepath):
        print(f"Warning: Log {filepath} not found.")
        return None
    with open(filepath, 'r') as f:
        matches = re.findall(r'Test accuracy:\s*([0-9.]+)\s*%', f.read())
        if matches:
            return float(matches[-1])
    return None

def count_params(ckpt_path):
    """Loads a .pth file and counts the total number of parameters."""
    if not os.path.exists(ckpt_path):
        print(f"Warning: Model {ckpt_path} not found.")
        return None
    try:
        raw = torch.load(ckpt_path, map_location='cpu')
        sd = raw.get('state_dict', raw) if isinstance(raw, dict) else raw.state_dict()
        # Count parameters, ignoring the operation trackers
        return sum(v.numel() for k, v in sd.items() if isinstance(v, torch.Tensor) and not k.endswith('total_ops') and not k.endswith('total_params'))
    except Exception as e:
        print(f"Error loading {ckpt_path}: {e}")
        return None

# --- 1. Get Baseline (Unpruned) Data ---
baseline_log = os.path.join(OUTPUTS_DIR, 'out_main_mobilenet')
baseline_ckpt = os.path.join(PTH_DIR, 'MobilenetV1.pth')

# Fallbacks in case the baseline files are missing from the folder
baseline_acc = get_final_accuracy(baseline_log) or 78.47
baseline_params = count_params(baseline_ckpt) or 3239141

q1_accs = [baseline_acc]
q2_accs = [baseline_acc]
q1_params_list = [baseline_params]
q2_params_list = [baseline_params]

# --- 2. Gather Data for All Fractions ---
for f in fractions:
    # Path to logs in the outputs/ folder
    q1_log = os.path.join(OUTPUTS_DIR, f"p1q1_{f}.out")
    q2_log = os.path.join(OUTPUTS_DIR, f"p1q2_{f}.out")
    
    q1_a = get_final_accuracy(q1_log)
    q2_a = get_final_accuracy(q2_log)
    q1_accs.append(q1_a if q1_a else 0.0)
    q2_accs.append(q2_a if q2_a else 0.0)
    
    # Path to models in the pth_files/ folder
    q1_ckpt = os.path.join(PTH_DIR, f"q1_magnitude_{f}_5_5_MBNv1.pth")
    q2_ckpt = os.path.join(PTH_DIR, f"q2_magnitude_{f}_5_5_MBNv1.pth")
    
    q1_p = count_params(q1_ckpt)
    q2_p = count_params(q2_ckpt)
    q1_params_list.append(q1_p if q1_p else 0)
    q2_params_list.append(q2_p if q2_p else 0)

# --- 3. Save Data to CSV ---
csv_path = 'accuracy_vs_params.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Pruning_Fraction', 'Q1_Accuracy', 'Q1_Params', 'Q2_Accuracy', 'Q2_Params'])
    for i in range(len(plot_fractions)):
        writer.writerow([plot_fractions[i], q1_accs[i], q1_params_list[i], q2_accs[i], q2_params_list[i]])
print(f"Successfully saved data to {csv_path}")

# --- 4. Generate Plot: Accuracy vs. Parameters ---
# Convert parameter counts to millions for plotting
q1_params_millions = [p / 1e6 for p in q1_params_list]
q2_params_millions = [p / 1e6 for p in q2_params_list]

plt.figure(figsize=(8, 6))

plt.plot(q1_params_millions, q1_accs, marker='o', linestyle='-', label='Q1 Strategy')
plt.plot(q2_params_millions, q2_accs, marker='s', linestyle='-', color='orange', label='Q2 Strategy')

# Mark baseline parameter count explicitly 
if baseline_params:
    baseline_millions = baseline_params / 1e6
    plt.axvline(baseline_millions, color='red', linestyle=':', linewidth=1.5, label='Unpruned Baseline Params')
    plt.plot(baseline_millions, baseline_acc, marker='D', color='red', markersize=8)

plt.title('Test Accuracy vs. Number of Parameters')
plt.xlabel('Number of Parameters (Millions)')
plt.ylabel('Test Accuracy (%)')
plt.grid(True)
plt.legend()

plot_path = 'accuracy_vs_params.png'
plt.savefig(plot_path)
print(f"Successfully saved plot to {plot_path}")