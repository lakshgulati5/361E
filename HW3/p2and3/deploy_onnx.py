import numpy as np
import onnxruntime as rt
from tqdm import tqdm
import os
from PIL import Image
import time
import math
import glob
import argparse

# --- Optional Sensors ---
try:
    import psutil
except Exception:
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_available = True
except Exception:
    _nvml_available = False

# --- Command-line Parsing ---
parser = argparse.ArgumentParser(description="Deploy an ONNX CIFAR10 model and compute accuracy")
parser.add_argument('--model', choices=['VGG11','VGG16','MobileNetv1'],
                    default='VGG11', help='which network architecture to use')
parser.add_argument('--board', choices=['MC1','RP3B'], default='MC1',
                    help='which target board the ONNX file was generated for')
args = parser.parse_args()

script_dir = os.path.dirname(__file__)
onnx_model_name = os.path.join(script_dir, f"{args.model}_{args.board}.onnx")

print("Using model:", onnx_model_name)

# --- Create Inference session ---
sess_options = rt.SessionOptions()
sess_options.intra_op_num_threads = 4
try:
    sess = rt.InferenceSession(onnx_model_name, sess_options)
except Exception as e:
    print(f"Failed to load ONNX model: {e}")
    raise

input_name = sess.get_inputs()[0].name

# --- Preprocessing Constants ---
# CIFAR-10 Normalization stats
mean = np.array((0.4914, 0.4822, 0.4465), dtype=np.float32).reshape(1, 3, 1, 1)
std = np.array((0.2023, 0.1994, 0.2010), dtype=np.float32).reshape(1, 3, 1, 1)
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# --- Telemetry Helpers ---
def _get_rss_bytes():
    if psutil:
        return psutil.Process().memory_info().rss
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) * 1024
    except: return None

def _get_temp_c():
    if psutil:
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                vals = [entry.current for k, v in temps.items() for entry in v if entry.current is not None]
                if vals: return max(vals)
        except: pass
    try:
        vals = []
        for p in glob.glob('/sys/class/thermal/thermal_zone*/temp'):
            with open(p) as f:
                t = int(f.read().strip())
                vals.append(t / 1000.0 if t > 1000 else t)
        if vals: return max(vals)
    except: pass
    return None

def _get_gpu_power_mw():
    if not _nvml_available: return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetPowerUsage(handle)
    except: return None

# --- Main Evaluation Loop ---
img_dir = "/home/student/HW3_files/test_deployment"
# Ensure we only grab images and sort them
files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])

correct = 0
total = 0
total_inference_time = 0.0
samples_mem, samples_temp, samples_power = [], [], []

# Progress bar with live accuracy stats
pbar = tqdm(files, desc="Processing Images")

for filename in pbar:
    file_path = os.path.join(img_dir, filename)
    
    try:
        with Image.open(file_path) as img:
            img = img.convert("RGB").resize((32, 32))
            
            # 1. Scale to [0, 1] and Transpose to NCHW (1, 3, 32, 32)
            input_image = np.array(img).astype(np.float32) / 255.0
            input_image = np.expand_dims(input_image, axis=0).transpose(0, 3, 1, 2)
            
            # 2. Apply Normalization (Critical!)
            input_image = (input_image - mean) / std

            # 3. Inference
            t0 = time.perf_counter()
            pred_onnx = sess.run(None, {input_name: input_image})[0]
            t1 = time.perf_counter()
            total_inference_time += (t1 - t0)

            # 4. Telemetry Collection
            rss = _get_rss_bytes()
            samples_mem.append(rss if rss is not None else float('nan'))
            samples_temp.append(_get_temp_c() or float('nan'))
            samples_power.append(_get_gpu_power_mw() or float('nan'))

            # 5. Prediction
            top_prediction = int(np.argmax(pred_onnx[0]))
            
            # 6. FIXED LABEL PARSING: 
            # Handles "1409_frog.png" -> word is "frog", ID is "1409"
            parts = filename.split('_')
            if len(parts) > 1:
                label_word = parts[1].split('.')[0] # Get 'frog' from 'frog.png'
                true_label = label_names.index(label_word)
            else:
                # Fallback if filename is just "6.png"
                true_label = int(parts[0].split('.')[0])

            if top_prediction == true_label:
                correct += 1
            total += 1

            # Update Progress Bar with current accuracy and class names
            current_acc = (correct / total) * 100 if total > 0 else 0
            pbar.set_postfix({
                'Acc': f'{current_acc:.2f}%', 
                'Pred': label_names[top_prediction],
                'True': label_names[true_label]
            })

    except Exception as e:
        # Skip corrupted images or files that don't match naming convention
        continue

# --- Final Results ---
print(f"\nFinal Test accuracy: {100.0 * correct/total:.2f}% ({correct}/{total})")

print('\n--- Telemetry summary ---')
print(f"Total inference time (s): {total_inference_time:.4f}")
throughput = (total / total_inference_time) if total_inference_time > 0 else 0
print(f"Throughput (images/s): {throughput:.2f}")

def _safe_stats(arr):
    vals = [x for x in arr if not math.isnan(x)]
    if not vals: return (None, None)
    return (float(sum(vals))/len(vals), max(vals))

avg_mem, peak_mem = _safe_stats(samples_mem)
if avg_mem:
    print(f"Memory (RSS) avg: {avg_mem/1024/1024:.2f} MB, peak: {peak_mem/1024/1024:.2f} MB")

avg_temp, peak_temp = _safe_stats(samples_temp)
if avg_temp:
    print(f"Temperature avg: {avg_temp:.2f} C, peak: {peak_temp:.2f} C")

avg_power_mw, peak_power_mw = _safe_stats(samples_power)
if avg_power_mw:
    avg_power_w = avg_power_mw / 1000.0
    print(f"Power avg: {avg_power_w:.3f} W, peak: {peak_power_mw/1000.0:.3f} W")
    print(f"Estimated energy during inference: {avg_power_w * total_inference_time:.3f} J")

print('--- End telemetry ---')