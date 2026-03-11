import numpy as np
import onnxruntime as rt
from tqdm import tqdm
import os
from PIL import Image
import time
import math
import glob

# optional sensors
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

# command-line parsing ----------------------------------------------------
import argparse

parser = argparse.ArgumentParser(description="Deploy an ONNX CIFAR10 model and compute accuracy")
parser.add_argument('--model', choices=['VGG11','VGG16','MobileNetv1'],
                    default='VGG11', help='which network architecture to use')
parser.add_argument('--board', choices=['MC1','RP3B'], default='MC1',
                    help='which target board the ONNX file was generated for')
args = parser.parse_args()

# construct filename from arguments; ONNX files live in same directory as this
# script
script_dir = os.path.dirname(__file__)
onnx_model_name = os.path.join(script_dir,
                               f"{args.model}_{args.board}.onnx")

print("Using model", onnx_model_name)

# Create Inference session using ONNX runtime
sess_options = rt.SessionOptions()
sess_options.intra_op_num_threads = 4
try:
    sess = rt.InferenceSession(onnx_model_name, sess_options)
except Exception as e:
    print(f"Failed to load ONNX model '{onnx_model_name}': {e}")
    print("This usually means the model file uses a newer ONNX format than the installed onnxruntime supports.")
    print("Try re-exporting the model with a lower opset version (e.g. opset 13) or update onnxruntime.")
    raise

# Get the input name for the ONNX model
input_name = sess.get_inputs()[0].name
print("Input name  :", input_name)

# Get the shape of the input
input_shape = sess.get_inputs()[0].shape
print("Input shape :", input_shape)

# Mean and standard deviation 
mean = np.array((0.4914, 0.4822, 0.4465))
std = np.array((0.2023, 0.1994, 0.2010))

# Label names for CIFAR10 Dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# The test_deployment folder contains all 10.000 images from the testing dataset of CIFAR10 in .png format

def _get_rss_bytes():
    if psutil:
        return psutil.Process().memory_info().rss
    # fallback: try reading /proc (Linux-only)
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    parts = line.split()
                    # kB
                    return int(parts[1]) * 1024
    except Exception:
        return None


def _get_temp_c():
    # try psutil sensors
    if psutil:
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # pick highest available temperature
                vals = []
                for k, v in temps.items():
                    for entry in v:
                        if entry.current is not None:
                            vals.append(entry.current)
                if vals:
                    return max(vals)
        except Exception:
            pass
    # fallback: read thermal zones
    try:
        vals = []
        for p in glob.glob('/sys/class/thermal/thermal_zone*/temp'):
            with open(p) as f:
                t = int(f.read().strip())
                # many sensors report millidegrees
                if t > 1000:
                    vals.append(t / 1000.0)
                else:
                    vals.append(t)
        if vals:
            return max(vals)
    except Exception:
        pass
    return None


def _get_gpu_power_mw():
    if not _nvml_available:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # returns milliwatts
        return pynvml.nvmlDeviceGetPowerUsage(handle)
    except Exception:
        return None


correct = 0
total = 0

# storage for telemetry
samples_ts = []
samples_mem = []
samples_temp = []
samples_power = []

total_inference_time = 0.0

files = list(os.listdir("/home/student/HW3_files/test_deployment"))
files = sorted(os.listdir("/home/student/HW3_files/test_deployment"))

correct = 0
total = 0

for filename in tqdm(files):

    if not filename.endswith(".png"):
        continue

    path = os.path.join("/home/student/HW3_files/test_deployment", filename)

    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((32, 32))

        img = np.array(img).astype(np.float32) / 255.0

        # ---- NORMALIZATION (comment out if accuracy still 0) ----
        img = (img - mean) / std

        img = np.expand_dims(img, axis=0)

        # ---- AUTO HANDLE NCHW vs NHWC ----
        model_shape = sess.get_inputs()[0].shape

        # If model expects channels first
        if len(model_shape) == 4 and model_shape[1] == 3:
            img = img.transpose(0, 3, 1, 2)

        # inference
        t0 = time.perf_counter()
        pred = sess.run(None, {input_name: img})[0]
        t1 = time.perf_counter()

        total_inference_time += (t1 - t0)

        # telemetry
        samples_ts.append(time.time())

        rss = _get_rss_bytes()
        samples_mem.append(rss if rss is not None else float('nan'))

        temp = _get_temp_c()
        samples_temp.append(temp if temp is not None else float('nan'))

        power = _get_gpu_power_mw()
        samples_power.append(power if power is not None else float('nan'))

        # prediction
        pred_label = int(np.argmax(pred[0]))

        # ---- SAFE LABEL PARSING ----
        label_token = filename.split('_')[0]

        if label_token.isdigit():
            true_label = int(label_token)
        else:
            true_label = label_names.index(label_token)

        if pred_label == true_label:
            correct += 1

        total += 1

print(f"\nTest accuracy: {100.0 * correct/total:.2f}% ({correct}/{total})")
# final accuracy
print(f"Test accuracy: {100.0 * correct/total:.2f}% ({correct}/{total})")

# summary telemetry
print('\n--- Telemetry summary ---')
print(f"Total inference time (s): {total_inference_time:.4f}")
throughput = (total / total_inference_time) if total_inference_time>0 else float('nan')
print(f"Throughput (images/s): {throughput:.2f}")

def _safe_stats(arr):
    import math
    vals = [x for x in arr if x is not None and (not isinstance(x, float) or not math.isnan(x))]
    if not vals:
        return (None, None)
    return (float(sum(vals))/len(vals), max(vals))

avg_mem, peak_mem = _safe_stats(samples_mem)
if avg_mem is not None:
    print(f"Memory (RSS) avg: {avg_mem/1024/1024:.2f} MB, peak: {peak_mem/1024/1024:.2f} MB")
else:
    print("Memory (RSS): not available")

avg_temp, peak_temp = _safe_stats(samples_temp)
if avg_temp is not None:
    print(f"Temperature avg: {avg_temp:.2f} C, peak: {peak_temp:.2f} C")
else:
    print("Temperature: not available")

avg_power_mw, peak_power_mw = _safe_stats(samples_power)
if avg_power_mw is not None:
    # avg_power_mw is in milliwatts
    avg_power_w = avg_power_mw / 1000.0
    total_energy_j = avg_power_w * total_inference_time
    print(f"Power avg: {avg_power_w:.3f} W, peak: {peak_power_mw/1000.0:.3f} W")
    print(f"Estimated energy during inference: {total_energy_j:.3f} J")
else:
    print("Power: not available")

print('--- End telemetry ---')

import csv

# Your telemetry arrays
# samples_ts = []
# samples_mem = []
# samples_temp = []
# samples_power = []

def save_list_to_csv(filename, data, header):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([header])
        for value in data:
            writer.writerow([value])

# Save each list
save_list_to_csv("timestamps.csv", samples_ts, "timestamp")
save_list_to_csv("memory.csv", samples_mem, "memory")
save_list_to_csv("temperature.csv", samples_temp, "temperature")
save_list_to_csv("power.csv", samples_power, "power")