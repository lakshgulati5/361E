import argparse
import numpy as np
import onnxruntime as ort
import os
import resource
import threading
import time
from PIL import Image

# --- Background Power Polling ---
keep_polling = True
power_readings_mw = []

def read_power_mw():
    try:
        with open('/sys/class/hwmon/hwmon0/power1_input', 'r') as f:
            return float(f.read().strip()) / 1000.0 
    except Exception:
        return float('nan')

def poll_power():
    while keep_polling:
        p = read_power_mw()
        if not np.isnan(p):
            power_readings_mw.append(p)
        time.sleep(0.100) # Poll every 100ms

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

img_dir = "/home/student/HW4_files/test_deployment"

# ONLY GRAB 100 IMAGES (Finishes in ~3 seconds)
files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])[:100]

model_size_mb = os.path.getsize(args.model) / (1024 * 1024)

sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name

mean = np.array((0.4914, 0.4822, 0.4465), dtype=np.float32).reshape(1, 3, 1, 1)
std = np.array((0.2023, 0.1994, 0.2010), dtype=np.float32).reshape(1, 3, 1, 1)

power_thread = threading.Thread(target=poll_power)
power_thread.start()

total_inference_time = 0.0

for filename in files:
    file_path = os.path.join(img_dir, filename)
    try:
        with Image.open(file_path) as img:
            img = img.convert("RGB").resize((32, 32))
            input_image = np.array(img).astype(np.float32) / 255.0
            input_image = np.expand_dims(input_image, axis=0).transpose(0, 3, 1, 2)
            input_image = (input_image - mean) / std
            
            # Time the inference to calculate mJ
            t0 = time.time()
            sess.run(None, {input_name: input_image})
            t1 = time.time()
            total_inference_time += (t1 - t0)
    except Exception:
        continue

keep_polling = False
power_thread.join()

# Calculate stats
max_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
avg_latency_ms = (total_inference_time / len(files)) * 1000

if len(power_readings_mw) > 0:
    avg_power_mw = sum(power_readings_mw) / len(power_readings_mw)
    avg_energy_mj = avg_power_mw * (avg_latency_ms / 1000.0)
else:
    avg_power_mw = 0.0
    avg_energy_mj = 0.0

print(f"\n--- MISSING STATS FOR: {args.model} ---")
print(f"Model Size [MB]:               {model_size_mb:.4f}")
print(f"Maximum memory usage [MB]:     {max_memory_mb:.4f}")
if avg_power_mw > 0:
    print(f"Average Energy per image [mJ]: {avg_energy_mj:.4f}")
else:
    print(f"Average Energy per image [mJ]: SENSOR NOT FOUND")
print("-" * 45)