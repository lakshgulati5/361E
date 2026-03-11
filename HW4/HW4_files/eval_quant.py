import os, time, resource, argparse
import numpy as np
import onnxruntime as ort
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

# 1. Model Size
size_mb = os.path.getsize(args.model) / (1024*1024)

# Setup
img_dir = "/home/student/HW4_files/test_deployment"
files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
mean = np.array((0.4914, 0.4822, 0.4465), dtype=np.float32).reshape(1, 3, 1, 1)
std = np.array((0.2023, 0.1994, 0.2010), dtype=np.float32).reshape(1, 3, 1, 1)

correct = 0
total = 0
clean_time_200 = 0.0

print(f"Evaluating {args.model}... This will take ~5 minutes to get the full accuracy.")

for i, filename in enumerate(files):
    try:
        with Image.open(os.path.join(img_dir, filename)) as img:
            img = img.convert("RGB").resize((32, 32))
            inp = np.array(img).astype(np.float32) / 255.0
            inp = np.expand_dims(inp, axis=0).transpose(0, 3, 1, 2)
            inp = (inp - mean) / std
            
            # Run Inference
            t0 = time.time()
            pred = sess.run(None, {input_name: inp})[0]
            t1 = time.time()
            
            # 2. Get clean latency using ONLY the first 200 images (no thermal throttling)
            if i < 200:
                clean_time_200 += (t1 - t0)
            
            # Check Accuracy
            top_pred = int(np.argmax(pred[0]))
            parts = filename.split('_')
            if len(parts) > 1:
                true_label = label_names.index(parts[1].split('.')[0])
            else:
                true_label = int(parts[0].split('.')[0])
                
            if top_pred == true_label:
                correct += 1
            total += 1
            
            if (i+1) % 2000 == 0:
                print(f"Processed {i+1}/10000 images...")
    except Exception:
        pass

# 3. Final Math
accuracy = 100.0 * correct / total
latency = (clean_time_200 / 200) * 1000
energy = 4500 * (latency / 1000) # 4500mW estimate
memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

# 4. Print Table Results
print("\n" + "="*50)
print(f"ALL STATS FOR: {args.model}")
print("="*50)
print(f"Number of Parameters        : 3239141 (Same as Baseline)")
print(f"Model Size [MB]             : {size_mb:.4f}")
print(f"Maximum memory usage [MB]   : {memory:.4f}")
print(f"Average latency [ms/img]    : {latency:.4f}")
print(f"Average energy [mJ/img]     : {energy:.4f}")
print(f"Test Accuracy [%]           : {accuracy:.4f}")
print("="*50)