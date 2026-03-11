import numpy as np
import onnxruntime as rt
from tqdm import tqdm
import os
from PIL import Image
import time
import argparse

# --- Command-line Parsing ---
parser = argparse.ArgumentParser(description="Deploy an ONNX CIFAR10 model")
parser.add_argument('--model', choices=['VGG11','VGG16','MobileNetv1'], default='VGG11')
parser.add_argument('--board', choices=['MC1','RP3B'], default='MC1')
args = parser.parse_args()

script_dir = os.path.dirname(__file__)
onnx_model_name = os.path.join(script_dir, f"{args.model}_{args.board}.onnx")

# --- Optimized Session ---
sess_options = rt.SessionOptions()
sess_options.intra_op_num_threads = 4 
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = rt.InferenceSession(onnx_model_name, sess_options)

input_name = sess.get_inputs()[0].name

# --- Pre-computed Preprocessing Constants ---
mean = np.array((0.4914, 0.4822, 0.4465), dtype=np.float32).reshape(1, 3, 1, 1)
std = np.array((0.2023, 0.1994, 0.2010), dtype=np.float32).reshape(1, 3, 1, 1)
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

img_dir = "/home/student/HW3_files/test_deployment"
files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])

correct = 0
total = 0
start_time = time.perf_counter()

# --- Fast Evaluation Loop ---
# Removed .set_postfix to stop terminal lag
for filename in tqdm(files, desc="Evaluating Accuracy"):
    file_path = os.path.join(img_dir, filename)
    
    try:
        with Image.open(file_path) as img:
            # Standard Preprocessing
            img = img.convert("RGB").resize((32, 32))
            input_image = np.array(img).astype(np.float32) / 255.0
            # NCHW Transpose
            input_image = input_image.transpose(2, 0, 1)[np.newaxis, ...] 
            input_image = (input_image - mean) / std

        # Inference
        pred_onnx = sess.run(None, {input_name: input_image})[0]
        top_prediction = np.argmax(pred_onnx[0])
        
        # Accurate Label Parsing (ID_WORD.png)
        label_word = filename.split('_')[1].split('.')[0]
        true_label = label_names.index(label_word)

        if top_prediction == true_label:
            correct += 1
        total += 1
        
    except Exception:
        continue

end_time = time.perf_counter()
total_time = end_time - start_time

# --- Final Results ---
print("-" * 30)
if total > 0:
    accuracy = 100.0 * correct / total
    print(f"Final Accuracy : {accuracy:.2f}%")
    print(f"Correct/Total  : {correct}/{total}")
    print(f"Total Time     : {total_time:.2f}s")
    print(f"Throughput     : {total / total_time:.2f} img/s")
else:
    print("No images were processed.")
print("-" * 30)