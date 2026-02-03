import tensorflow as tf
import numpy as np
interpreter = tf.lite.Interpreter(model_path="mobile_vit-mobile-vit-float.tflite")
interpreter.allocate_tensors()

total_bytes = 0
total_params = 0

for t in interpreter.get_tensor_details():
    try:
        data = interpreter.get_tensor(t['index'])
        print(data.shape, data.dtype, data.nbytes, data.size)
        total_params += data.size
        total_bytes += data.nbytes
    except ValueError:
        pass

print("Total parameters:", total_params)
print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")

print("Model size (MB):", total_bytes / (1024 * 1024))

