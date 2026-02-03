"""
TFLite Model Tensor Details Extractor
Logs raw get_tensor_details() output to JSON
"""

import tensorflow as tf
import os
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, type):
            return obj.__name__
        return super().default(obj)


def get_tensor_details_to_json(model_paths, output_json="tensor_details.json"):
    """
    Extract raw tensor details from TFLite models and write to JSON.
    """
    all_tensor_data = {}
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found, skipping...")
            continue
            
        model_name = os.path.basename(model_path).replace('.tflite', '')
        print(f"Processing: {model_name}")
        
        # Load interpreter
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get tensor details - raw output
        tensor_details = interpreter.get_tensor_details()
        
        all_tensor_data[model_name] = tensor_details
        print(f"  Found {len(tensor_details)} tensors")
    
    # Write to JSON
    if all_tensor_data:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(all_tensor_data, f, indent=2, cls=NumpyEncoder)
        
        total = sum(len(v) for v in all_tensor_data.values())
        print(f"\nWrote {total} tensor records to {output_json}")
    else:
        print("No tensor data found.")


if __name__ == "__main__":
    # Define model paths
    models = [
        "conditional_detr_resnet50.tflite",
        "convnext_base-convnext-base-float.tflite",
        "mobile_vit-mobile-vit-float.tflite",
        "mobilenet_v3_large-mobilenet-v3-large-float.tflite",
    ]
    
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_paths = [os.path.join(script_dir, m) for m in models]
    
    # Output JSON path
    output_json = os.path.join(script_dir, "tensor_details.json")
    
    get_tensor_details_to_json(model_paths, output_json)
