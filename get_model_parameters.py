"""
TFLite Model Parameter Counter
Calculates the number of parameters for each TFLite model in the workspace.
"""

import tensorflow as tf
import os
import glob
import json
import csv
from collections import defaultdict


def count_model_parameters(model_path):
    """
    Count the number of parameters in a TFLite model.
    
    Parameters are typically stored in weight tensors (kernels, biases, etc.)
    
    Args:
        model_path: Path to the TFLite model file
        
    Returns:
        Dictionary with parameter counts and details
    """
    # Load the model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get tensor details
    tensor_details = interpreter.get_tensor_details()
    
    total_params = 0
    weight_params = 0
    trainable_tensors = []
    
    # Weight-related keywords that indicate parameter tensors
    # In TFLite models converted from various frameworks, weights are often named:
    # - 'arith.constant' (from MLIR/StableHLO conversion)
    # - 'kernel', 'weight', 'bias' (from TensorFlow/Keras)
    # - 'Const', 'Constant' (from various converters)
    weight_keywords = ['arith.constant', 'constant', 'kernel', 'weight', 'bias', 
                       'gamma', 'beta', 'moving_mean', 'moving_variance', 
                       'depthwise', 'pointwise', 'bn', 'batch_norm',
                       'embedding', 'scale', 'const']
    
    # Keywords that indicate this is NOT a weight tensor (activations, inputs, outputs)
    exclude_keywords = ['input', 'output', 'placeholder', 'identity', 'reshape',
                        'transpose', 'concat', 'split', 'pad', 'slice', 'gather',
                        'serving_default', 'partitionedcall', 'statefulpartitionedcall']
    
    for tensor in tensor_details:
        name = tensor['name'].lower()
        shape = tensor['shape']
        
        # Calculate number of elements in tensor
        num_elements = 1
        for dim in shape:
            num_elements *= int(dim)
        
        # Skip empty or scalar tensors for parameter counting
        if num_elements <= 1:
            continue
            
        # Check if this is explicitly excluded (activation/intermediate tensor)
        is_excluded = any(keyword in name for keyword in exclude_keywords)
        
        # Check if this is a weight/parameter tensor
        is_weight = any(keyword in name for keyword in weight_keywords)
        
        # Count as parameter if it matches weight keywords and isn't excluded
        if is_weight and not is_excluded:
            weight_params += num_elements
            trainable_tensors.append({
                'name': tensor['name'],
                'shape': [int(s) for s in shape],
                'params': int(num_elements),
                'dtype': str(tensor['dtype'])
            })
    
    # Get file size
    file_size = os.path.getsize(model_path)
    
    return {
        'model_name': os.path.basename(model_path),
        'file_size_bytes': file_size,
        'file_size_mb': round(file_size / (1024 * 1024), 2),
        'total_parameters': weight_params,
        'total_parameters_millions': round(weight_params / 1_000_000, 2),
        'num_weight_tensors': len(trainable_tensors),
        'total_tensors': len(tensor_details),
        'weight_tensors': trainable_tensors
    }


def analyze_all_models(directory=None):
    """
    Analyze all TFLite models in the specified directory.
    
    Args:
        directory: Directory to search for .tflite files. Defaults to current directory.
        
    Returns:
        List of analysis results for each model
    """
    if directory is None:
        directory = os.path.dirname(os.path.abspath(__file__))
    
    # Find all tflite files
    tflite_files = glob.glob(os.path.join(directory, "*.tflite"))
    
    if not tflite_files:
        print("No .tflite files found in the directory.")
        return []
    
    results = []
    
    print("=" * 80)
    print("TFLite Model Parameter Analysis")
    print("=" * 80)
    
    for model_path in sorted(tflite_files):
        print(f"\nAnalyzing: {os.path.basename(model_path)}")
        print("-" * 60)
        
        try:
            result = count_model_parameters(model_path)
            results.append(result)
            
            print(f"  File Size:        {result['file_size_mb']} MB")
            print(f"  Total Parameters: {result['total_parameters']:,} ({result['total_parameters_millions']}M)")
            print(f"  Weight Tensors:   {result['num_weight_tensors']}")
            print(f"  Total Tensors:    {result['total_tensors']}")
            
        except Exception as e:
            print(f"  Error analyzing model: {e}")
            results.append({
                'model_name': os.path.basename(model_path),
                'error': str(e)
            })
    
    return results


def save_results(results, output_dir=None):
    """Save analysis results to JSON and CSV files."""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save to JSON
    json_path = os.path.join(output_dir, "model_parameters.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # Save summary to CSV
    csv_path = os.path.join(output_dir, "model_parameters.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Model Name', 'File Size (MB)', 'Total Parameters', 'Parameters (Millions)', 
                        'Weight Tensors', 'Total Tensors'])
        
        for result in results:
            if 'error' not in result:
                writer.writerow([
                    result['model_name'],
                    result['file_size_mb'],
                    result['total_parameters'],
                    result['total_parameters_millions'],
                    result['num_weight_tensors'],
                    result['total_tensors']
                ])
    print(f"Summary saved to: {csv_path}")


def print_summary_table(results):
    """Print a formatted summary table of all models."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Model Name':<50} {'Size (MB)':<12} {'Parameters':<15} {'Millions':<10}")
    print("-" * 80)
    
    total_params = 0
    for result in results:
        if 'error' not in result:
            print(f"{result['model_name']:<50} {result['file_size_mb']:<12} {result['total_parameters']:<15,} {result['total_parameters_millions']:<10}")
            total_params += result['total_parameters']
    
    print("-" * 80)
    print(f"{'TOTAL':<50} {'':<12} {total_params:<15,} {round(total_params/1_000_000, 2):<10}")
    print("=" * 80)


if __name__ == "__main__":
    # Analyze all models in the current directory
    results = analyze_all_models()
    
    if results:
        # Print summary table
        print_summary_table(results)
        
        # Save results to files
        save_results(results)
