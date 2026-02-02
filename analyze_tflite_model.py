"""
TFLite Model Analyzer - Calculates bytes per layer for multiple models
Outputs results to JSON file and CSV including encryption cycle estimates
"""

import tensorflow as tf
import os
import json
import csv
import glob
from datetime import datetime

# Encryption cipher cycle rates (cycles per byte)
CIPHER_RATES = {
    'grain128': 16250,
    'chacha20_poly1305': 170,

}


def analyze_tflite_model(model_path):
    """
    Analyze a TFLite model and report the number of bytes per layer.
    Returns a dictionary with all analysis results.
    """
    # Get total file size
    total_file_size = os.path.getsize(model_path)
    model_name = os.path.basename(model_path)
    
    print(f"\nModel: {model_name}")
    print(f"Total file size: {total_file_size:,} bytes ({total_file_size / (1024*1024):.2f} MB)")
    print("=" * 80)
    
    # Load interpreter to analyze the model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get tensor details
    tensor_details = interpreter.get_tensor_details()
    
    # Calculate bytes per tensor/layer
    print(f"\n{'Index':<6} {'Name':<50} {'Shape':<25} {'Type':<12} {'Bytes':<15}")
    print("-" * 110)
    
    total_tensor_bytes = 0
    layer_info = []
    
    for tensor in tensor_details:
        index = tensor['index']
        name = tensor['name']
        shape = tensor['shape']
        dtype = tensor['dtype']
        
        # Calculate tensor size in bytes
        # Convert to Python int to avoid numpy int32 overflow
        num_elements = 1
        for dim in shape:
            num_elements *= int(dim)
        
        # Get bytes per element based on dtype
        dtype_bytes = {
            'float32': 4,
            'float16': 2,
            'int32': 4,
            'int16': 2,
            'int8': 1,
            'uint8': 1,
            'int64': 8,
            'bool': 1,
        }
        
        dtype_name = dtype.__name__ if hasattr(dtype, '__name__') else str(dtype)
        bytes_per_element = dtype_bytes.get(dtype_name, 4)  # Default to 4 bytes
        tensor_bytes = int(num_elements * bytes_per_element)
        total_tensor_bytes += tensor_bytes
        
        # Determine if this is a weight tensor (constant) or activation tensor
        # Weight tensors typically have names containing 'constant', 'kernel', 'bias', 'weight'
        # or are small shape tensors used for reshaping
        name_lower = name.lower()
        is_weight = ('constant' in name_lower or 'kernel' in name_lower or 
                     'bias' in name_lower or 'weight' in name_lower or
                     'arith.constant' in name_lower)
        
        # Truncate name if too long for display
        display_name = name[:47] + "..." if len(name) > 50 else name
        shape_str = str(list(shape))
        
        layer_info.append({
            'index': int(index),
            'name': name,
            'shape': [int(d) for d in shape],  # Convert to native Python int
            'dtype': dtype_name,
            'bytes': int(tensor_bytes),
            'is_weight': is_weight,
            'encryption_cycles': {
                'grain128': int(tensor_bytes * CIPHER_RATES['grain128']),
                'chacha20_poly1305': int(tensor_bytes * CIPHER_RATES['chacha20_poly1305'])
            }
        })
        
        print(f"{index:<6} {display_name:<50} {shape_str:<25} {dtype_name:<12} {tensor_bytes:>12,}")
    
    print("-" * 110)
    print(f"\nSummary:")
    print(f"  Total tensors: {len(tensor_details)}")
    print(f"  Total tensor data: {total_tensor_bytes:,} bytes ({total_tensor_bytes / (1024*1024):.2f} MB)")
    
    # Calculate peak memory usage
    # Separate weight tensors from activation tensors
    weight_bytes = sum(info['bytes'] for info in layer_info if info['is_weight'])
    activation_bytes = sum(info['bytes'] for info in layer_info if not info['is_weight'])
    
    # Find the largest activation tensors (top N that might be concurrent)
    activation_tensors = [info for info in layer_info if not info['is_weight']]
    activation_tensors_sorted = sorted(activation_tensors, key=lambda x: -x['bytes'])
    
    # Peak memory estimate: weights (always in memory) + largest concurrent activations
    # Typically, during inference, you need input + output of current layer + possibly skip connections
    # A conservative estimate is weights + top 5-10 largest activation tensors
    top_activations = activation_tensors_sorted[:10] if len(activation_tensors_sorted) >= 10 else activation_tensors_sorted
    peak_activation_estimate = sum(t['bytes'] for t in top_activations)
    
    # Also calculate max single activation (minimum activation memory needed)
    max_single_activation = activation_tensors_sorted[0]['bytes'] if activation_tensors_sorted else 0
    
    # Peak memory estimates
    peak_memory_conservative = weight_bytes + peak_activation_estimate  # Weights + top 10 activations
    peak_memory_minimum = weight_bytes + max_single_activation  # Weights + largest single activation
    
    print(f"\nMemory Analysis:")
    print(f"  Weight tensors: {weight_bytes:,} bytes ({weight_bytes / (1024*1024):.2f} MB)")
    print(f"  Activation tensors: {activation_bytes:,} bytes ({activation_bytes / (1024*1024):.2f} MB)")
    print(f"  Largest single activation: {max_single_activation:,} bytes ({max_single_activation / (1024*1024):.2f} MB)")
    print(f"\nEstimated Peak Memory Usage:")
    print(f"  Minimum (weights + max activation): {peak_memory_minimum:,} bytes ({peak_memory_minimum / (1024*1024):.2f} MB)")
    print(f"  Conservative (weights + top 10 activations): {peak_memory_conservative:,} bytes ({peak_memory_conservative / (1024*1024):.2f} MB)")
    
    # Group by operation type (based on common naming conventions)
    print("\n" + "=" * 80)
    print("Breakdown by Layer Type (based on tensor names):")
    print("-" * 80)
    
    layer_types = {}
    for info in layer_info:
        name = info['name'].lower()
        
        # Categorize based on common layer name patterns
        if 'conv' in name or 'depthwise' in name:
            layer_type = 'Convolution'
        elif 'batch' in name or 'bn' in name:
            layer_type = 'BatchNorm'
        elif 'dense' in name or 'fc' in name or 'matmul' in name:
            layer_type = 'Dense/FC'
        elif 'add' in name:
            layer_type = 'Add'
        elif 'relu' in name or 'activation' in name or 'gelu' in name:
            layer_type = 'Activation'
        elif 'pool' in name:
            layer_type = 'Pooling'
        elif 'norm' in name or 'layer_norm' in name:
            layer_type = 'LayerNorm'
        elif 'reshape' in name or 'transpose' in name:
            layer_type = 'Reshape/Transpose'
        elif 'input' in name:
            layer_type = 'Input'
        elif 'output' in name or 'identity' in name:
            layer_type = 'Output'
        else:
            layer_type = 'Other'
        
        if layer_type not in layer_types:
            layer_types[layer_type] = {'count': 0, 'bytes': 0}
        layer_types[layer_type]['count'] += 1
        layer_types[layer_type]['bytes'] += info['bytes']
    
    for layer_type, stats in sorted(layer_types.items(), key=lambda x: -x[1]['bytes']):
        pct = (stats['bytes'] / total_tensor_bytes * 100) if total_tensor_bytes > 0 else 0
        print(f"  {layer_type:<20} Count: {stats['count']:<6} Bytes: {stats['bytes']:>15,} ({pct:>6.2f}%)")
    
    # Calculate total encryption cycles
    total_grain128_cycles = total_tensor_bytes * CIPHER_RATES['grain128']
    total_chacha20_cycles = total_tensor_bytes * CIPHER_RATES['chacha20_poly1305']
    
    print("\n" + "=" * 80)
    print("Encryption Cycle Estimates:")
    print("-" * 80)
    print(f"  Grain128 (16,250 cycles/byte):      {total_grain128_cycles:>20,} cycles")
    print(f"  ChaCha20+Poly1305 (170 cycles/byte): {total_chacha20_cycles:>20,} cycles")
    
    # Build result dictionary
    result = {
        'model_name': model_name,
        'model_path': model_path,
        'total_file_size_bytes': total_file_size,
        'total_tensor_bytes': int(total_tensor_bytes),
        'total_tensor_mb': round(total_tensor_bytes / (1024 * 1024), 2),
        'total_tensors': len(tensor_details),
        'memory_analysis': {
            'weight_bytes': int(weight_bytes),
            'weight_mb': round(weight_bytes / (1024 * 1024), 2),
            'activation_bytes': int(activation_bytes),
            'activation_mb': round(activation_bytes / (1024 * 1024), 2),
            'largest_activation_bytes': int(max_single_activation),
            'largest_activation_mb': round(max_single_activation / (1024 * 1024), 2),
            'peak_memory_minimum_bytes': int(peak_memory_minimum),
            'peak_memory_minimum_mb': round(peak_memory_minimum / (1024 * 1024), 2),
            'peak_memory_conservative_bytes': int(peak_memory_conservative),
            'peak_memory_conservative_mb': round(peak_memory_conservative / (1024 * 1024), 2)
        },
        'encryption_cycles_total': {
            'grain128': {
                'cycles_per_byte': CIPHER_RATES['grain128'],
                'total_cycles': int(total_grain128_cycles)
            },
            'chacha20_poly1305': {
                'cycles_per_byte': CIPHER_RATES['chacha20_poly1305'],
                'total_cycles': int(total_chacha20_cycles)
            }
        },
        'layers': layer_info,
        'layer_type_summary': {
            layer_type: {
                'count': stats['count'],
                'bytes': stats['bytes'],
                'percentage': round((stats['bytes'] / total_tensor_bytes * 100), 2) if total_tensor_bytes > 0 else 0,
                'encryption_cycles': {
                    'grain128': stats['bytes'] * CIPHER_RATES['grain128'],
                    'chacha20_poly1305': stats['bytes'] * CIPHER_RATES['chacha20_poly1305']
                }
            }
            for layer_type, stats in layer_types.items()
        }
    }
    
    return result


def find_tflite_models(directory):
    """Find all .tflite files in the given directory."""
    pattern = os.path.join(directory, "*.tflite")
    return glob.glob(pattern)


def main():
    # Directory containing TFLite models
    models_dir = r"c:\Users\kevin\source\repos\CompArch\ConvNext"
    
    # Find all TFLite models
    model_files = find_tflite_models(models_dir)
    
    if not model_files:
        print(f"No .tflite files found in {models_dir}")
        return
    
    print(f"Found {len(model_files)} TFLite model(s):")
    for f in model_files:
        print(f"  - {os.path.basename(f)}")
    
    # Analyze each model
    all_results = {
        'analysis_timestamp': datetime.now().isoformat(),
        'models_directory': models_dir,
        'total_models_analyzed': len(model_files),
        'models': {}
    }
    
    for model_path in model_files:
        model_name = os.path.basename(model_path)
        try:
            result = analyze_tflite_model(model_path)
            all_results['models'][model_name] = result
        except Exception as e:
            print(f"\nError analyzing {model_name}: {e}")
            all_results['models'][model_name] = {'error': str(e)}
    
    # Save results to JSON file
    output_file = os.path.join(models_dir, "model_analysis_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    # Save results to CSV file
    csv_output_file = os.path.join(models_dir, "model_analysis_results.csv")
    with open(csv_output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'Model Name',
            'Layer Index',
            'Layer Name',
            'Shape',
            'Data Type',
            'Bytes',
            'MB',
            'Grain128 Cycles (16250 cycles/byte)',
            'ChaCha20+Poly1305 Cycles (170 cycles/byte)'
        ])
        
        # Write data for each model
        for model_name, model_data in all_results['models'].items():
            if 'error' in model_data:
                writer.writerow([model_name, 'ERROR', model_data['error'], '', '', '', '', '', ''])
                continue
            
            for layer in model_data['layers']:
                writer.writerow([
                    model_name,
                    layer['index'],
                    layer['name'],
                    str(layer['shape']),
                    layer['dtype'],
                    layer['bytes'],
                    round(layer['bytes'] / (1024 * 1024), 6),
                    layer['encryption_cycles']['grain128'],
                    layer['encryption_cycles']['chacha20_poly1305']
                ])
            
            # Add a summary row for each model
            writer.writerow([])
            writer.writerow([
                f"{model_name} - TOTAL",
                '',
                'TOTAL',
                '',
                '',
                model_data['total_tensor_bytes'],
                round(model_data['total_tensor_bytes'] / (1024 * 1024), 2),
                model_data['encryption_cycles_total']['grain128']['total_cycles'],
                model_data['encryption_cycles_total']['chacha20_poly1305']['total_cycles']
            ])
            writer.writerow([])
    
    # Also create a summary CSV
    summary_csv_file = os.path.join(models_dir, "model_analysis_summary.csv")
    with open(summary_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'Model Name',
            'Total File Size (bytes)',
            'Total File Size (MB)',
            'Total Tensor Bytes',
            'Total Tensor MB',
            'Total Tensors',
            'Weight Bytes',
            'Weight MB',
            'Activation Bytes',
            'Activation MB',
            'Peak Memory Min (bytes)',
            'Peak Memory Min (MB)',
            'Peak Memory Conservative (bytes)',
            'Peak Memory Conservative (MB)',
            'Grain128 Total Cycles (on Tensor Bytes)',
            'ChaCha20+Poly1305 Total Cycles (on Tensor Bytes)'
        ])
        
        # Write summary for each model
        for model_name, model_data in all_results['models'].items():
            if 'error' in model_data:
                writer.writerow([model_name, 'ERROR', '', '', '', '', '', '', '', '', '', '', '', '', '', ''])
                continue
            
            mem = model_data['memory_analysis']
            writer.writerow([
                model_name,
                model_data['total_file_size_bytes'],
                round(model_data['total_file_size_bytes'] / (1024 * 1024), 2),
                model_data['total_tensor_bytes'],
                round(model_data['total_tensor_bytes'] / (1024 * 1024), 2),
                model_data['total_tensors'],
                mem['weight_bytes'],
                mem['weight_mb'],
                mem['activation_bytes'],
                mem['activation_mb'],
                mem['peak_memory_minimum_bytes'],
                mem['peak_memory_minimum_mb'],
                mem['peak_memory_conservative_bytes'],
                mem['peak_memory_conservative_mb'],
                model_data['encryption_cycles_total']['grain128']['total_cycles'],
                model_data['encryption_cycles_total']['chacha20_poly1305']['total_cycles']
            ])
    
    print("\n" + "=" * 80)
    print(f"Results saved to:")
    print(f"  JSON: {output_file}")
    print(f"  CSV (detailed): {csv_output_file}")
    print(f"  CSV (summary): {summary_csv_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
