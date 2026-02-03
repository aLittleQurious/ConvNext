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


def load_cipher_config(config_path=None):
    """
    Load cipher configuration from JSON file.
    Returns cipher rates dictionary and config metadata.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "cipher_config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    else:
        # Default configuration if file doesn't exist
        return {
            "ciphers": {
                "grain128": 16250,
                "chacha20_poly1305": 170
            },
            "analysis_settings": {
                "top_activations_count": 10,
                "default_bytes_per_element": 4
            }
        }


# Load cipher configuration
CIPHER_CONFIG = load_cipher_config()
CIPHER_RATES = CIPHER_CONFIG.get('ciphers', {})


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
    
    total_parameters = 0
    
    for tensor in tensor_details:
        index = tensor['index']
        name = tensor['name']
        shape = tensor['shape']
        dtype = tensor['dtype']
        dtype_name = dtype.__name__ if hasattr(dtype, '__name__') else str(dtype)
        
        # Get actual tensor data to determine bytes and parameters
        # This approach matches detailer_getter.py exactly
        try:
            data = interpreter.get_tensor(index)
            tensor_bytes = int(data.nbytes)
            num_elements = int(data.size)
            total_tensor_bytes += tensor_bytes
            total_parameters += num_elements
        except ValueError:
            # Skip tensors that cannot be retrieved (same as detailer_getter.py)
            continue
        
        # Truncate name if too long for display
        display_name = name[:47] + "..." if len(name) > 50 else name
        shape_str = str(list(shape))
        
        # Calculate encryption cycles for all configured ciphers
        layer_encryption_cycles = {
            cipher_name: int(tensor_bytes * cycles_per_byte)
            for cipher_name, cycles_per_byte in CIPHER_RATES.items()
        }
        
        layer_info.append({
            'index': int(index),
            'name': name,
            'shape': [int(d) for d in shape],  # Convert to native Python int
            'dtype': dtype_name,
            'bytes': tensor_bytes,
            'num_parameters': num_elements,
            'encryption_cycles': layer_encryption_cycles
        })
        
        print(f"{index:<6} {display_name:<50} {shape_str:<25} {dtype_name:<12} {tensor_bytes:>12,}")
    
    print("-" * 110)
    print(f"\nSummary:")
    print(f"  Total tensors: {len(tensor_details)}")
    print(f"  Total parameters: {total_parameters:,} ({total_parameters / 1e6:.2f}M)")
    print(f"  Total tensor data: {total_tensor_bytes:,} bytes ({total_tensor_bytes / (1024*1024):.2f} MB)")
    
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
    
    # Calculate total encryption cycles for all configured ciphers
    total_encryption_cycles = {
        cipher_name: int(total_tensor_bytes * cycles_per_byte)
        for cipher_name, cycles_per_byte in CIPHER_RATES.items()
    }
    
    print("\n" + "=" * 80)
    print("Encryption Cycle Estimates:")
    print("-" * 80)
    for cipher_name, cycles_per_byte in CIPHER_RATES.items():
        total_cycles = total_encryption_cycles[cipher_name]
        print(f"  {cipher_name} ({cycles_per_byte:,} cycles/byte): {total_cycles:>20,} cycles")
    
    # Build result dictionary
    result = {
        'model_name': model_name,
        'model_path': model_path,
        'total_file_size_bytes': total_file_size,
        'total_tensor_bytes': int(total_tensor_bytes),
        'total_tensor_mb': round(total_tensor_bytes / (1024 * 1024), 2),
        'total_parameters': int(total_parameters),
        'total_parameters_millions': round(total_parameters / 1e6, 2),
        'total_tensors': len(tensor_details),
        'cipher_config': CIPHER_CONFIG,
        'encryption_cycles_total': {
            cipher_name: {
                'cycles_per_byte': cycles_per_byte,
                'total_cycles': total_encryption_cycles[cipher_name]
            }
            for cipher_name, cycles_per_byte in CIPHER_RATES.items()
        },
        'layers': layer_info,
        'layer_type_summary': {
            layer_type: {
                'count': stats['count'],
                'bytes': stats['bytes'],
                'percentage': round((stats['bytes'] / total_tensor_bytes * 100), 2) if total_tensor_bytes > 0 else 0,
                'encryption_cycles': {
                    cipher_name: stats['bytes'] * cycles_per_byte
                    for cipher_name, cycles_per_byte in CIPHER_RATES.items()
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
        
        # Build dynamic header based on configured ciphers
        cipher_headers = [
            f"{cipher_name} Cycles ({cycles_per_byte} cycles/byte)"
            for cipher_name, cycles_per_byte in CIPHER_RATES.items()
        ]
        
        # Write header
        writer.writerow([
            'Model Name',
            'Layer Index',
            'Layer Name',
            'Shape',
            'Data Type',
            'Parameters',
            'Bytes',
            'MB'
        ] + cipher_headers)
        
        # Write data for each model
        for model_name, model_data in all_results['models'].items():
            if 'error' in model_data:
                error_row = [model_name, 'ERROR', model_data['error']] + [''] * (5 + len(CIPHER_RATES))
                writer.writerow(error_row)
                continue
            
            for layer in model_data['layers']:
                cipher_cycles = [layer['encryption_cycles'].get(cipher_name, 0) for cipher_name in CIPHER_RATES.keys()]
                writer.writerow([
                    model_name,
                    layer['index'],
                    layer['name'],
                    str(layer['shape']),
                    layer['dtype'],
                    layer['num_parameters'],
                    layer['bytes'],
                    round(layer['bytes'] / (1024 * 1024), 6)
                ] + cipher_cycles)
            
            # Add a summary row for each model
            writer.writerow([])
            total_cipher_cycles = [
                model_data['encryption_cycles_total'][cipher_name]['total_cycles']
                for cipher_name in CIPHER_RATES.keys()
            ]
            writer.writerow([
                f"{model_name} - TOTAL",
                '',
                'TOTAL',
                '',
                '',
                model_data['total_parameters'],
                model_data['total_tensor_bytes'],
                round(model_data['total_tensor_bytes'] / (1024 * 1024), 2)
            ] + total_cipher_cycles)
            writer.writerow([])
    
    # Also create a summary CSV
    summary_csv_file = os.path.join(models_dir, "model_analysis_summary.csv")
    with open(summary_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Build dynamic cipher headers for summary
        summary_cipher_headers = [
            f"{cipher_name} Total Cycles ({cycles_per_byte} cycles/byte)"
            for cipher_name, cycles_per_byte in CIPHER_RATES.items()
        ]
        
        # Write header
        writer.writerow([
            'Model Name',
            'Total File Size (bytes)',
            'Total File Size (MB)',
            'Total Parameters',
            'Total Parameters (M)',
            'Total Tensor Bytes',
            'Total Tensor MB',
            'Total Tensors'
        ] + summary_cipher_headers)
        
        # Write summary for each model
        for model_name, model_data in all_results['models'].items():
            if 'error' in model_data:
                error_row = [model_name, 'ERROR'] + [''] * (6 + len(CIPHER_RATES))
                writer.writerow(error_row)
                continue
            
            summary_cipher_cycles = [
                model_data['encryption_cycles_total'][cipher_name]['total_cycles']
                for cipher_name in CIPHER_RATES.keys()
            ]
            writer.writerow([
                model_name,
                model_data['total_file_size_bytes'],
                round(model_data['total_file_size_bytes'] / (1024 * 1024), 2),
                model_data['total_parameters'],
                model_data['total_parameters_millions'],
                model_data['total_tensor_bytes'],
                round(model_data['total_tensor_bytes'] / (1024 * 1024), 2),
                model_data['total_tensors']
            ] + summary_cipher_cycles)
    
    print("\n" + "=" * 80)
    print(f"Results saved to:")
    print(f"  JSON: {output_file}")
    print(f"  CSV (detailed): {csv_output_file}")
    print(f"  CSV (summary): {summary_csv_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
