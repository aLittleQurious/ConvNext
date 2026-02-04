"""
ONNX Model Analyzer - Calculates bytes per layer for multiple models
Outputs results to JSON file and CSV including encryption cycle estimates
"""

import onnx
import numpy as np
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

# ONNX tensor element type to numpy dtype mapping
ONNX_DTYPE_MAP = {
    1: np.float32,    # FLOAT
    2: np.uint8,      # UINT8
    3: np.int8,       # INT8
    4: np.uint16,     # UINT16
    5: np.int16,      # INT16
    6: np.int32,      # INT32
    7: np.int64,      # INT64
    8: str,           # STRING
    9: np.bool_,      # BOOL
    10: np.float16,   # FLOAT16
    11: np.float64,   # DOUBLE
    12: np.uint32,    # UINT32
    13: np.uint64,    # UINT64
    14: np.complex64, # COMPLEX64
    15: np.complex128,# COMPLEX128
    16: np.float16,   # BFLOAT16 (approximated as float16)
}


def get_onnx_dtype_name(elem_type):
    """Get human-readable dtype name from ONNX element type."""
    dtype_names = {
        1: 'float32', 2: 'uint8', 3: 'int8', 4: 'uint16', 5: 'int16',
        6: 'int32', 7: 'int64', 8: 'string', 9: 'bool', 10: 'float16',
        11: 'float64', 12: 'uint32', 13: 'uint64', 14: 'complex64',
        15: 'complex128', 16: 'bfloat16'
    }
    return dtype_names.get(elem_type, f'unknown({elem_type})')


def get_tensor_size_bytes(shape, elem_type):
    """Calculate tensor size in bytes from shape and element type."""
    if not shape or 0 in shape:
        return 0, 0
    
    num_elements = int(np.prod(shape))
    
    # Get dtype and calculate bytes per element
    dtype = ONNX_DTYPE_MAP.get(elem_type)
    if dtype is not None and dtype != str:
        try:
            bytes_per_element = np.dtype(dtype).itemsize
        except TypeError:
            bytes_per_element = CIPHER_CONFIG.get('analysis_settings', {}).get('default_bytes_per_element', 4)
    else:
        bytes_per_element = CIPHER_CONFIG.get('analysis_settings', {}).get('default_bytes_per_element', 4)
    
    tensor_bytes = num_elements * bytes_per_element
    return int(tensor_bytes), int(num_elements)


def analyze_onnx_model(model_path):
    """
    Analyze an ONNX model and report the number of bytes per layer.
    Returns a dictionary with all analysis results.
    """
    # Get total file size
    total_file_size = os.path.getsize(model_path)
    model_name = os.path.basename(model_path)
    
    print(f"\nModel: {model_name}")
    print(f"Total file size: {total_file_size:,} bytes ({total_file_size / (1024*1024):.2f} MB)")
    print("=" * 80)
    
    # Load ONNX model
    model = onnx.load(model_path)
    graph = model.graph
    
    # Calculate bytes per tensor/layer
    print(f"\n{'Index':<6} {'Name':<50} {'Shape':<25} {'Type':<12} {'Bytes':<15}")
    print("-" * 110)
    
    total_tensor_bytes = 0
    layer_info = []
    total_parameters = 0
    index = 0
    
    # Process initializers (weights/parameters)
    for initializer in graph.initializer:
        name = initializer.name
        shape = list(initializer.dims)
        elem_type = initializer.data_type
        dtype_name = get_onnx_dtype_name(elem_type)
        
        # Get raw data size directly from initializer
        if initializer.raw_data:
            tensor_bytes = len(initializer.raw_data)
            num_elements = int(np.prod(shape)) if shape else 0
        else:
            tensor_bytes, num_elements = get_tensor_size_bytes(shape, elem_type)
        
        total_tensor_bytes += tensor_bytes
        total_parameters += num_elements
        
        # Truncate name if too long for display
        display_name = name[:47] + "..." if len(name) > 50 else name
        shape_str = str(shape)
        
        # Calculate encryption cycles for all configured ciphers
        layer_encryption_cycles = {
            cipher_name: int(tensor_bytes * cycles_per_byte)
            for cipher_name, cycles_per_byte in CIPHER_RATES.items()
        }
        
        layer_info.append({
            'index': index,
            'name': name,
            'shape': shape,
            'dtype': dtype_name,
            'bytes': tensor_bytes,
            'num_parameters': num_elements,
            'tensor_type': 'initializer',
            'encryption_cycles': layer_encryption_cycles
        })
        
        print(f"{index:<6} {display_name:<50} {shape_str:<25} {dtype_name:<12} {tensor_bytes:>12,}")
        index += 1
    
    # Process graph inputs (that are not initializers)
    initializer_names = {init.name for init in graph.initializer}
    for input_tensor in graph.input:
        if input_tensor.name in initializer_names:
            continue  # Skip initializers already processed
        
        name = input_tensor.name
        tensor_type = input_tensor.type.tensor_type
        elem_type = tensor_type.elem_type
        dtype_name = get_onnx_dtype_name(elem_type)
        
        # Extract shape
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            elif dim.dim_param:
                shape.append(1)  # Use 1 for dynamic dimensions
            else:
                shape.append(1)
        
        tensor_bytes, num_elements = get_tensor_size_bytes(shape, elem_type)
        total_tensor_bytes += tensor_bytes
        total_parameters += num_elements
        
        display_name = name[:47] + "..." if len(name) > 50 else name
        shape_str = str(shape)
        
        layer_encryption_cycles = {
            cipher_name: int(tensor_bytes * cycles_per_byte)
            for cipher_name, cycles_per_byte in CIPHER_RATES.items()
        }
        
        layer_info.append({
            'index': index,
            'name': name,
            'shape': shape,
            'dtype': dtype_name,
            'bytes': tensor_bytes,
            'num_parameters': num_elements,
            'tensor_type': 'input',
            'encryption_cycles': layer_encryption_cycles
        })
        
        print(f"{index:<6} {display_name:<50} {shape_str:<25} {dtype_name:<12} {tensor_bytes:>12,}")
        index += 1
    
    # Process graph outputs
    for output_tensor in graph.output:
        name = output_tensor.name
        tensor_type = output_tensor.type.tensor_type
        elem_type = tensor_type.elem_type
        dtype_name = get_onnx_dtype_name(elem_type)
        
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            elif dim.dim_param:
                shape.append(1)
            else:
                shape.append(1)
        
        tensor_bytes, num_elements = get_tensor_size_bytes(shape, elem_type)
        total_tensor_bytes += tensor_bytes
        total_parameters += num_elements
        
        display_name = name[:47] + "..." if len(name) > 50 else name
        shape_str = str(shape)
        
        layer_encryption_cycles = {
            cipher_name: int(tensor_bytes * cycles_per_byte)
            for cipher_name, cycles_per_byte in CIPHER_RATES.items()
        }
        
        layer_info.append({
            'index': index,
            'name': name,
            'shape': shape,
            'dtype': dtype_name,
            'bytes': tensor_bytes,
            'num_parameters': num_elements,
            'tensor_type': 'output',
            'encryption_cycles': layer_encryption_cycles
        })
        
        print(f"{index:<6} {display_name:<50} {shape_str:<25} {dtype_name:<12} {tensor_bytes:>12,}")
        index += 1
    
    print("-" * 110)
    print(f"\nSummary:")
    print(f"  Total tensors: {len(layer_info)}")
    print(f"  Total parameters: {total_parameters:,} ({total_parameters / 1e6:.2f}M)")
    print(f"  Total tensor data: {total_tensor_bytes:,} bytes ({total_tensor_bytes / (1024*1024):.2f} MB)")
    
    # Calculate activation and peak memory usage
    tensors_sorted_by_size = sorted(layer_info, key=lambda x: -x['bytes'])
    
    activation_bytes = tensors_sorted_by_size[0]['bytes'] if tensors_sorted_by_size else 0
    
    top_n = CIPHER_CONFIG.get('analysis_settings', {}).get('top_activations_count', 10)
    top_tensors = tensors_sorted_by_size[:top_n] if len(tensors_sorted_by_size) >= top_n else tensors_sorted_by_size
    peak_memory_bytes = sum(t['bytes'] for t in top_tensors)
    
    print(f"\nMemory Estimates:")
    print(f"  Activation (largest tensor): {activation_bytes:,} bytes ({activation_bytes / (1024*1024):.2f} MB)")
    print(f"  Peak memory (top {len(top_tensors)} tensors): {peak_memory_bytes:,} bytes ({peak_memory_bytes / (1024*1024):.2f} MB)")
    
    # Group by operation type
    print("\n" + "=" * 80)
    print("Breakdown by Layer Type (based on tensor names):")
    print("-" * 80)
    
    layer_types = {}
    for info in layer_info:
        name = info['name'].lower()
        
        if 'conv' in name or 'depthwise' in name:
            layer_type = 'Convolution'
        elif 'batch' in name or 'bn' in name:
            layer_type = 'BatchNorm'
        elif 'dense' in name or 'fc' in name or 'matmul' in name or 'gemm' in name:
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
    
    # Calculate encryption cycles
    total_encryption_cycles = {
        cipher_name: int(total_tensor_bytes * cycles_per_byte)
        for cipher_name, cycles_per_byte in CIPHER_RATES.items()
    }
    
    activation_encryption_cycles = {
        cipher_name: int(activation_bytes * cycles_per_byte)
        for cipher_name, cycles_per_byte in CIPHER_RATES.items()
    }
    
    peak_memory_encryption_cycles = {
        cipher_name: int(peak_memory_bytes * cycles_per_byte)
        for cipher_name, cycles_per_byte in CIPHER_RATES.items()
    }
    
    print("\n" + "=" * 80)
    print("Encryption Cycle Estimates:")
    print("-" * 80)
    print("Total (all tensors):")
    for cipher_name, cycles_per_byte in CIPHER_RATES.items():
        total_cycles = total_encryption_cycles[cipher_name]
        print(f"  {cipher_name} ({cycles_per_byte:,} cycles/byte): {total_cycles:>20,} cycles")
    
    print("\nActivation (largest tensor):")
    for cipher_name, cycles_per_byte in CIPHER_RATES.items():
        cycles = activation_encryption_cycles[cipher_name]
        print(f"  {cipher_name} ({cycles_per_byte:,} cycles/byte): {cycles:>20,} cycles")
    
    print(f"\nPeak Memory (top {len(top_tensors)} tensors):")
    for cipher_name, cycles_per_byte in CIPHER_RATES.items():
        cycles = peak_memory_encryption_cycles[cipher_name]
        print(f"  {cipher_name} ({cycles_per_byte:,} cycles/byte): {cycles:>20,} cycles")
    
    # Build result dictionary
    result = {
        'model_name': model_name,
        'model_path': model_path,
        'total_file_size_bytes': total_file_size,
        'total_tensor_bytes': int(total_tensor_bytes),
        'total_tensor_mb': round(total_tensor_bytes / (1024 * 1024), 2),
        'total_parameters': int(total_parameters),
        'total_parameters_millions': round(total_parameters / 1e6, 2),
        'total_tensors': len(layer_info),
        'activation_bytes': int(activation_bytes),
        'activation_mb': round(activation_bytes / (1024 * 1024), 2),
        'peak_memory_bytes': int(peak_memory_bytes),
        'peak_memory_mb': round(peak_memory_bytes / (1024 * 1024), 2),
        'cipher_config': CIPHER_CONFIG,
        'encryption_cycles_total': {
            cipher_name: {
                'cycles_per_byte': cycles_per_byte,
                'total_cycles': total_encryption_cycles[cipher_name]
            }
            for cipher_name, cycles_per_byte in CIPHER_RATES.items()
        },
        'encryption_cycles_activation': {
            cipher_name: {
                'cycles_per_byte': cycles_per_byte,
                'cycles': activation_encryption_cycles[cipher_name]
            }
            for cipher_name, cycles_per_byte in CIPHER_RATES.items()
        },
        'encryption_cycles_peak_memory': {
            cipher_name: {
                'cycles_per_byte': cycles_per_byte,
                'cycles': peak_memory_encryption_cycles[cipher_name]
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


def find_onnx_models(directory):
    """Find all .onnx files in the given directory."""
    pattern = os.path.join(directory, "*.onnx")
    return glob.glob(pattern)


def main():
    # Directory containing ONNX models
    models_dir = r"c:\Users\kevin\source\repos\CompArch\ConvNext"
    
    # Find all ONNX models
    model_files = find_onnx_models(models_dir)
    
    if not model_files:
        print(f"No .onnx files found in {models_dir}")
        return
    
    print(f"Found {len(model_files)} ONNX model(s):")
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
            result = analyze_onnx_model(model_path)
            all_results['models'][model_name] = result
        except Exception as e:
            print(f"\nError analyzing {model_name}: {e}")
            all_results['models'][model_name] = {'error': str(e)}
    
    # Save results to JSON file
    output_file = os.path.join(models_dir, "onnx_model_analysis_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    # Save results to CSV file
    csv_output_file = os.path.join(models_dir, "onnx_model_analysis_results.csv")
    with open(csv_output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        cipher_headers = [
            f"{cipher_name} Cycles ({cycles_per_byte} cycles/byte)"
            for cipher_name, cycles_per_byte in CIPHER_RATES.items()
        ]
        
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
    
    # Create summary CSV
    summary_csv_file = os.path.join(models_dir, "onnx_model_analysis_summary.csv")
    with open(summary_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        total_cipher_headers = [
            f"{cipher_name} Total Cycles ({cycles_per_byte} c/b)"
            for cipher_name, cycles_per_byte in CIPHER_RATES.items()
        ]
        activation_cipher_headers = [
            f"{cipher_name} Activation Cycles ({cycles_per_byte} c/b)"
            for cipher_name, cycles_per_byte in CIPHER_RATES.items()
        ]
        peak_memory_cipher_headers = [
            f"{cipher_name} Peak Memory Cycles ({cycles_per_byte} c/b)"
            for cipher_name, cycles_per_byte in CIPHER_RATES.items()
        ]
        
        writer.writerow([
            'Model Name',
            'Total File Size (bytes)',
            'Total File Size (MB)',
            'Total Parameters',
            'Total Parameters (M)',
            'Total Tensor Bytes',
            'Total Tensor MB',
            'Total Tensors',
            'Activation Bytes',
            'Activation MB',
            'Peak Memory Bytes',
            'Peak Memory MB'
        ] + total_cipher_headers + activation_cipher_headers + peak_memory_cipher_headers)
        
        for model_name, model_data in all_results['models'].items():
            if 'error' in model_data:
                error_row = [model_name, 'ERROR'] + [''] * (10 + len(CIPHER_RATES) * 3)
                writer.writerow(error_row)
                continue
            
            total_cipher_cycles = [
                model_data['encryption_cycles_total'][cipher_name]['total_cycles']
                for cipher_name in CIPHER_RATES.keys()
            ]
            activation_cipher_cycles = [
                model_data['encryption_cycles_activation'][cipher_name]['cycles']
                for cipher_name in CIPHER_RATES.keys()
            ]
            peak_memory_cipher_cycles = [
                model_data['encryption_cycles_peak_memory'][cipher_name]['cycles']
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
                model_data['total_tensors'],
                model_data['activation_bytes'],
                model_data['activation_mb'],
                model_data['peak_memory_bytes'],
                model_data['peak_memory_mb']
            ] + total_cipher_cycles + activation_cipher_cycles + peak_memory_cipher_cycles)
    
    print("\n" + "=" * 80)
    print(f"Results saved to:")
    print(f"  JSON: {output_file}")
    print(f"  CSV (detailed): {csv_output_file}")
    print(f"  CSV (summary): {summary_csv_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
