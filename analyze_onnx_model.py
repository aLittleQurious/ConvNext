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

# Directory configuration
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


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

# Extract cipher parameters (handle both old flat format and new nested format)
_raw_ciphers = CIPHER_CONFIG.get('ciphers', {})
CIPHER_RATES = {}
for cipher_name, cipher_data in _raw_ciphers.items():
    if isinstance(cipher_data, dict):
        CIPHER_RATES[cipher_name] = {
            'cycles_per_byte': cipher_data.get('cycles_per_byte', 0),
            'power_per_byte_watts': cipher_data.get('power_per_byte_watts', 0)
        }
    else:
        # Legacy format: just cycles_per_byte as a number
        CIPHER_RATES[cipher_name] = {
            'cycles_per_byte': cipher_data,
            'power_per_byte_watts': 0
        }

# Processor configuration
PROCESSOR_FREQ_MHZ = CIPHER_CONFIG.get('processor', {}).get('frequency_mhz', 200)
PROCESSOR_FREQ_HZ = PROCESSOR_FREQ_MHZ * 1_000_000

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


def calculate_encryption_energy_per_op(cycles_per_byte, power_per_byte_watts):
    """
    Calculate encryption energy per operation in picojoules (pJ/operation).
    pJ/op = (Watts/byte) * (Cycles/byte) / (Processor Frequency in Hz) * 1e12
    
    Energy_per_op = Power_per_byte * Time_per_byte
    Time_per_byte = Cycles_per_byte / Frequency
    
    Args:
        cycles_per_byte: Encryption performance cost in cycles per byte
        power_per_byte_watts: Power consumption in Watts per byte
    
    Returns:
        Energy per operation in picojoules (pJ/op)
    """
    time_per_byte_seconds = cycles_per_byte / PROCESSOR_FREQ_HZ
    energy_per_op_joules = power_per_byte_watts * time_per_byte_seconds
    energy_per_op_pj = energy_per_op_joules * 1e12  # Convert to picojoules
    return energy_per_op_pj


def calculate_total_encryption_energy(total_bytes, energy_per_op_pj):
    """
    Calculate total encryption energy in picojoules.
    
    Args:
        total_bytes: Number of bytes to encrypt
        energy_per_op_pj: Energy per operation in picojoules
    
    Returns:
        Total energy in picojoules (pJ)
    """
    return total_bytes * energy_per_op_pj


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


def analyze_onnx_model(model_path, model_name=None):
    """
    Analyze an ONNX model and report the number of bytes per layer.
    Returns a dictionary with all analysis results.
    
    Args:
        model_path: Path to the .onnx file
        model_name: Optional model name (defaults to filename if not provided)
    """
    # Get total file size (including external data files if present)
    total_file_size = os.path.getsize(model_path)
    
    # Check for external data files (.data or .bin) in the same directory
    model_dir = os.path.dirname(model_path)
    for ext in ['.data', '.onnx', '.bin']:
        external_data_path = os.path.join(model_dir, os.path.splitext(os.path.basename(model_path))[0] + ext)
        if os.path.exists(external_data_path):
            total_file_size += os.path.getsize(external_data_path)
    
    if model_name is None:
        model_name = os.path.basename(model_path)
    
    print(f"\nModel: {model_name}")
    print(f"Total file/model size: {total_file_size:,} bytes ({total_file_size / (1024*1024):.2f} MB)")
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
        
        # Calculate encryption cycles and energy for all configured ciphers

        layer_encryption_cycles = {}
        layer_encryption_energy_pj = {}
        for cipher_name, cipher_params in CIPHER_RATES.items():
            cycles = int(tensor_bytes * cipher_params['cycles_per_byte'])
            energy_per_op = calculate_encryption_energy_per_op(
                cipher_params['cycles_per_byte'], 
                cipher_params['power_per_byte_watts']
            )
            total_energy = calculate_total_encryption_energy(tensor_bytes, energy_per_op)
            layer_encryption_cycles[cipher_name] = cycles
            layer_encryption_energy_pj[cipher_name] = total_energy
        
        layer_info.append({
            'index': index,
            'name': name,
            'shape': shape,
            'dtype': dtype_name,
            'bytes': tensor_bytes,
            'num_parameters': num_elements,
            'tensor_type': 'initializer',
            'encryption_cycles': layer_encryption_cycles,
            'encryption_energy_pj': layer_encryption_energy_pj
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
        
        layer_encryption_cycles = {}
        layer_encryption_energy_pj = {}
        for cipher_name, cipher_params in CIPHER_RATES.items():
            cycles = int(tensor_bytes * cipher_params['cycles_per_byte'])
            energy_per_op = calculate_encryption_energy_per_op(
                cipher_params['cycles_per_byte'], 
                cipher_params['power_per_byte_watts']
            )
            total_energy = calculate_total_encryption_energy(tensor_bytes, energy_per_op)
            layer_encryption_cycles[cipher_name] = cycles
            layer_encryption_energy_pj[cipher_name] = total_energy
        
        layer_info.append({
            'index': index,
            'name': name,
            'shape': shape,
            'dtype': dtype_name,
            'bytes': tensor_bytes,
            'num_parameters': num_elements,
            'tensor_type': 'input',
            'encryption_cycles': layer_encryption_cycles,
            'encryption_energy_pj': layer_encryption_energy_pj
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
        
        layer_encryption_cycles = {}
        layer_encryption_energy_pj = {}
        for cipher_name, cipher_params in CIPHER_RATES.items():
            cycles = int(tensor_bytes * cipher_params['cycles_per_byte'])
            energy_per_op = calculate_encryption_energy_per_op(
                cipher_params['cycles_per_byte'], 
                cipher_params['power_per_byte_watts']
            )
            total_energy = calculate_total_encryption_energy(tensor_bytes, energy_per_op)
            layer_encryption_cycles[cipher_name] = cycles
            layer_encryption_energy_pj[cipher_name] = total_energy
        
        layer_info.append({
            'index': index,
            'name': name,
            'shape': shape,
            'dtype': dtype_name,
            'bytes': tensor_bytes,
            'num_parameters': num_elements,
            'tensor_type': 'output',
            'encryption_cycles': layer_encryption_cycles,
            'encryption_energy_pj': layer_encryption_energy_pj
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
    
    # Calculate encryption cycles and energy (in pJ)
    total_encryption_cycles = {}
    total_encryption_energy_pj = {}
    energy_per_op_pj = {}  # Store energy per operation for each cipher
    activation_encryption_cycles = {}
    activation_encryption_energy_pj = {}
    peak_memory_encryption_cycles = {}
    peak_memory_encryption_energy_pj = {}
    
    for cipher_name, cipher_params in CIPHER_RATES.items():
        cpb = cipher_params['cycles_per_byte']
        ppb = cipher_params['power_per_byte_watts']
        
        # Calculate energy per operation (pJ/op)
        energy_per_op_pj[cipher_name] = calculate_encryption_energy_per_op(cpb, ppb)
        
        total_encryption_cycles[cipher_name] = int(total_tensor_bytes * cpb)
        total_encryption_energy_pj[cipher_name] = calculate_total_encryption_energy(total_tensor_bytes, energy_per_op_pj[cipher_name])
        
        activation_encryption_cycles[cipher_name] = int(activation_bytes * cpb)
        activation_encryption_energy_pj[cipher_name] = calculate_total_encryption_energy(activation_bytes, energy_per_op_pj[cipher_name])
        
        peak_memory_encryption_cycles[cipher_name] = int(peak_memory_bytes * cpb)
        peak_memory_encryption_energy_pj[cipher_name] = calculate_total_encryption_energy(peak_memory_bytes, energy_per_op_pj[cipher_name])
    
    print("\n" + "=" * 80)
    print(f"Encryption Estimates (Processor: {PROCESSOR_FREQ_MHZ} MHz):")
    print("-" * 80)
    
    print("Energy per operation (pJ/op):")
    for cipher_name in CIPHER_RATES.keys():
        print(f"  {cipher_name}: {energy_per_op_pj[cipher_name]:.6f} pJ/op")
    
    print("\nTotal (all tensors):")
    for cipher_name, cipher_params in CIPHER_RATES.items():
        cycles = total_encryption_cycles[cipher_name]
        energy = total_encryption_energy_pj[cipher_name]
        print(f"  {cipher_name}: {cycles:>20,} cycles, {energy:.6e} pJ")
    
    print("\nActivation (largest tensor):")
    for cipher_name, cipher_params in CIPHER_RATES.items():
        cycles = activation_encryption_cycles[cipher_name]
        energy = activation_encryption_energy_pj[cipher_name]
        print(f"  {cipher_name}: {cycles:>20,} cycles, {energy:.6e} pJ")
    
    print(f"\nPeak Memory (top {len(top_tensors)} tensors):")
    for cipher_name, cipher_params in CIPHER_RATES.items():
        cycles = peak_memory_encryption_cycles[cipher_name]
        energy = peak_memory_encryption_energy_pj[cipher_name]
        print(f"  {cipher_name}: {cycles:>20,} cycles, {energy:.6e} pJ")
    
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
        'processor_frequency_mhz': PROCESSOR_FREQ_MHZ,
        'cipher_config': CIPHER_CONFIG,
        'encryption_total': {
            cipher_name: {
                'cycles_per_byte': cipher_params['cycles_per_byte'],
                'power_per_byte_watts': cipher_params['power_per_byte_watts'],
                'energy_per_op_pj': energy_per_op_pj[cipher_name],
                'total_cycles': total_encryption_cycles[cipher_name],
                'total_energy_pj': total_encryption_energy_pj[cipher_name]
            }
            for cipher_name, cipher_params in CIPHER_RATES.items()
        },
        'encryption_activation': {
            cipher_name: {
                'cycles_per_byte': cipher_params['cycles_per_byte'],
                'power_per_byte_watts': cipher_params['power_per_byte_watts'],
                'energy_per_op_pj': energy_per_op_pj[cipher_name],
                'cycles': activation_encryption_cycles[cipher_name],
                'energy_pj': activation_encryption_energy_pj[cipher_name]
            }
            for cipher_name, cipher_params in CIPHER_RATES.items()
        },
        'encryption_peak_memory': {
            cipher_name: {
                'cycles_per_byte': cipher_params['cycles_per_byte'],
                'power_per_byte_watts': cipher_params['power_per_byte_watts'],
                'energy_per_op_pj': energy_per_op_pj[cipher_name],
                'cycles': peak_memory_encryption_cycles[cipher_name],
                'energy_pj': peak_memory_encryption_energy_pj[cipher_name]
            }
            for cipher_name, cipher_params in CIPHER_RATES.items()
        },
        'layers': layer_info,
        'layer_type_summary': {
            layer_type: {
                'count': stats['count'],
                'bytes': stats['bytes'],
                'percentage': round((stats['bytes'] / total_tensor_bytes * 100), 2) if total_tensor_bytes > 0 else 0,
                'encryption': {
                    cipher_name: {
                        'cycles': stats['bytes'] * cipher_params['cycles_per_byte'],
                        'energy_pj': calculate_total_encryption_energy(
                            stats['bytes'], 
                            energy_per_op_pj[cipher_name]
                        )
                    }
                    for cipher_name, cipher_params in CIPHER_RATES.items()
                }
            }
            for layer_type, stats in layer_types.items()
        }
    }
    
    return result


def find_onnx_models(directory):
    """
    Find all .onnx files in subfolders of the given directory.
    Returns a list of tuples: (model_name, model_path)
    where model_name is derived from the containing folder name.
    """
    models = []
    
    # Look for .onnx files in immediate subfolders
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path):
            # Find .onnx files in this subfolder
            pattern = os.path.join(folder_path, "*.onnx")
            onnx_files = glob.glob(pattern)
            for onnx_file in onnx_files:
                models.append((folder_name, onnx_file))
    
    return models


def main():
    # Ensure output directory exists
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Find all ONNX models in the models directory (searches subfolders)
    model_files = find_onnx_models(MODELS_DIR)
    
    if not model_files:
        print(f"No .onnx files found in {MODELS_DIR}")
        print(f"Please organize your models in subfolders:")
        print(f"  {MODELS_DIR}/ModelName/model.onnx")
        return
    
    print(f"Found {len(model_files)} ONNX model(s):")
    for model_name, model_path in model_files:
        print(f"  - {model_name}: {os.path.basename(model_path)}")
    
    # Analyze each model
    all_results = {
        'analysis_timestamp': datetime.now().isoformat(),
        'models_directory': MODELS_DIR,
        'outputs_directory': OUTPUTS_DIR,
        'total_models_analyzed': len(model_files),
        'models': {}
    }
    
    for model_name, model_path in model_files:
        try:
            result = analyze_onnx_model(model_path, model_name)
            all_results['models'][model_name] = result
        except Exception as e:
            print(f"\nError analyzing {model_name}: {e}")
            all_results['models'][model_name] = {'error': str(e)}
    
    # Save results to JSON file
    output_file = os.path.join(OUTPUTS_DIR, "onnx_model_analysis_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    # Save results to CSV file
    csv_output_file = os.path.join(OUTPUTS_DIR, "onnx_model_analysis_results.csv")
    with open(csv_output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        cipher_cycle_headers = [
            f"{cipher_name} Cycles"
            for cipher_name in CIPHER_RATES.keys()
        ]
        cipher_energy_headers = [
            f"{cipher_name} Energy (pJ)"
            for cipher_name in CIPHER_RATES.keys()
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
        ] + cipher_cycle_headers + cipher_energy_headers)
        
        for model_name, model_data in all_results['models'].items():
            if 'error' in model_data:
                error_row = [model_name, 'ERROR', model_data['error']] + [''] * (5 + len(CIPHER_RATES) * 2)
                writer.writerow(error_row)
                continue
            
            for layer in model_data['layers']:
                cipher_cycles = [layer['encryption_cycles'].get(cipher_name, 0) for cipher_name in CIPHER_RATES.keys()]
                cipher_energies = [layer['encryption_energy_pj'].get(cipher_name, 0) for cipher_name in CIPHER_RATES.keys()]
                writer.writerow([
                    model_name,
                    layer['index'],
                    layer['name'],
                    str(layer['shape']),
                    layer['dtype'],
                    layer['num_parameters'],
                    layer['bytes'],
                    round(layer['bytes'] / (1024 * 1024), 6)
                ] + cipher_cycles + cipher_energies)
            
            writer.writerow([])
            total_cipher_cycles = [
                model_data['encryption_total'][cipher_name]['total_cycles']
                for cipher_name in CIPHER_RATES.keys()
            ]
            total_cipher_energies = [
                model_data['encryption_total'][cipher_name]['total_energy_pj']
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
            ] + total_cipher_cycles + total_cipher_energies)
            writer.writerow([])
    
    # Create summary CSV
    summary_csv_file = os.path.join(OUTPUTS_DIR, "onnx_model_analysis_summary.csv")
    with open(summary_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        energy_per_op_headers = [f"{cn} (pJ/op)" for cn in CIPHER_RATES.keys()]
        pj_per_cycle_headers = [f"{cn} (pJ/cycle)" for cn in CIPHER_RATES.keys()]
        total_cipher_cycle_headers = [f"{cn} Total Cycles" for cn in CIPHER_RATES.keys()]
        avg_cycles_per_layer_headers = [f"{cn} Avg Cycles/Layer" for cn in CIPHER_RATES.keys()]
        total_cipher_energy_headers = [f"{cn} Total Energy (pJ)" for cn in CIPHER_RATES.keys()]
        avg_energy_per_layer_headers = [f"{cn} Avg Energy/Layer (pJ)" for cn in CIPHER_RATES.keys()]
        activation_cipher_cycle_headers = [f"{cn} Activation Cycles" for cn in CIPHER_RATES.keys()]
        activation_cipher_energy_headers = [f"{cn} Activation Energy (pJ)" for cn in CIPHER_RATES.keys()]
        peak_memory_cipher_cycle_headers = [f"{cn} Peak Mem Cycles" for cn in CIPHER_RATES.keys()]
        peak_memory_cipher_energy_headers = [f"{cn} Peak Mem Energy (pJ)" for cn in CIPHER_RATES.keys()]
        
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
            'Peak Memory MB',
            'Processor Freq (MHz)'
        ] + energy_per_op_headers + pj_per_cycle_headers
          + total_cipher_cycle_headers + avg_cycles_per_layer_headers + total_cipher_energy_headers + avg_energy_per_layer_headers
          + activation_cipher_cycle_headers + activation_cipher_energy_headers
          + peak_memory_cipher_cycle_headers + peak_memory_cipher_energy_headers)
        
        for model_name, model_data in all_results['models'].items():
            if 'error' in model_data:
                error_row = [model_name, 'ERROR'] + [''] * (12 + len(CIPHER_RATES) * 10)
                writer.writerow(error_row)
                continue
            
            energy_per_op = [model_data['encryption_total'][cn]['energy_per_op_pj'] for cn in CIPHER_RATES.keys()]
            pj_per_cycle = [model_data['encryption_total'][cn]['energy_per_op_pj'] / model_data['encryption_total'][cn]['cycles_per_byte'] for cn in CIPHER_RATES.keys()]
            total_cipher_cycles = [model_data['encryption_total'][cn]['total_cycles'] for cn in CIPHER_RATES.keys()]
            num_layers = model_data['total_tensors']
            avg_cycles_per_layer = [model_data['encryption_total'][cn]['total_cycles'] / num_layers if num_layers > 0 else 0 for cn in CIPHER_RATES.keys()]
            total_cipher_energies = [model_data['encryption_total'][cn]['total_energy_pj'] for cn in CIPHER_RATES.keys()]
            avg_energy_per_layer = [model_data['encryption_total'][cn]['total_energy_pj'] / num_layers if num_layers > 0 else 0 for cn in CIPHER_RATES.keys()]
            activation_cipher_cycles = [model_data['encryption_activation'][cn]['cycles'] for cn in CIPHER_RATES.keys()]
            activation_cipher_energies = [model_data['encryption_activation'][cn]['energy_pj'] for cn in CIPHER_RATES.keys()]
            peak_memory_cipher_cycles = [model_data['encryption_peak_memory'][cn]['cycles'] for cn in CIPHER_RATES.keys()]
            peak_memory_cipher_energies = [model_data['encryption_peak_memory'][cn]['energy_pj'] for cn in CIPHER_RATES.keys()]
            
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
                model_data['peak_memory_mb'],
                model_data.get('processor_frequency_mhz', PROCESSOR_FREQ_MHZ)
            ] + energy_per_op + pj_per_cycle
              + total_cipher_cycles + avg_cycles_per_layer + total_cipher_energies + avg_energy_per_layer
              + activation_cipher_cycles + activation_cipher_energies
              + peak_memory_cipher_cycles + peak_memory_cipher_energies)
    
    print("\n" + "=" * 80)
    print(f"Results saved to:")
    print(f"  JSON: {output_file}")
    print(f"  CSV (detailed): {csv_output_file}")
    print(f"  CSV (summary): {summary_csv_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
