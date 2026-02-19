import onnx
import os

models_dir = 'models'
data_type_names = {1: 'FLOAT32', 2: 'UINT8', 3: 'INT8', 4: 'UINT16', 5: 'INT16', 6: 'INT32', 7: 'INT64', 10: 'FLOAT16', 11: 'DOUBLE', 12: 'UINT32', 13: 'UINT64'}

for root, dirs, files in os.walk(models_dir):
    for f in files:
        if f.endswith('.onnx'):
            model_path = os.path.join(root, f)
            print(f'\n=== {model_path} ===')
            try:
                model = onnx.load(model_path, load_external_data=False)
                data_types = {}
                for init in model.graph.initializer:
                    dt = init.data_type
                    data_types[dt] = data_types.get(dt, 0) + 1
                print(f'Data types (Type ID): {data_types}')
                for dt_id, count in data_types.items():
                    name = data_type_names.get(dt_id, f'UNKNOWN({dt_id})')
                    print(f'  {name} (ID {dt_id}): {count} tensors')
                # Quantization check
                has_quant = any(dt in [2, 3] for dt in data_types.keys())
                print(f'  Quantized (INT8/UINT8): {has_quant}')
            except Exception as e:
                print(f'Error: {e}')