"""
ONNX Model Encryption Cycles Plotter
Plots ASCON encryption cycles per layer for multiple models.
"""

import json
import math
import os
import matplotlib.pyplot as plt

# Configuration
BASE_DIR = os.path.dirname(__file__)
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

CIPHER = 'ascon-154'
MAX_LAYERS = 800

# Custom start layer for specific models
START_LAYERS = {
    'Whisper-Small-Encoder': 40,
    'Whisper-Small-Decoder': 140,
}

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']

# Plot style
plt.rcParams.update({
    'font.family': 'arial',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.figsize': (8, 5),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_analysis_results():
    """Load model analysis results from JSON file."""
    json_path = os.path.join(OUTPUTS_DIR, "onnx_model_analysis_results.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    data = load_analysis_results()
    
    # Filter out models with errors
    models = {k: v for k, v in data['models'].items() if 'error' not in v}
    n_models = len(models)
    
    # Create subplot grid (2 columns)
    n_cols = 2
    n_rows = (n_models + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()
    
    for idx, (model_name, model_data) in enumerate(models.items()):
        ax = axes[idx]
        
        start_layer = START_LAYERS.get(model_name, 0)
        all_layers = model_data['layers']
        
        # Filter layers by index and limit count
        layers = [l for l in all_layers if l['index'] >= start_layer][:MAX_LAYERS]
        layer_indices = [layer['index'] for layer in layers]
        cycles = [layer['encryption_cycles'].get(CIPHER, 0) for layer in layers]
        
        ax.plot(layer_indices, cycles, 
                color=COLORS[idx % len(COLORS)],
                linewidth=1.2,
                alpha=0.85)
        
        # Calculate average cycles and format as x×10^k
        avg_cycles = sum(cycles) / len(cycles) if cycles else 0
        if avg_cycles > 0:
            exponent = int(math.floor(math.log10(avg_cycles)))
            mantissa = avg_cycles / (10 ** exponent)
            avg_str = f'{mantissa:.2f}×10$^{{{exponent}}}$'
        else:
            avg_str = '0'
        
        ax.set_xlabel('Layer Index')
        ax.set_ylabel(f'Cycles (Avg: {avg_str})')
        ax.set_yscale('log')
        ax.set_title(model_name)
        if layer_indices:
            ax.set_xlim(layer_indices[0], layer_indices[-1])
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('ASCON Encryption Cycles per Layer', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'ascon_cycles_per_layer.png'), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
