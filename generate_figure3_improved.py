#!/usr/bin/env python3
"""
Generate an improved Figure 3 (System Architecture) with better quality and less space
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

def create_improved_figure3():
    """Create an improved, more compact Figure 3: System Architecture"""
    
    # Set publication style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 9,
        'axes.linewidth': 1.5,
        'mathtext.fontset': 'cm',
        'pdf.fonttype': 42,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    # Create figure with reduced size
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.axis('off')
    
    # Enhanced color scheme
    colors = {
        'cloud': '#E3F2FD',
        'cloud_border': '#1976D2',
        'edge': '#FFF3E0',
        'edge_border': '#F57C00',
        'mas': '#E8F5E9',
        'mas_border': '#388E3C',
        'arrow': '#424242',
        'text_dark': '#212121',
        'text_light': '#616161'
    }
    
    # Layer dimensions
    layer_height = 0.7
    layer_y_positions = [2.6, 1.5, 0.4]
    layer_width = 9.2
    
    # Cloud Layer (Top)
    cloud_box = FancyBboxPatch((0.4, layer_y_positions[0]), layer_width, layer_height,
                               boxstyle="round,pad=0.05",
                               facecolor=colors['cloud'], 
                               edgecolor=colors['cloud_border'], 
                               linewidth=2.5,
                               alpha=0.9)
    ax.add_patch(cloud_box)
    
    # Cloud icon
    cloud_icon_x = 1.0
    cloud_icon_y = layer_y_positions[0] + layer_height/2
    for dx, dy, r in [(0, 0, 0.15), (-0.1, -0.05, 0.12), (0.1, -0.05, 0.12)]:
        circle = Circle((cloud_icon_x + dx, cloud_icon_y + dy), r, 
                       color=colors['cloud_border'], alpha=0.7)
        ax.add_patch(circle)
    
    ax.text(2.5, layer_y_positions[0] + layer_height/2, 
            'CLOUD: Federated Learning', 
            fontsize=10, weight='bold', va='center', color=colors['text_dark'])
    
    ax.text(5.5, layer_y_positions[0] + layer_height/2 + 0.1, 
            'Physics-Informed Neural ODEs', 
            fontsize=8, va='center', style='italic')
    ax.text(5.5, layer_y_positions[0] + layer_height/2 - 0.1, 
            'Multi-Agent RL Training', 
            fontsize=8, va='center', style='italic')
    
    ax.text(8.5, layer_y_positions[0] + layer_height/2, 
            'Federated\nAggregation', 
            fontsize=8, va='center', ha='center', color=colors['text_light'])
    
    # Edge Layer (Middle)
    edge_box = FancyBboxPatch((0.4, layer_y_positions[1]), layer_width, layer_height,
                              boxstyle="round,pad=0.05",
                              facecolor=colors['edge'], 
                              edgecolor=colors['edge_border'], 
                              linewidth=2.5,
                              alpha=0.9)
    ax.add_patch(edge_box)
    
    # Edge icon (chip symbol)
    chip_x = 1.0
    chip_y = layer_y_positions[1] + layer_height/2
    chip_rect = mpatches.Rectangle((chip_x-0.1, chip_y-0.1), 0.2, 0.2,
                                  facecolor=colors['edge_border'], 
                                  edgecolor='none', alpha=0.7)
    ax.add_patch(chip_rect)
    
    ax.text(2.5, layer_y_positions[1] + layer_height/2, 
            'EDGE: Real-Time (<10ms)', 
            fontsize=10, weight='bold', va='center', color=colors['text_dark'])
    
    ax.text(5.5, layer_y_positions[1] + layer_height/2 + 0.1, 
            'BITW Hardware + ONNX', 
            fontsize=8, va='center', style='italic')
    ax.text(5.5, layer_y_positions[1] + layer_height/2 - 0.1, 
            'Control Barrier Functions', 
            fontsize=8, va='center', style='italic')
    
    ax.text(8.5, layer_y_positions[1] + layer_height/2, 
            'Safety\nEnforcement', 
            fontsize=8, va='center', ha='center', color=colors['text_light'])
    
    # MAS Layer (Bottom)
    mas_box = FancyBboxPatch((0.4, layer_y_positions[2]), layer_width, layer_height,
                             boxstyle="round,pad=0.05",
                             facecolor=colors['mas'], 
                             edgecolor=colors['mas_border'], 
                             linewidth=2.5,
                             alpha=0.9)
    ax.add_patch(mas_box)
    
    # Network icon
    net_x = 1.0
    net_y = layer_y_positions[2] + layer_height/2
    # Draw simple network nodes
    for angle in [0, 120, 240]:
        x = net_x + 0.12 * np.cos(np.radians(angle))
        y = net_y + 0.12 * np.sin(np.radians(angle))
        circle = Circle((x, y), 0.05, color=colors['mas_border'], alpha=0.7)
        ax.add_patch(circle)
        # Connect to center
        ax.plot([net_x, x], [net_y, y], color=colors['mas_border'], 
                alpha=0.5, linewidth=1)
    
    ax.text(2.5, layer_y_positions[2] + layer_height/2, 
            'MAS: Distributed Control', 
            fontsize=10, weight='bold', va='center', color=colors['text_dark'])
    
    ax.text(5.5, layer_y_positions[2] + layer_height/2 + 0.1, 
            'Primary (ms) • Secondary (s)', 
            fontsize=8, va='center', style='italic')
    ax.text(5.5, layer_y_positions[2] + layer_height/2 - 0.1, 
            'Tertiary (min) • GNN-ADMM', 
            fontsize=8, va='center', style='italic')
    
    ax.text(8.5, layer_y_positions[2] + layer_height/2, 
            'Consensus\nOptimization', 
            fontsize=8, va='center', ha='center', color=colors['text_light'])
    
    # Enhanced arrows with gradient effect
    arrow_x = 5.0
    for i in range(2):
        arrow = FancyArrowPatch((arrow_x, layer_y_positions[i] - 0.05),
                               (arrow_x, layer_y_positions[i+1] + layer_height + 0.05),
                               connectionstyle="arc3,rad=0", 
                               arrowstyle='->', 
                               mutation_scale=20,
                               linewidth=2.5,
                               color=colors['arrow'],
                               alpha=0.8)
        ax.add_patch(arrow)
    
    # Add side labels for time scales
    ax.text(0.2, layer_y_positions[0] + layer_height/2, 'Hours', 
            fontsize=7, rotation=90, va='center', ha='center', 
            color=colors['text_light'], weight='bold')
    ax.text(0.2, layer_y_positions[1] + layer_height/2, 'ms', 
            fontsize=7, rotation=90, va='center', ha='center', 
            color=colors['text_light'], weight='bold')
    ax.text(0.2, layer_y_positions[2] + layer_height/2, 'ms-min', 
            fontsize=7, rotation=90, va='center', ha='center', 
            color=colors['text_light'], weight='bold')
    
    # Add performance indicators
    perf_y = 3.4
    ax.text(1.5, perf_y, '↑ 82% Cost Reduction', fontsize=7, 
            color=colors['mas_border'], weight='bold')
    ax.text(4.0, perf_y, '↑ 30% Faster Convergence', fontsize=7, 
            color=colors['cloud_border'], weight='bold')
    ax.text(7.0, perf_y, '↑ 150ms Delay Tolerance', fontsize=7, 
            color=colors['edge_border'], weight='bold')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Generate the improved figure
    fig = create_improved_figure3()
    
    # Save as PDF
    fig.savefig('figure3_system_architecture.pdf', dpi=300, bbox_inches='tight', 
                pad_inches=0.05, format='pdf')
    print("✓ Generated improved figure3_system_architecture.pdf")
    
    # Also save as PNG for preview
    fig.savefig('figure3_preview.png', dpi=150, bbox_inches='tight', 
                pad_inches=0.05, format='png')
    print("✓ Generated figure3_preview.png for preview")
    
    plt.close()