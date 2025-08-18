import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set professional color scheme
COLORS = {
    'conventional': '#C41E3A',      # Red
    'enhanced': '#2E8B57',          # Sea Green  
    'accent': '#FFD700',            # Gold
    'text': '#2F4F4F'               # Dark Slate Gray
}

# Set style for professional appearance
plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

def generate_system_architecture_figure():
    """Generate Figure 3: BITW System Architecture and Control Hierarchy"""
    
    print("Generating Figure 3: System Architecture...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left subplot: BITW Architecture
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    
    # Draw inverter
    inverter = patches.Rectangle((1, 4), 2, 2, linewidth=2, edgecolor='black', 
                                facecolor=COLORS['conventional'], alpha=0.7)
    ax1.add_patch(inverter)
    ax1.text(2, 5, 'Inverter', ha='center', va='center', fontweight='bold', color='white')
    
    # Draw BITW controller
    bitw = patches.Rectangle((4, 3.5), 2.5, 3, linewidth=2, edgecolor=COLORS['enhanced'], 
                            facecolor=COLORS['enhanced'], alpha=0.8)
    ax1.add_patch(bitw)
    ax1.text(5.25, 5, 'BITW\nController', ha='center', va='center', fontweight='bold', color='white')
    
    # Draw cloud
    cloud = patches.Ellipse((8, 8), 3, 1.5, linewidth=2, edgecolor=COLORS['accent'],
                           facecolor=COLORS['accent'], alpha=0.6)
    ax1.add_patch(cloud)
    ax1.text(8, 8, 'Cloud\nML Training', ha='center', va='center', fontweight='bold')
    
    # Draw grid connection
    grid = patches.Rectangle((7.5, 1), 2, 2, linewidth=2, edgecolor='green',
                           facecolor='lightgreen', alpha=0.7)
    ax1.add_patch(grid)
    ax1.text(8.5, 2, 'Grid', ha='center', va='center', fontweight='bold')
    
    # Draw connections
    # Inverter to BITW
    ax1.arrow(3.1, 5, 0.8, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax1.arrow(4, 5.5, -0.8, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # BITW to Cloud
    ax1.arrow(6, 6.2, 1.5, 1.3, head_width=0.1, head_length=0.1, fc=COLORS['accent'], ec=COLORS['accent'])
    ax1.text(6.8, 6.8, 'Training\nData', ha='center', va='center', fontsize=10, color=COLORS['accent'])
    
    # BITW to Grid
    ax1.arrow(6.2, 4, 1.2, -1.5, head_width=0.1, head_length=0.1, fc='green', ec='green')
    ax1.text(7, 3, 'Control\nSignals', ha='center', va='center', fontsize=10, color='green')
    
    # Add component labels
    ax1.text(2, 3.5, 'Modbus/RS-485', ha='center', va='center', fontsize=10, style='italic')
    ax1.text(5.25, 2.8, 'Edge ML\nInference', ha='center', va='center', fontsize=10, style='italic')
    
    ax1.set_title('A) BITW Controller Architecture', fontweight='bold', fontsize=14)
    ax1.axis('off')
    
    # Right subplot: Control Hierarchy
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    # Primary Control Layer
    primary = patches.Rectangle((1, 7), 8, 1.5, linewidth=2, edgecolor=COLORS['conventional'],
                               facecolor=COLORS['conventional'], alpha=0.8)
    ax2.add_patch(primary)
    ax2.text(5, 7.75, 'PRIMARY CONTROL (ms)', ha='center', va='center', 
             fontweight='bold', color='white', fontsize=12)
    ax2.text(5, 7.3, 'LMI-Passivity Droop + PINODEs + CBF Safety', ha='center', va='center', 
             color='white', fontsize=10)
    
    # Secondary Control Layer
    secondary = patches.Rectangle((1, 4.5), 8, 1.5, linewidth=2, edgecolor=COLORS['enhanced'],
                                 facecolor=COLORS['enhanced'], alpha=0.8)
    ax2.add_patch(secondary)
    ax2.text(5, 5.25, 'SECONDARY CONTROL (s)', ha='center', va='center', 
             fontweight='bold', color='white', fontsize=12)
    ax2.text(5, 4.8, 'MARL-Enhanced Consensus + Event Triggering', ha='center', va='center', 
             color='white', fontsize=10)
    
    # Tertiary Control Layer
    tertiary = patches.Rectangle((1, 2), 8, 1.5, linewidth=2, edgecolor=COLORS['accent'],
                                facecolor=COLORS['accent'], alpha=0.8)
    ax2.add_patch(tertiary)
    ax2.text(5, 2.75, 'TERTIARY CONTROL (min)', ha='center', va='center', 
             fontweight='bold', color='white', fontsize=12)
    ax2.text(5, 2.3, 'GNN-Accelerated ADMM + Privacy-Preserving OPF', ha='center', va='center', 
             color='white', fontsize=10)
    
    # Draw arrows between layers
    ax2.arrow(5, 6.9, 0, -0.3, head_width=0.2, head_length=0.1, fc='gray', ec='gray', linewidth=2)
    ax2.arrow(5, 4.4, 0, -0.3, head_width=0.2, head_length=0.1, fc='gray', ec='gray', linewidth=2)
    
    # Add performance annotations
    ax2.text(9.5, 7.75, '19.8%\nImprovement', ha='center', va='center', fontsize=10, 
             fontweight='bold', color=COLORS['conventional'],
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    ax2.text(9.5, 5.25, '30.0%\nImprovement', ha='center', va='center', fontsize=10, 
             fontweight='bold', color=COLORS['enhanced'],
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    ax2.text(9.5, 2.75, '28.0%\nImprovement\n(Projected)', ha='center', va='center', fontsize=10, 
             fontweight='bold', color=COLORS['accent'],
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    ax2.set_title('B) Hierarchical Control Architecture', fontweight='bold', fontsize=14)
    ax2.axis('off')
    
    plt.suptitle('Figure 3: BITW Controller System Architecture', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Add performance banner
    performance_text = '↓ 82% Cost Reduction • ↑ 30% Faster Convergence (36% fewer iterations) • ↑ 150 ms Delay Tolerance'
    plt.figtext(0.5, 0.02, performance_text, ha='center', va='bottom', 
                fontsize=12, fontweight='bold', style='italic')
    
    plt.tight_layout()
    plt.savefig('figure3_system_architecture_original.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure3_system_architecture_original.pdf', dpi=300, bbox_inches='tight')
    print("✓ Figure 3 saved: figure3_system_architecture_original.png/.pdf")
    
    return fig

if __name__ == "__main__":
    generate_system_architecture_figure()
    plt.show()