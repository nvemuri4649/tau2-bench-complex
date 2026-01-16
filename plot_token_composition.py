#!/usr/bin/env python3
"""
Pie charts comparing token composition across three benchmarks.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Set style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        pass

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14

# ============================================================================
# DATA - Token composition (excluding System Prompt)
# ============================================================================

# Categories (excluding System Prompt)
categories = ['User Input', 'Assistant Output', 'Tool Call Args', 'Tool Results']

# Tau-bench (System Prompt was 0%, so no adjustment needed)
tau_bench = {
    'User Input': 10.5,
    'Assistant Output': 45.5,
    'Tool Call Args': 0.0,
    'Tool Results': 44.0,
}

# Monaco - Original had 15.2% system prompt, need to renormalize
# Original: User 0.6%, Assistant 7.1%, Tool Args 0.4%, Tool Results 76.8%
# Total without system: 84.9%
monaco_total = 0.6 + 7.1 + 0.4 + 76.8
monaco = {
    'User Input': 0.6 / monaco_total * 100,
    'Assistant Output': 7.1 / monaco_total * 100,
    'Tool Call Args': 0.4 / monaco_total * 100,
    'Tool Results': 76.8 / monaco_total * 100,
}

# JLBench (from our analysis - already no system prompt)
jlbench = {
    'User Input': 0.95,
    'Assistant Output': 1.69,
    'Tool Call Args': 0.22,
    'Tool Results': 97.14,
}

# ============================================================================
# COLORS - Blue-green scheme (matching other plots)
# ============================================================================

colors = {
    'User Input': '#4A90A4',       # Soft blue
    'Assistant Output': '#2E8B57', # Sea green
    'Tool Call Args': '#5DADE2',   # Light blue
    'Tool Results': '#0D7377',     # Deep teal
}

edge_colors = {
    'User Input': '#2E6A7A',
    'Assistant Output': '#1D5C3A',
    'Tool Call Args': '#3498DB',
    'Tool Results': '#095456',
}

# ============================================================================
# PLOTTING - Three pie charts side by side
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))  # Narrower for closer spacing

datasets = [
    ('τ²-Bench', tau_bench),
    ('Monaco (Production)', monaco),
    ('JLBench', jlbench),
]

def make_autopct(values):
    """Create autopct function that hides small percentages"""
    def autopct(pct):
        if pct < 2:
            return ''
        return f'{pct:.1f}%'
    return autopct

for ax, (name, data) in zip(axes, datasets):
    values = [data[cat] for cat in categories]
    color_list = [colors[cat] for cat in categories]
    edge_list = [edge_colors[cat] for cat in categories]
    
    # Create wedges with transparency (matching histogram style)
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,  # We'll add a legend instead
        autopct=make_autopct(values),
        colors=color_list,
        wedgeprops=dict(
            width=1.0,
            edgecolor='white',
            linewidth=2.5,
            alpha=0.45,  # Match histogram transparency
        ),
        pctdistance=0.5,
        startangle=90,
    )
    
    # Style the percentage labels
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    ax.set_title(name, fontweight='bold', fontsize=14, pad=10)

# Add a single legend for all charts
fig.legend(
    [plt.Rectangle((0,0), 1, 1, facecolor=colors[cat], edgecolor=edge_colors[cat], 
                   linewidth=2, alpha=0.45) for cat in categories],
    categories,
    loc='lower center',
    ncol=4,
    framealpha=0.95,
    edgecolor='#888888',
    fancybox=True,
    fontsize=11,
    bbox_to_anchor=(0.5, -0.02),
)

plt.suptitle('Token Composition by Benchmark', fontweight='bold', fontsize=16, y=1.02)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# Save figures
plt.savefig('figs/token_composition_pies.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('figs/token_composition_pies.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("✅ Saved: figs/token_composition_pies.png")
print("✅ Saved: figs/token_composition_pies.pdf")

# ============================================================================
# ALTERNATIVE: Donut charts for a cleaner look
# ============================================================================

fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4.5))  # Narrower for closer spacing

for ax, (name, data) in zip(axes2, datasets):
    values = [data[cat] for cat in categories]
    color_list = [colors[cat] for cat in categories]
    
    # Create donut chart
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        autopct=make_autopct(values),
        colors=color_list,
        wedgeprops=dict(
            width=0.6,  # Makes it a donut
            edgecolor='white',
            linewidth=2.5,
            alpha=0.45,  # Match histogram transparency
        ),
        pctdistance=0.75,
        startangle=90,
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
        autotext.set_color('#333333')
    
    # Add center text - always show Tool Results percentage
    tool_results_pct = data['Tool Results']
    ax.text(0, 0, f'{tool_results_pct:.0f}%\nTool\nResults',
            ha='center', va='center', fontsize=11, fontweight='bold', color='#333333')
    
    ax.set_title(name, fontweight='bold', fontsize=14, pad=10)

fig2.legend(
    [plt.Rectangle((0,0), 1, 1, facecolor=colors[cat], edgecolor=edge_colors[cat], 
                   linewidth=2, alpha=0.45) for cat in categories],
    categories,
    loc='lower center',
    ncol=4,
    framealpha=0.95,
    edgecolor='#888888',
    fancybox=True,
    fontsize=11,
    bbox_to_anchor=(0.5, -0.02),
)

plt.suptitle('Token Composition by Benchmark', fontweight='bold', fontsize=16, y=1.02)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

plt.savefig('figs/token_composition_donuts.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('figs/token_composition_donuts.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

print("✅ Saved: figs/token_composition_donuts.png")
print("✅ Saved: figs/token_composition_donuts.pdf")

# ============================================================================
# PRINT SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("TOKEN COMPOSITION SUMMARY (excluding System Prompt)")
print("=" * 70)

print(f"\n{'Category':<20} {'τ²-Bench':>12} {'Monaco':>12} {'JLBench':>12}")
print("-" * 58)

for cat in categories:
    print(f"{cat:<20} {tau_bench[cat]:>11.1f}% {monaco[cat]:>11.1f}% {jlbench[cat]:>11.1f}%")

print("-" * 58)
print(f"{'Total':<20} {'100.0':>11}% {'100.0':>11}% {'100.0':>11}%")

print("\n✅ All figures saved successfully!")
