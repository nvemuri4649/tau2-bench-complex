#!/usr/bin/env python3
"""
Plot the evolution of partial evaluation signals across model capabilities.
Shows how different evaluation metrics improve as models get more capable.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from collections import defaultdict

# Set up styling - match histogram/model performance plots
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        pass

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

# Color scheme - matching histogram/model performance plots exactly
# Colors and edge colors
COLORS = {
    'env_assertion': '#0D7377',    # Deep teal (JLBench color)
    'nl_assertion': '#4A90A4',     # Soft blue (Tau-bench color)
    'communicate': '#2E8B57',      # Sea green (Monaco color)
    'total_reward': '#7c3aed',     # Purple (only used in other charts)
}

EDGE_COLORS = {
    'env_assertion': '#095456',
    'nl_assertion': '#2E6A7A',
    'communicate': '#1D5C3A',
    'total_reward': '#5a2d9e',
}

FILL_ALPHA = 0.45

SIMULATIONS_DIR = "data/simulations"

file_model_map = {
    "telecom_complex.json": ("telecom", "gpt-4.1"),
    "airline_complex.json": ("airline", "gpt-4.1"),
    "retail_complex.json": ("retail", "gpt-4.1"),
    "telecom_gpt4o.json": ("telecom", "gpt-4o"),
    "airline_gpt4o.json": ("airline", "gpt-4o"),
    "retail_gpt4o.json": ("retail", "gpt-4o"),
    "telecom_gpt52.json": ("telecom", "gpt-5.2"),
    "airline_gpt52.json": ("airline", "gpt-5.2"),
    "retail_gpt52.json": ("retail", "gpt-5.2"),
}

# Collect partial signals by model
model_signals = defaultdict(lambda: {
    "env_assertion": [],
    "nl_assertion": [],
    "communicate": [],
    "total_reward": [],
})

for filename, (domain, model) in file_model_map.items():
    filepath = os.path.join(SIMULATIONS_DIR, filename)
    if not os.path.exists(filepath):
        continue
    
    with open(filepath) as f:
        data = json.load(f)
    
    for sim in data.get("simulations", []):
        # Skip looping trajectories
        if sim.get("termination_reason") == "max_steps":
            continue
        
        reward_info = sim.get("reward_info", {})
        
        # ENV assertions
        env_checks = reward_info.get("env_assertions", [])
        if isinstance(env_checks, list) and len(env_checks) > 0:
            env_pass_rate = sum(1 for e in env_checks if e.get("met", False)) / len(env_checks)
            model_signals[model]["env_assertion"].append(env_pass_rate)
        
        # NL assertions
        nl_checks = reward_info.get("nl_assertions", [])
        if isinstance(nl_checks, list) and len(nl_checks) > 0:
            nl_pass_rate = sum(1 for n in nl_checks if n.get("met", False)) / len(nl_checks)
            model_signals[model]["nl_assertion"].append(nl_pass_rate)
        
        # Communicate checks
        comm_checks = reward_info.get("communicate_checks", [])
        if isinstance(comm_checks, list) and len(comm_checks) > 0:
            comm_pass_rate = sum(1 for c in comm_checks if c.get("met", False)) / len(comm_checks)
            model_signals[model]["communicate"].append(comm_pass_rate)
        
        # Total reward
        total_reward = reward_info.get("reward", 0)
        model_signals[model]["total_reward"].append(total_reward)

# Calculate averages
models = ["gpt-4o", "gpt-4.1", "gpt-5.2"]
signal_types = ["env_assertion", "nl_assertion", "communicate", "total_reward"]
signal_labels = {
    "env_assertion": "Environment\nAssertions",
    "nl_assertion": "NL Assertions",
    "communicate": "Communicate\nInfo",
    "total_reward": "Overall\nTask Success",
}

# Build data matrix
data_matrix = {}
for signal in signal_types:
    data_matrix[signal] = []
    for model in models:
        vals = model_signals[model][signal]
        if vals:
            data_matrix[signal].append(sum(vals) / len(vals) * 100)
        else:
            data_matrix[signal].append(0)

# ============================================================================
# Plot 1: Line chart showing evolution
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))

for signal in signal_types:
    color = COLORS[signal]
    y = data_matrix[signal]
    
    # Plot line with markers
    ax.plot(x, y, 
            color=color, 
            linewidth=2.5, 
            marker='o', 
            markersize=12,
            markerfacecolor=color,
            markeredgecolor='white',
            markeredgewidth=2,
            label=signal_labels[signal].replace('\n', ' '),
            alpha=0.9)
    
    # Add value labels
    for i, val in enumerate(y):
        ax.annotate(f'{val:.1f}%', 
                   (i, val), 
                   textcoords="offset points",
                   xytext=(0, 12),
                   ha='center',
                   fontsize=9,
                   fontweight='bold',
                   color=color)

ax.set_xticks(x)
ax.set_xticklabels(['GPT-4o\n(Base)', 'GPT-4.1\n(Improved)', 'GPT-5.2\n(Advanced)'], fontsize=11)
ax.set_ylabel('Pass Rate (%)', fontsize=12)
ax.set_xlabel('Model Capability →', fontsize=12)
ax.set_title('Evolution of Evaluation Signals with Model Capability', fontsize=14, fontweight='bold')

ax.set_ylim(0, 100)
ax.legend(loc='upper left', framealpha=0.9, fontsize=10)

# Add annotation
ax.text(0.98, 0.02, 
        'Higher capability models show consistent\nimprovement across all evaluation dimensions',
        transform=ax.transAxes,
        fontsize=9,
        fontstyle='italic',
        color='gray',
        ha='right',
        va='bottom')

plt.tight_layout()
plt.savefig('figs/partial_signals_evolution.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('figs/partial_signals_evolution.pdf', bbox_inches='tight', facecolor='white')
print("Saved: figs/partial_signals_evolution.png")

# ============================================================================
# Plot 2: Grouped bar chart (only 3 partial signals, no overall task success)
# Matches style of model_performance plot exactly
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(models))
width = 0.25
offsets = [-1, 0, 1]

# Only use 3 signals (no total_reward) - same colors as histogram
bar_signals = ["env_assertion", "nl_assertion", "communicate"]
bar_colors = ['#0D7377', '#4A90A4', '#2E8B57']  # Deep teal, Soft blue, Sea green
bar_edges = ['#095456', '#2E6A7A', '#1D5C3A']

for i, signal in enumerate(bar_signals):
    color = bar_colors[i]
    edge = bar_edges[i]
    y = data_matrix[signal]
    
    bars = ax.bar(x + offsets[i] * width, y, width,
                  label=signal_labels[signal].replace('\n', ' '),
                  color=color,
                  alpha=FILL_ALPHA,
                  edgecolor=edge,
                  linewidth=2.5)
    
    # Add value labels on bars - dark gray text like model_performance plot
    for bar, val in zip(bars, y):
        height = bar.get_height()
        ax.annotate(f'{val:.0f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 4),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=11,
                   fontweight='medium',
                   color='#333333')

ax.set_xticks(x)
ax.set_xticklabels(['GPT-4o', 'GPT-4.1', 'GPT-5.2'], fontsize=13)
ax.set_ylabel('Pass Rate (%)', fontweight='medium')
ax.set_xlabel('Model', fontweight='medium')
ax.set_title('Partial Evaluation Signals on JLBench Tasks by Task-Agent Capability', fontweight='bold', pad=15)

ax.set_ylim(0, 105)

# Legend with custom styling - matching other plots
legend = ax.legend(
    loc='upper left', 
    framealpha=0.95, 
    edgecolor='#888888', 
    fancybox=True,
    title='Signal Type',
    title_fontsize=12,
)
legend.get_frame().set_linewidth(1.5)

# Grid styling
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
ax.set_axisbelow(True)

# Spine styling
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('#444444')

plt.tight_layout()
plt.savefig('figs/partial_signals_bars.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('figs/partial_signals_bars.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
print("Saved: figs/partial_signals_bars.png")

# ============================================================================
# Plot 3: Heatmap style
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

# Build matrix for heatmap
heatmap_data = np.array([data_matrix[s] for s in signal_types])

# Custom colormap (blue-green)
from matplotlib.colors import LinearSegmentedColormap
colors_cmap = ['#f0fdfa', '#99f6e4', '#2dd4bf', '#0d9488', '#115e59']
cmap = LinearSegmentedColormap.from_list('teal', colors_cmap)

im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=0, vmax=100)

# Labels
ax.set_xticks(np.arange(len(models)))
ax.set_yticks(np.arange(len(signal_types)))
ax.set_xticklabels(['GPT-4o', 'GPT-4.1', 'GPT-5.2'], fontsize=11)
ax.set_yticklabels([signal_labels[s].replace('\n', ' ') for s in signal_types], fontsize=10)

# Add text annotations
for i in range(len(signal_types)):
    for j in range(len(models)):
        val = heatmap_data[i, j]
        text_color = 'white' if val > 50 else 'black'
        ax.text(j, i, f'{val:.1f}%',
               ha='center', va='center',
               color=text_color,
               fontsize=11,
               fontweight='bold')

ax.set_title('Evaluation Signal Pass Rates Across Models', fontsize=14, fontweight='bold')
ax.set_xlabel('Model Capability →', fontsize=12)

# Colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Pass Rate (%)', fontsize=10)

plt.tight_layout()
plt.savefig('figs/partial_signals_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('figs/partial_signals_heatmap.pdf', bbox_inches='tight', facecolor='white')
print("Saved: figs/partial_signals_heatmap.png")

# ============================================================================
# Plot 4: Delta/Improvement chart
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate improvements
improvements = {}
for signal in signal_types:
    base = data_matrix[signal][0]  # GPT-4o
    final = data_matrix[signal][2]  # GPT-5.2
    improvements[signal] = final - base

# Sort by improvement
sorted_signals = sorted(signal_types, key=lambda s: improvements[s], reverse=True)

y_pos = np.arange(len(sorted_signals))
improvement_vals = [improvements[s] for s in sorted_signals]
colors_list = [COLORS[s] for s in sorted_signals]

bars = ax.barh(y_pos, improvement_vals,
               color=colors_list,
               alpha=0.45,
               edgecolor=colors_list,
               linewidth=2.5)

ax.set_yticks(y_pos)
ax.set_yticklabels([signal_labels[s].replace('\n', ' ') for s in sorted_signals], fontsize=11)
ax.set_xlabel('Improvement (percentage points)', fontsize=12)
ax.set_title('Performance Improvement: GPT-4o → GPT-5.2', fontsize=14, fontweight='bold')

# Add value labels
for bar, val, color in zip(bars, improvement_vals, colors_list):
    width = bar.get_width()
    ax.annotate(f'+{val:.1f}pp',
               xy=(width, bar.get_y() + bar.get_height()/2),
               xytext=(5, 0),
               textcoords="offset points",
               ha='left', va='center',
               fontsize=11,
               fontweight='bold',
               color=color)

ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('figs/partial_signals_improvement.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('figs/partial_signals_improvement.pdf', bbox_inches='tight', facecolor='white')
print("Saved: figs/partial_signals_improvement.png")

plt.close('all')
print("\nAll plots saved successfully!")

# Print summary
print("\n" + "="*60)
print("SUMMARY: Model Capability vs Evaluation Signals")
print("="*60)
for signal in signal_types:
    vals = data_matrix[signal]
    print(f"\n{signal_labels[signal].replace(chr(10), ' ')}:")
    print(f"  GPT-4o:  {vals[0]:.1f}%")
    print(f"  GPT-4.1: {vals[1]:.1f}%")
    print(f"  GPT-5.2: {vals[2]:.1f}%")
    print(f"  Δ (4o→5.2): +{vals[2]-vals[0]:.1f}pp")
