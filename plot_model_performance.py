#!/usr/bin/env python3
"""
Grouped bar plot comparing model performance across domains.
Shows how agent performance improves with more capable models.
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
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

# ============================================================================
# DATA - Model performance (average reward) by domain
# ============================================================================

domains = ['Telecom', 'Airline', 'Retail']

# Performance data (average reward = pass rate since reward is binary 0/1)
# Format: [GPT-4o, GPT-4.1, GPT-5.2] for each domain
performance = {
    'Telecom': [0.00, 0.07, 0.21],   # GPT-4o, GPT-4.1, GPT-5.2
    'Airline': [0.21, 0.36, 0.43],
    'Retail':  [0.43, 0.50, 0.64],
}

models = ['GPT-4o', 'GPT-4.1', 'GPT-5.2']

# ============================================================================
# COLORS - Blue-green scheme (matching histogram transparency style)
# ============================================================================

# Colors from less capable to more capable (same as histogram)
colors = {
    'GPT-4o':  '#4A90A4',   # Soft blue (least capable)
    'GPT-4.1': '#2E8B57',   # Sea green (middle)
    'GPT-5.2': '#0D7377',   # Deep teal (most capable)
}

edge_colors = {
    'GPT-4o':  '#2E6A7A',
    'GPT-4.1': '#1D5C3A',
    'GPT-5.2': '#095456',
}

# Alpha values matching histogram style (more transparent fill, saturated edge)
fill_alpha = 0.45

# ============================================================================
# PLOTTING
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(domains))
width = 0.25  # Width of bars
multiplier = 0

for i, model in enumerate(models):
    # Get performance for this model across all domains
    values = [performance[domain][i] for domain in domains]
    offset = width * (i - 1)  # Center the groups
    
    bars = ax.bar(
        x + offset, 
        values, 
        width, 
        label=model,
        color=colors[model],
        edgecolor=edge_colors[model],
        linewidth=2.5,
        alpha=fill_alpha,
    )
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f'{val:.0%}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='medium',
            color='#333333',
        )

# Styling
ax.set_xlabel('Task Domain', fontweight='medium')
ax.set_ylabel('Pass Rate (Fraction of Tasks Passed)', fontweight='medium')
ax.set_title('Task-Agent Performance by Model and Domain on JLBench Tasks', fontweight='bold', pad=15)

ax.set_xticks(x)
ax.set_xticklabels(domains, fontsize=13)
ax.set_ylim(0, 0.85)

# Format y-axis as percentage
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

# Legend
legend = ax.legend(
    loc='upper left', 
    framealpha=0.95, 
    edgecolor='#888888', 
    fancybox=True,
    title='Agent Model',
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

# Add a subtle annotation about the trend
ax.annotate(
    'More capable models perform better on JLBench tasks',
    xy=(0.98, 0.96),
    xycoords='axes fraction',
    fontsize=10,
    color='#666666',
    style='italic',
    ha='right',
    va='top',
)

plt.tight_layout()

# Save figures
plt.savefig('figs/model_performance_by_domain.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('figs/model_performance_by_domain.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("✅ Saved: figs/model_performance_by_domain.png")
print("✅ Saved: figs/model_performance_by_domain.pdf")

# ============================================================================
# ALTERNATIVE: Horizontal bar chart for different perspective
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(11, 7))

y = np.arange(len(models))
height = 0.25

for i, domain in enumerate(domains):
    values = [performance[domain][j] for j in range(len(models))]
    offset = height * (i - 1)
    
    domain_colors = {
        'Telecom': '#0D7377',
        'Airline': '#2E8B57', 
        'Retail':  '#4A90A4',
    }
    domain_edges = {
        'Telecom': '#095456',
        'Airline': '#1D5C3A',
        'Retail':  '#2E6A7A',
    }
    
    bars = ax2.barh(
        y + offset,
        values,
        height,
        label=domain,
        color=domain_colors[domain],
        edgecolor=domain_edges[domain],
        linewidth=2.5,
        alpha=fill_alpha,
    )
    
    # Add value labels
    for bar, val in zip(bars, values):
        width_val = bar.get_width()
        ax2.annotate(
            f'{val:.0%}',
            xy=(width_val, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha='left',
            va='center',
            fontsize=11,
            fontweight='medium',
            color='#333333',
        )

ax2.set_ylabel('Agent Model', fontweight='medium')
ax2.set_xlabel('Pass Rate (Fraction of Tasks Passed)', fontweight='medium')
ax2.set_title('Agent Performance by Model and Domain', fontweight='bold', pad=15)

ax2.set_yticks(y)
ax2.set_yticklabels(models, fontsize=13)
ax2.set_xlim(0, 0.85)

ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

legend2 = ax2.legend(
    loc='lower right',
    framealpha=0.95,
    edgecolor='#888888',
    fancybox=True,
    title='Domain',
    title_fontsize=12,
)
legend2.get_frame().set_linewidth(1.5)

ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='x')
ax2.set_axisbelow(True)

for spine in ax2.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('#444444')

plt.tight_layout()

plt.savefig('figs/model_performance_horizontal.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('figs/model_performance_horizontal.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

print("✅ Saved: figs/model_performance_horizontal.png")
print("✅ Saved: figs/model_performance_horizontal.pdf")

# ============================================================================
# PRINT SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("PERFORMANCE SUMMARY")
print("=" * 60)

print(f"\n{'Domain':<12} {'GPT-4o':>10} {'GPT-4.1':>10} {'GPT-5.2':>10} {'Δ (4o→5.2)':>12}")
print("-" * 56)

for domain in domains:
    vals = performance[domain]
    delta = vals[2] - vals[0]  # GPT-5.2 - GPT-4o
    print(f"{domain:<12} {vals[0]:>10.0%} {vals[1]:>10.0%} {vals[2]:>10.0%} {delta:>+11.0%}")

# Overall
overall_4o = sum(performance[d][0] for d in domains) / 3
overall_41 = sum(performance[d][1] for d in domains) / 3
overall_52 = sum(performance[d][2] for d in domains) / 3

print("-" * 56)
print(f"{'Overall':<12} {overall_4o:>10.0%} {overall_41:>10.0%} {overall_52:>10.0%} {overall_52-overall_4o:>+11.0%}")

print("\n✅ All figures saved successfully!")
