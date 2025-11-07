#!/usr/bin/env python3
"""
Visualize Rocket League Camera Settings Data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the data
print("Loading camera settings data...")
df = pd.read_csv('camera_settings.csv')

print(f"Loaded {len(df)} player settings")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Basic statistics
print("\n" + "="*50)
print("BASIC STATISTICS")
print("="*50)
print(df.describe())

# Clean the data - convert numeric columns
numeric_cols = ['FOV', 'Height', 'Angle', 'Distance', 'Stiffness', 'Swivel speed', 'Transition speed']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 1. Distribution of numeric settings (Histograms)
print("\nCreating distribution histograms...")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Distribution of Camera Settings', fontsize=16, fontweight='bold')

for idx, col in enumerate(numeric_cols):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]

    ax.hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel(col, fontweight='bold')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{col} Distribution')
    ax.grid(True, alpha=0.3)

# Remove extra subplot
fig.delaxes(axes[2, 2])

# Add summary stats box
stats_text = f"Total Players: {len(df)}\n"
stats_text += f"Most Common FOV: {df['FOV'].mode().values[0]}\n"
stats_text += f"Avg Height: {df['Height'].mean():.1f}\n"
stats_text += f"Avg Distance: {df['Distance'].mean():.1f}"
axes[2, 2] = fig.add_subplot(3, 3, 9)
axes[2, 2].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[2, 2].axis('off')

plt.tight_layout()
plt.savefig('distributions.png', dpi=300, bbox_inches='tight')
print("Saved: distributions.png")

# 2. Box plots for all numeric settings
print("\nCreating box plots...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Camera Settings Box Plots', fontsize=16, fontweight='bold')

for idx, col in enumerate(numeric_cols):
    row = idx // 4
    col_idx = idx % 4
    ax = axes[row, col_idx]

    bp = ax.boxplot(df[col].dropna(), patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['medians'][0].set_color('red')
    bp['medians'][0].set_linewidth(2)

    ax.set_ylabel(col, fontweight='bold')
    ax.set_title(f'{col}')
    ax.grid(True, alpha=0.3)

# Remove extra subplot
fig.delaxes(axes[1, 3])

plt.tight_layout()
plt.savefig('boxplots.png', dpi=300, bbox_inches='tight')
print("Saved: boxplots.png")

# 3. Correlation heatmap
print("\nCreating correlation heatmap...")
plt.figure(figsize=(10, 8))
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Between Camera Settings', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: correlation_heatmap.png")

# 4. Scatter plots showing relationships
print("\nCreating scatter plot matrix...")
# Select key settings for scatter matrix
key_settings = ['FOV', 'Height', 'Angle', 'Distance', 'Stiffness']
sns.pairplot(df[key_settings].dropna(), diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle('Relationships Between Key Camera Settings', y=1.01, fontsize=16, fontweight='bold')
plt.savefig('scatter_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: scatter_matrix.png")

# 5. Popular settings combinations
print("\nCreating popular settings visualization...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Most Common Camera Setting Values', fontsize=16, fontweight='bold')

# FOV distribution
fov_counts = df['FOV'].value_counts().head(5)
axes[0, 0].bar(fov_counts.index.astype(str), fov_counts.values, color='steelblue', edgecolor='black')
axes[0, 0].set_xlabel('FOV', fontweight='bold')
axes[0, 0].set_ylabel('Number of Players')
axes[0, 0].set_title('Most Popular FOV Settings')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Height distribution
height_counts = df['Height'].value_counts().head(10)
axes[0, 1].bar(height_counts.index.astype(str), height_counts.values, color='coral', edgecolor='black')
axes[0, 1].set_xlabel('Height', fontweight='bold')
axes[0, 1].set_ylabel('Number of Players')
axes[0, 1].set_title('Most Popular Height Settings')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Distance distribution
distance_counts = df['Distance'].value_counts().head(10)
axes[1, 0].bar(distance_counts.index.astype(str), distance_counts.values, color='lightgreen', edgecolor='black')
axes[1, 0].set_xlabel('Distance', fontweight='bold')
axes[1, 0].set_ylabel('Number of Players')
axes[1, 0].set_title('Most Popular Distance Settings')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Angle distribution
angle_counts = df['Angle'].value_counts().head(10)
axes[1, 1].bar(angle_counts.index.astype(str), angle_counts.values, color='plum', edgecolor='black')
axes[1, 1].set_xlabel('Angle', fontweight='bold')
axes[1, 1].set_ylabel('Number of Players')
axes[1, 1].set_title('Most Popular Angle Settings')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('popular_settings.png', dpi=300, bbox_inches='tight')
print("Saved: popular_settings.png")

# 6. Stiffness vs Swivel Speed (interesting relationship)
print("\nCreating stiffness vs swivel speed plot...")
plt.figure(figsize=(10, 8))
plt.scatter(df['Stiffness'], df['Swivel speed'], alpha=0.5, s=50, c='darkblue')
plt.xlabel('Stiffness', fontsize=12, fontweight='bold')
plt.ylabel('Swivel Speed', fontsize=12, fontweight='bold')
plt.title('Stiffness vs Swivel Speed', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(df['Stiffness'].dropna(), df['Swivel speed'].dropna(), 1)
p = np.poly1d(z)
plt.plot(df['Stiffness'].sort_values(), p(df['Stiffness'].sort_values()),
         "r--", linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
plt.legend()

plt.tight_layout()
plt.savefig('stiffness_vs_swivel.png', dpi=300, bbox_inches='tight')
print("Saved: stiffness_vs_swivel.png")

# 7. Camera shake usage
print("\nCreating camera shake analysis...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Camera Shake and Ball Camera Usage', fontsize=16, fontweight='bold')

# Camera shake
camera_shake_counts = df['Camera shake'].value_counts()
colors_shake = ['#ff6b6b', '#4ecdc4']
ax1.pie(camera_shake_counts.values, labels=camera_shake_counts.index, autopct='%1.1f%%',
        colors=colors_shake, startangle=90)
ax1.set_title('Camera Shake Usage')

# Ball camera
ball_camera_counts = df['Ball camera'].value_counts()
colors_ball = ['#95e1d3', '#f38181']
ax2.pie(ball_camera_counts.values, labels=ball_camera_counts.index, autopct='%1.1f%%',
        colors=colors_ball, startangle=90)
ax2.set_title('Ball Camera Setting')

plt.tight_layout()
plt.savefig('categorical_settings.png', dpi=300, bbox_inches='tight')
print("Saved: categorical_settings.png")

print("\n" + "="*50)
print("VISUALIZATION COMPLETE!")
print("="*50)
print("\nGenerated files:")
print("  1. distributions.png - Histograms of all numeric settings")
print("  2. boxplots.png - Box plots showing ranges and outliers")
print("  3. correlation_heatmap.png - Correlation between settings")
print("  4. scatter_matrix.png - Pairwise relationships")
print("  5. popular_settings.png - Most common setting values")
print("  6. stiffness_vs_swivel.png - Relationship between stiffness and swivel")
print("  7. categorical_settings.png - Camera shake and ball camera usage")
print("\n" + "="*50)
