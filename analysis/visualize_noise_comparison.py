"""
Visualization & Analysis for the Math Noise vs. Code Smell Noise Experiment
===========================================================================

Generates the following visualizations:
1. Stabilization Step Distribution (Box Plot) - comparing all 3 groups
2. Confidence Trajectory Comparison (Line Plot) - avg confidence over steps
3. Entropy Trajectory Comparison (Line Plot) - avg entropy over steps
4. Change Rate Bar Chart - how often each group changes the identifier
5. Recovery/Refactoring Rates
6. Statistical Significance Tests (Welch's t-test, Mann-Whitney U)
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

# Style Configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color Palette
COLORS = {
    'math_noise': '#E74C3C',    # Red - mathematical noise
    'smell_noise': '#F39C12',   # Orange - code smell noise
    'control': '#27AE60',       # Green - clean code control
}

GROUP_LABELS = {
    'math_noise': 'Mathematical Noise\n(Mask Tokens)',
    'smell_noise': 'Code Smell Noise\n(Bad Naming)',
    'control': 'Control\n(Clean Code)',
}


def load_results(csv_path):
    """Load the detail CSV results."""
    df = pd.read_csv(csv_path)
    # Parse trajectory columns from JSON strings
    df['confidence_traj'] = df['confidence_trajectory'].apply(
        lambda x: json.loads(x) if isinstance(x, str) and x.startswith('[') else []
    )
    df['entropy_traj'] = df['entropy_trajectory'].apply(
        lambda x: json.loads(x) if isinstance(x, str) and x.startswith('[') else []
    )
    return df


def plot_stabilization_boxplot(df, output_dir):
    """Box plot comparing stabilization steps across groups."""
    fig, ax = plt.subplots(figsize=(10, 6))

    groups = ['math_noise', 'smell_noise', 'control']
    data = []
    labels = []
    colors = []

    for g in groups:
        gdf = df[df['group'] == g]
        valid = pd.to_numeric(gdf['stabilization_step'], errors='coerce')
        valid = valid[valid >= 0]
        if not valid.empty:
            data.append(valid.values)
            labels.append(GROUP_LABELS[g])
            colors.append(COLORS[g])

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                    showmeans=True, meanprops=dict(marker='D', markerfacecolor='white', markersize=8))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Stabilization Step')
    ax.set_title('Identifier Stabilization Step by Noise Type\n'
                 '(Lower = faster convergence, analogous to faster denoising)')
    ax.grid(axis='y', alpha=0.3)

    # Add sample counts
    for i, g in enumerate(groups):
        gdf = df[df['group'] == g]
        valid = pd.to_numeric(gdf['stabilization_step'], errors='coerce')
        valid = valid[valid >= 0]
        ax.text(i + 1, ax.get_ylim()[0] - 5, f'n={len(valid)}',
                ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    path = os.path.join(output_dir, 'stabilization_step_boxplot.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_trajectory_comparison(df, output_dir):
    """Line plots comparing confidence and entropy trajectories."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    groups = ['math_noise', 'smell_noise', 'control']
    x_points = np.linspace(0, 1, 10)  # Normalized step (0% to 100%)

    for g in groups:
        gdf = df[df['group'] == g]

        # Confidence trajectory
        conf_trajs = gdf['confidence_traj'].tolist()
        valid_trajs = [t for t in conf_trajs if len(t) == 10]
        if valid_trajs:
            mean_traj = np.mean(valid_trajs, axis=0)
            std_traj = np.std(valid_trajs, axis=0)
            ax1.plot(x_points, mean_traj, color=COLORS[g], label=GROUP_LABELS[g].replace('\n', ' '),
                     linewidth=2)
            ax1.fill_between(x_points, mean_traj - std_traj, mean_traj + std_traj,
                             color=COLORS[g], alpha=0.15)

        # Entropy trajectory
        ent_trajs = gdf['entropy_traj'].tolist()
        valid_trajs = [t for t in ent_trajs if len(t) == 10]
        if valid_trajs:
            mean_traj = np.mean(valid_trajs, axis=0)
            std_traj = np.std(valid_trajs, axis=0)
            ax2.plot(x_points, mean_traj, color=COLORS[g], label=GROUP_LABELS[g].replace('\n', ' '),
                     linewidth=2)
            ax2.fill_between(x_points, mean_traj - std_traj, mean_traj + std_traj,
                             color=COLORS[g], alpha=0.15)

    ax1.set_xlabel('Normalized Diffusion Step (0% → 100%)')
    ax1.set_ylabel('Confidence (Softmax Probability)')
    ax1.set_title('Confidence Trajectory at Identifier Positions')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    ax2.set_xlabel('Normalized Diffusion Step (0% → 100%)')
    ax2.set_ylabel('Shannon Entropy (nats)')
    ax2.set_title('Entropy Trajectory at Identifier Positions')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    plt.tight_layout()
    path = os.path.join(output_dir, 'trajectory_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_change_rates(df, output_dir):
    """Bar chart showing change rates and recovery/refactoring rates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    groups = ['math_noise', 'smell_noise', 'control']

    # Change Rate
    change_rates = []
    for g in groups:
        gdf = df[df['group'] == g]
        rate = gdf['changed'].astype(bool).mean() * 100 if not gdf.empty else 0
        change_rates.append(rate)

    bars = ax1.bar([GROUP_LABELS[g] for g in groups], change_rates,
                   color=[COLORS[g] for g in groups], alpha=0.8, edgecolor='white', linewidth=1.5)

    for bar, rate in zip(bars, change_rates):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                 f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

    ax1.set_ylabel('Change Rate (%)')
    ax1.set_title('Identifier Change Rate by Noise Type\n'
                  '(How often does the model modify the identifier?)')
    ax1.set_ylim(0, 110)
    ax1.grid(axis='y', alpha=0.3)

    # Recovery Rate (Math) vs Refactoring Rate (Smell) vs Stability (Control)
    special_rates = {}

    math_df = df[df['group'] == 'math_noise']
    if not math_df.empty and 'recovered_gt' in math_df.columns:
        special_rates['Math Noise\nRecovery → GT'] = {
            'rate': math_df['recovered_gt'].astype(bool).mean() * 100,
            'color': COLORS['math_noise']
        }

    smell_df = df[df['group'] == 'smell_noise']
    if not smell_df.empty:
        special_rates['Smell Noise\nRefactoring Rate'] = {
            'rate': smell_df['changed'].astype(bool).mean() * 100,
            'color': COLORS['smell_noise']
        }

    ctrl_df = df[df['group'] == 'control']
    if not ctrl_df.empty:
        # Stability = 100% - change rate
        special_rates['Control\nStability Rate'] = {
            'rate': (1 - ctrl_df['changed'].astype(bool).mean()) * 100,
            'color': COLORS['control']
        }

    if special_rates:
        names = list(special_rates.keys())
        rates = [v['rate'] for v in special_rates.values()]
        colors = [v['color'] for v in special_rates.values()]

        bars2 = ax2.bar(names, rates, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        for bar, rate in zip(bars2, rates):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                     f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

        ax2.set_ylabel('Rate (%)')
        ax2.set_title('Key Rates by Group\n'
                      '(Recovery = math noise → GT; Refactoring = smell changed;\n'
                      'Stability = control unchanged)')
        ax2.set_ylim(0, 110)
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'change_rates.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_entropy_confidence_scatter(df, output_dir):
    """Scatter plot: avg entropy vs avg confidence, colored by group."""
    fig, ax = plt.subplots(figsize=(10, 8))

    groups = ['math_noise', 'smell_noise', 'control']
    for g in groups:
        gdf = df[df['group'] == g]
        if gdf.empty:
            continue
        ax.scatter(
            pd.to_numeric(gdf['avg_entropy']),
            pd.to_numeric(gdf['avg_confidence']),
            c=COLORS[g], label=GROUP_LABELS[g].replace('\n', ' '),
            alpha=0.5, s=30, edgecolors='white', linewidth=0.5
        )

    ax.set_xlabel('Average Entropy (nats)')
    ax.set_ylabel('Average Confidence')
    ax.set_title('Entropy vs. Confidence Distribution by Noise Type\n'
                 '(Lower entropy + higher confidence = more certain predictions)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'entropy_confidence_scatter.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def run_statistical_tests(df, output_dir):
    """Run statistical tests comparing groups and save to text file."""
    output_lines = []
    output_lines.append("=" * 60)
    output_lines.append("STATISTICAL ANALYSIS: Math Noise vs. Code Smell Noise")
    output_lines.append("=" * 60)

    groups_data = {}
    for g in ['math_noise', 'smell_noise', 'control']:
        gdf = df[df['group'] == g]
        stab = pd.to_numeric(gdf['stabilization_step'], errors='coerce')
        valid_stab = stab[stab >= 0]
        groups_data[g] = {
            'stabilization': valid_stab.values,
            'confidence': pd.to_numeric(gdf['avg_confidence']).values,
            'entropy': pd.to_numeric(gdf['avg_entropy']).values,
        }

    # Pairwise comparisons
    pairs = [
        ('math_noise', 'smell_noise'),
        ('math_noise', 'control'),
        ('smell_noise', 'control'),
    ]

    for metric in ['stabilization', 'confidence', 'entropy']:
        output_lines.append(f"\n--- Metric: {metric} ---")
        for g1, g2 in pairs:
            d1 = groups_data[g1][metric]
            d2 = groups_data[g2][metric]

            if len(d1) < 2 or len(d2) < 2:
                output_lines.append(f"  {g1} vs {g2}: Insufficient data")
                continue

            # Welch's t-test
            t_stat, t_p = stats.ttest_ind(d1, d2, equal_var=False)

            # Mann-Whitney U test (non-parametric)
            u_stat, u_p = stats.mannwhitneyu(d1, d2, alternative='two-sided')

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(d1) + np.var(d2)) / 2)
            cohens_d = (np.mean(d1) - np.mean(d2)) / pooled_std if pooled_std > 0 else 0

            output_lines.append(f"\n  {g1} vs {g2}:")
            output_lines.append(f"    {g1}: mean={np.mean(d1):.4f}, std={np.std(d1):.4f}, n={len(d1)}")
            output_lines.append(f"    {g2}: mean={np.mean(d2):.4f}, std={np.std(d2):.4f}, n={len(d2)}")
            output_lines.append(f"    Welch's t-test:   t={t_stat:.4f}, p={t_p:.6f} {'***' if t_p < 0.001 else '**' if t_p < 0.01 else '*' if t_p < 0.05 else 'ns'}")
            output_lines.append(f"    Mann-Whitney U:   U={u_stat:.0f}, p={u_p:.6f} {'***' if u_p < 0.001 else '**' if u_p < 0.01 else '*' if u_p < 0.05 else 'ns'}")
            output_lines.append(f"    Cohen's d:        {cohens_d:.4f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small' if abs(cohens_d) > 0.2 else 'negligible'})")

    output_lines.append(f"\n{'='*60}")
    output_lines.append("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

    report = "\n".join(output_lines)
    print(report)

    path = os.path.join(output_dir, 'statistical_analysis.txt')
    with open(path, 'w') as f:
        f.write(report)
    print(f"\n  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Math vs Smell Noise experiment results")
    parser.add_argument("csv_path", type=str, help="Path to the detail CSV file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: same dir as CSV)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.csv_path), 'noise_comparison_plots')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from: {args.csv_path}")
    df = load_results(args.csv_path)
    print(f"Loaded {len(df)} records.")
    print(f"Groups: {df['group'].value_counts().to_dict()}")

    print("\nGenerating visualizations...")
    plot_stabilization_boxplot(df, args.output_dir)
    plot_trajectory_comparison(df, args.output_dir)
    plot_change_rates(df, args.output_dir)
    plot_entropy_confidence_scatter(df, args.output_dir)

    print("\nRunning statistical tests...")
    run_statistical_tests(df, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
