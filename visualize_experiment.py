
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def create_visualizations(csv_file):
    # Set the aesthetic style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 300
    })

    # Load data
    df = pd.read_csv(csv_file)
    
    # Preprocessing: Cleanup
    # Ensure step columns are numeric
    df['step_good'] = pd.to_numeric(df['step_good'], errors='coerce')
    df['step_bad'] = pd.to_numeric(df['step_bad'], errors='coerce')
    
    # Drop rows with NaN steps if they exist
    df = df.dropna(subset=['step_good', 'step_bad'])
    
    # Filter for the 50 datapoints (repeated ~10 times)
    # The user mentioned多余数据需要删除, we can keep only first 10 runs per id if needed
    df = df.sort_values(['id', 'run_id'])
    df = df.groupby('id').head(10).reset_index(drop=True)

    # Calculate Aggregate Statistics for Labels
    mean_good = df['step_good'].mean()
    mean_bad = df['step_bad'].mean()
    std_good = df['step_good'].std()
    std_bad = df['step_bad'].std()
    
    # Perform t-test for significance
    t_stat, p_val = stats.ttest_rel(df['step_good'], df['step_bad'])

    # Create a figure with subplots - 2x3 grid for more details
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # --- Plot 1: Violin Plot (Distribution) ---
    ax1 = fig.add_subplot(gs[0, 0])
    plot_df = pd.melt(df[['step_good', 'step_bad']], var_name='Condition', value_name='Convergence Step')
    plot_df['Condition'] = plot_df['Condition'].map({'step_good': 'Well-Named', 'step_bad': 'Poorly-Named'})
    
    sns.violinplot(data=plot_df, x='Condition', y='Convergence Step', inner="box", hue='Condition', palette=["#4C72B0", "#C44E52"], legend=False, ax=ax1)
    ax1.set_title("Distribution of Convergence Steps", fontweight='bold')
    ax1.set_ylabel("Step Number")
    
    # --- Plot 2: Paired Mean Comparison ---
    ax2 = fig.add_subplot(gs[0, 1])
    id_means = df.groupby('id')[['step_good', 'step_bad']].mean().reset_index()
    for i, row in id_means.iterrows():
        ax2.plot([0, 1], [row['step_good'], row['step_bad']], color='gray', alpha=0.2, linewidth=0.8)
        ax2.scatter([0, 1], [row['step_good'], row['step_bad']], 
                    c=['#4C72B0', '#C44E52'], s=20, alpha=0.5, edgecolors='none', zorder=5)
    
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Well-Named', 'Poorly-Named'])
    ax2.set_xlim(-0.3, 1.3)
    ax2.set_title(f"Paired Mean Step (per ID)\nWilcoxon p={stats.wilcoxon(id_means['step_good'], id_means['step_bad'])[1]:.2e}", fontweight='bold')
    ax2.set_ylabel("Mean Step")

    # --- Plot 3: Stability Comparison (STD) ---
    ax3 = fig.add_subplot(gs[0, 2])
    id_stds = df.groupby('id')[['step_good', 'step_bad']].std().reset_index()
    std_plot_df = pd.melt(id_stds[['step_good', 'step_bad']], var_name='Condition', value_name='Step STD')
    std_plot_df['Condition'] = std_plot_df['Condition'].map({'step_good': 'Well-Named', 'step_bad': 'Poorly-Named'})
    
    sns.boxplot(data=std_plot_df, x='Condition', y='Step STD', palette=["#4C72B0", "#C44E52"], hue='Condition', legend=False, ax=ax3)
    sns.stripplot(data=std_plot_df, x='Condition', y='Step STD', color=".3", size=4, alpha=0.4, jitter=True, ax=ax3)
    ax3.set_title("Convergence Stability (Lower is Better)", fontweight='bold')
    ax3.set_ylabel("STD of Steps across 10 Runs")

    # --- Plot 4: Cumulative Success Rate ---
    ax4 = fig.add_subplot(gs[1, 0])
    steps = np.sort(df['step_good'].unique())
    max_step = int(max(df['step_good'].max(), df['step_bad'].max()))
    x_range = np.linspace(0, max_step, 100)
    
    ecdf_good = [sum(df['step_good'] <= x) / len(df) for x in x_range]
    ecdf_bad = [sum(df['step_bad'] <= x) / len(df) for x in x_range]
    
    ax4.plot(x_range, ecdf_good, label='Well-Named', color="#4C72B0", lw=2)
    ax4.plot(x_range, ecdf_bad, label='Poorly-Named', color="#C44E52", lw=2)
    ax4.fill_between(x_range, ecdf_good, alpha=0.1, color="#4C72B0")
    ax4.fill_between(x_range, ecdf_bad, alpha=0.1, color="#C44E52")
    ax4.set_title("Cumulative Convergence Rate", fontweight='bold')
    ax4.set_xlabel("Diffusion Step")
    ax4.set_ylabel("Fraction Converged")
    ax4.legend()

    # --- Plot 5: KDE Overlap ---
    ax5 = fig.add_subplot(gs[1, 1])
    sns.kdeplot(df['step_good'], fill=True, label='Well-Named', color="#4C72B0", alpha=0.5, ax=ax5)
    sns.kdeplot(df['step_bad'], fill=True, label='Poorly-Named', color="#C44E52", alpha=0.5, ax=ax5)
    ax5.set_title("Step Density Comparison", fontweight='bold')
    ax5.set_xlabel("Step")
    ax5.legend()

    # --- Plot 6: Statistics Table Effect ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    stats_text = (
        f"Summary Statistics\n"
        f"------------------\n"
        f"Good Mean: {mean_good:.1f} ± {std_good:.1f}\n"
        f"Bad Mean:  {mean_bad:.1f} ± {std_bad:.1f}\n\n"
        f"Mean Difference: {mean_bad - mean_good:.1f}\n"
        f"Relative Delay: {((mean_bad/mean_good)-1)*100:.1f}%\n\n"
        
    )
    ax6.text(0.1, 0.5, stats_text, family='monospace', fontsize=14, verticalalignment='center')
    ax6.set_title("Statistical summary", fontweight='bold')

    plt.suptitle("Impact of Identifier Naming on Diffusion Convergence (Diffucoder)", fontsize=26, fontweight='bold', y=1.02)
    
    output_filename = csv_file.replace('.csv', '_publication_plot_v2.png')
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.savefig(output_filename.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Enhanced plots saved to {output_filename}")
    plt.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "/Users/davidhuang/Desktop/CoRefusion/results/dreamcoder_scale_naming_exp_20260122_053530.csv"
    
    create_visualizations(csv_path)
