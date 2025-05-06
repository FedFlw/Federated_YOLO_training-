import matplotlib.pyplot as plt
import pandas as pd

colors = {'taco': 'blue', 'plasto': 'red', 'all': 'green'}
medianprops = {'linewidth': 0}
meanprops = {'markerfacecolor': 'black', 'markeredgecolor': 'black'}

file_path = "rq1_recall_data.xlsx"  # put here the .xlsx file path
df = pd.read_excel(file_path, engine='openpyxl')

# Group columns by keywords
datasets = {
    'All': df.filter(like='all'),
    'TACO': df.filter(like='TACO'),
    'Plasto': df.filter(like='Plasto')
}

# Baseline for Recall metric:
centralized_limits = {
    'All': [(0.577, 'ALL'), (0.501, 'TACO'), (0.555, 'Plasto')],
    'TACO': [(0.487, 'ALL'), (0.432, 'TACO'), (0.391, 'Plasto')],
    'Plasto': [(0.679, 'ALL'), (0.543, 'TACO'), (0.659, 'Plasto')]
}

for test in datasets.keys():
    fig, ax = plt.subplots(figsize=(6, 5))  # Separate figure for each dataset

    # Plots horizontal lines for baseline (centralized training)
    for value, label in centralized_limits[test]:
        ax.axhline(y=value, color=colors[label.lower()], linestyle='-', alpha=0.6)
        # ax.text(x=1.0, y=value + 0.005, s=label, color='red', fontsize=9)

    # Plots FL accuracy distribution by scenario (n. nodes and data split)
    x = [col for col in df.columns.tolist() if test.lower() in col.lower()]
    y = df[x].values
    ax.boxplot(y, widths=0.75, showmeans=True, medianprops=medianprops, meanprops=meanprops)

    ax.set_ylim(0.2, 0.8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xticklabels([col.split(' ')[0] for col in x], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)

    plt.tight_layout()
    plt.savefig(f'{test.lower()}_recall_plot.pdf', dpi=600)
    plt.show()

file_path = "rq2_mAP50_data.xlsx"  # put here the .xlsx file path
df = pd.read_excel(file_path, engine='openpyxl')

# Group columns by keywords
datasets = {
    'All': df.filter(like='all'),
    'TACO': df.filter(like='TACO'),
    'Plasto': df.filter(like='Plasto')
}

# Baseline for mAP50 metric:
centralized_limits = {
    'All': [(0.679, 'ALL'), (0.566, 'TACO'), (0.625, 'Plasto')],
    'TACO': [(0.543, 'ALL'), (0.511, 'TACO'), (0.454, 'Plasto')],
    'Plasto': [(0.786, 'ALL'), (0.616, 'TACO'), (0.759, 'Plasto')]
}

for test in datasets.keys():
    fig, ax = plt.subplots(figsize=(6, 5))  # Separate figure for each dataset

    # Plots horizontal lines for baseline (centralized training)
    for value, label in centralized_limits[test]:
        ax.axhline(y=value, color=colors[label.lower()], linestyle='-', alpha=0.6)
        # ax.text(x=1.0, y=value + 0.005, s=label, color='red', fontsize=9)

    # Plots FL accuracy distribution by scenario (n. nodes and data split)
    x = [col for col in df.columns.tolist() if test.lower() in col.lower()]
    y = df[x].values
    ax.boxplot(y, widths=0.75, showmeans=True, medianprops=medianprops, meanprops=meanprops)

    ax.set_ylim(0.2, 0.8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xticklabels([col.split(' ')[0] for col in x], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)

    plt.tight_layout()
    plt.savefig(f'{test.lower()}_mAP50_plot.pdf', dpi=600)
    plt.show()
