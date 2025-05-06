!pip install pandas openpyxl matplotlib
import pandas as pd
import matplotlib.pyplot as plt

file_path = "R_data.xlsx" #put here the .xlsx file path
df = pd.read_excel(file_path, engine='openpyxl')

# Group columns by keywords
datasets = {
    'All': df.filter(like='all'),
    'TACO': df.filter(like='TACO'),
    'Plasto': df.filter(like='Plasto')
}

stats = {}
for name, data in datasets.items():
    stats[name] = {
        'mean': data.mean(),
        'std': data.std()
    }
centralized_limits = {
    'All': [(0.679, 'ALL'), (0.566, 'TACO'), (0.625, 'Plasto')],
    'TACO': [(0.543, 'ALL'), (0.511, 'TACO'),(0.454, 'Plasto') ],
    'Plasto': [(0.786, 'ALL'), (0.616, 'TACO'), (0.759, 'Plasto')]
}

'''
#if you are doing the Recall metric use this insted:
centralized_limits = {
    'All': [(0.577, 'ALL'), (0.501, 'TACO'), (0.555, 'Plasto')],
    'TACO': [(0.487, 'ALL'), (0.432, 'TACO'),(0.391, 'Plasto') ],
    'Plasto': [(0.679, 'ALL'), (0.543, 'TACO'), (0.659, 'Plasto')]
}
'''

for name, stat in stats.items():
    fig, ax = plt.subplots(figsize=(6, 5))  # Separate figure for each dataset

    
    stat['mean'].plot(
        kind='bar',
        yerr=stat['std'],
        ax=ax,
        capsize=5,
        color='skyblue'
    )
    for value, label in centralized_limits[name]:
        ax.axhline(y=value, color='red', linestyle='--', alpha=0.6)
        ax.text(x=0.1, y=value + 0.005, s=label, color='red', fontsize=9)

  
    ax.set_title(f'{name} Dataset - Recall', fontsize=14)
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('R', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Optional: Save each figure
    # plt.savefig(f'{name}_mAP50_plot.png', dpi=300)

    plt.show()
