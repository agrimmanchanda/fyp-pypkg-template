"""
Network graphs to visualise correlations
===========================================

Using ``networkx`` library to visualise biomarker correlations

"""
#######################################
# Import the relevant libraries first
import numpy as np 
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from pkgname.utils.widgets import TidyWidget

#######################################
# -------------------------------------
# Data import 
# -------------------------------------

# Set relative data path and set FBC panel list
path_data = 'datasets/Transformed_First_FBC_dataset.csv'

FBC_CODES = ["EOS", "MONO", "BASO", "NEUT", "RBC", "WBC", 
                "MCHC", "MCV", "LY", "HCT", "RDW", "HGB", 
                "MCH", "PLT", "MPV", "NRBCA"]

# Read data and drop Nan _uid records
df = pd.read_csv(path_data).dropna(subset=['pid'])

df.reset_index(drop=True, inplace=True)

#######################################
# -------------------------------------
# Split data into input and output
# -------------------------------------

# Obtain the biomarkers DataFrame only
biomarkers_df = df[FBC_CODES].dropna(subset=FBC_CODES)

biomarkers_original_df_copy = biomarkers_df.copy(deep=True)

biomarkers_data = biomarkers_df.values

# Calculate correlation matrix using Pearson Correlation Coefficient
corr_mat = biomarkers_df.dropna().corr(method='pearson')

# Plot seaborn heatmap, histogram and PDF of correlation values.

plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.title('Correlation Matrix for FBC panel', fontweight='bold', fontsize=15)

min_v = corr_mat.values.min()
ax = sns.heatmap(
    corr_mat, 
    vmin=min_v, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot = True,
    annot_kws={"fontsize":8}
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right',
    fontsize=15
);

ax.set_yticklabels(
    ax.get_yticklabels(),
    fontsize=15
);

ax.set_yticklabels(biomarkers_df.columns)
ax.set_xticklabels(biomarkers_df.columns)

plt.subplot(1,2,2)
plt.title('Histogram and PDF of FBC panel correlations', fontweight='bold', fontsize=15)
sns.distplot(corr_mat.values.reshape(-1), bins=50, kde_kws={'color': 'red','linewidth': 2, }, hist_kws={'edgecolor':'black'})
plt.ylabel("Density", fontsize=18)
plt.xlabel("Correlation values", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid()
plt.show()

# Print the mean correlation value for each biomarker

print("\nSorted mean correlation values by biomarkers:")
print(corr_mat.mean(1).sort_values(ascending=False))

#######################################
# -------------------------------------
# Plotting graphs using networkx
# -------------------------------------

thresholds = [x/10 for x in range(4,10)]

# for each threshold value
for threshold in thresholds:

    # Transform correlation matrix in a links data frame (3 columns only):
    links = corr_mat.stack().reset_index()
    links.columns = ['var1', 'var2', 'value']
    
    # Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)
    links_filtered=links.loc[ (links['value'] > threshold) & (links['var1'] != links['var2']) ]
    
    # Build graph
    G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2', 'value')

    pos = nx.spring_layout(G, k=0.45, iterations=20)

    plt.figure(figsize=(15,5))
    plt.title(f'Graph with Weight Threshold: {threshold}', fontweight='bold', fontsize=16)
    # Plot the network:
    nx.draw(G, with_labels=True,pos=pos, node_color='orange', node_size=1500, linewidths=2, font_size=12, edge_color='black', edgecolors='black')
    plt.axis('off')
    axis = plt.gca()
    axis.set_xlim([1.2*x for x in axis.get_xlim()])
    axis.set_ylim([1.2*y for y in axis.get_ylim()])
    plt.show()