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
from labimputer.utils.load_dataset import *

#######################################
# -------------------------------------
# Data import 
# -------------------------------------

# Set relative data path and set FBC panel list
path_data = '../resources/datasets/nhs/Transformed_First_FBC_dataset.csv'

FBC_CODES = ["EOS", "MONO", "BASO", "NEUT", "RBC", "WBC", 
                "MCHC", "MCV", "LY", "HCT", "RDW", "HGB", 
                "MCH", "PLT", "MPV", "NRBCA"]

# Read data and drop Nan _uid records
df = pd.read_csv(path_data).dropna(subset=['pid'])

df.reset_index(drop=True, inplace=True)

#######################################
# -------------------------------------
# Remove data outliers
# -------------------------------------

# Obtain the biomarkers DataFrame only
biomarkers_df = df[FBC_CODES].dropna(subset=FBC_CODES)

# Remove outliers from dataset
complete_profiles, outlier_count = remove_data_outliers(biomarkers_df)

# Constant variables to drop
drop_features = ['BASO', 'NRBCA']

complete_profiles = complete_profiles.drop(drop_features, axis=1)

# Create array of the data
biomarkers_data = complete_profiles.values

outlier_count

#######################################
# -------------------------------------
# Calculate data correlations
# ------------------------------------- 

# Calculate correlation matrix using Pearson Correlation Coefficient
corr_mat = complete_profiles.corr(method='pearson')

# Create a mask
corr_mask = np.triu(np.ones_like(corr_mat, dtype=bool))

# Plot seaborn heatmap, histogram and PDF of correlation values.

plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.title('Correlation Matrix for FBC panel', 
        fontweight='bold', 
        fontsize=15)

min_v = corr_mat.values.min()
max_v = corr_mat.values.max()
ax = sns.heatmap(
    corr_mat,
    mask=corr_mask, 
    vmin=min_v, 
    vmax=max_v, 
    center=0,
    cmap='coolwarm',
    square=True,
    annot = True,
    annot_kws={"fontsize":8}
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right',
    fontsize=10
);

ax.set_yticklabels(
    ax.get_yticklabels(),
    fontsize=10
);

ax.set_yticklabels(complete_profiles.columns)
ax.set_xticklabels(complete_profiles.columns)

plt.subplot(1,2,2)
plt.title('Histogram and PDF of FBC panel correlations', 
        fontweight='bold', 
        fontsize=15)
sns.distplot(corr_mat.values.reshape(-1), 
            bins=50, 
            kde_kws={'color': 'red','linewidth': 2}, 
            hist_kws={'edgecolor':'black', 'color': 'midnightblue'})
plt.ylabel("Density", fontsize=18)
plt.xlabel("Correlation values", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid()
plt.show()

#######################################
# -------------------------------------
# Correlation statistics
# -------------------------------------

# Print the mean and std correlation value for each biomarker

statistics = pd.DataFrame(index=corr_mat.columns)

statistics.loc[:, 'Mean'] = np.array(corr_mat.mean())
statistics.loc[:, 'Std'] = np.array(corr_mat.std())

statistics

#######################################
# -------------------------------------
# Plotting graphs using networkx
# -------------------------------------

# Transform correlation matrix in a links data frame (3 columns only):
links = corr_mat.stack().reset_index()
links.columns = ['var1', 'var2', 'value']

# Define thresholds to investigate
thresholds = [x/10 for x in range(1,5)]

# Define each subplot size
plt.figure(figsize=(15, 15))

for plot_idx, threshold in enumerate(thresholds, start=1):
    
    # Keep only correlation over a threshold and remove self-correlations
    idx1 = abs(links['value'] > threshold) # absolute value above threshold 
    idx2 = links['var1'] != links['var2'] # self correlation
    links_filtered=links[idx1 & idx2]

    # Build graph
    G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2', 'value')

    # Get weights of the edges 
    weights = tuple(nx.get_edge_attributes(G,'value').values())
    
    # Get the degree of the nodes 
    degree = [v for k, v in nx.degree(G)]
    
    # Set for circular networks only
    pos = nx.circular_layout(G)

    plt.subplot(2, 2, plot_idx)

    # Draw the graph
    nx.draw(G, with_labels=True,pos=pos, 
            edge_cmap = plt.cm.Blues,  
            node_color='skyblue', 
            node_size=[d * 800 for d in degree], 
            linewidths=2, 
            font_size=16, 
            edge_color=weights, 
            edgecolors='black',
            font_weight='bold',
            width=3
            )
    plt.title(f'Network with weight threshold: {threshold}', 
            fontweight='bold', 
            fontsize=16)
    plt.axis('off')
    axis = plt.gca()
    axis.set_xlim([1.5*x for x in axis.get_xlim()])
    axis.set_ylim([1.5*y for y in axis.get_ylim()])

# Space out the plots
plt.tight_layout()

# Show
plt.show()