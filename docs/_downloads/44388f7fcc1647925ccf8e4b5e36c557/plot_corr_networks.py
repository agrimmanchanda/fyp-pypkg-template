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
# Data handling
# -------------------------------------
# First, let's define the data set path and relevant variables of interest

path_data = 'datasets/pathology-sample-march-may.csv'

FBC_codes = ["EOS", "MONO", "BASO", "NEUT", "RBC", "WBC", 
                "MCHC", "MCV", "LY", "HCT", "RDW", "HGB", 
                "MCH", "PLT", "MPV", "NRBCA"]

INTEREST_cols = ["_uid", "orderCode", "result", "dateResult"]

#############################

#######################################
# Next, import only variables of interest and FBC panel results
df = pd.read_csv(path_data, usecols=INTEREST_cols)

df = df.loc[df['orderCode'].isin(FBC_codes)]

df = df.dropna() # drop records of patients with NaN _uid

df.reset_index(drop=True, inplace=True)

# Define function to set pid (patient ID) sorted by datetime

def change_pid_datetime_format(df):
    df['pid'] = df['_uid'].str.extract('(\d+)').astype(int)

    pid_col = df.pop('pid')

    df.insert(0, 'pid', pid_col)

    df.drop('_uid', inplace=True, axis=1)

    df.sort_values(by=['pid', 'dateResult'], inplace=True)

    return df

#######################################
# -------------------------------------
# Transform data using TidyWidget
# -------------------------------------

# Parameters
index = ['_uid', 'dateResult', 'orderCode']
value = 'result'

# Create widget
widget = TidyWidget(index=index, value=value)

# Transform (keep all)
transform, duplicated = \
    widget.transform(df, report_duplicated=True)

# Set pid for each patient and sort accordingly
transform_fmt = change_pid_datetime_format(transform)

# Transform (keep first)
transform_first = \
    widget.transform(df, keep='first')

# Set pid for each patient and sort accordingly
transform_first_fmt = change_pid_datetime_format(transform_first)

#######################################
# -------------------------------------
# Correlation matrix
# -------------------------------------

# Obtain the biomarkers DataFrame only
biomarkers_df = transform_fmt.iloc[:,2:]

# Calculate correlation matrix using Pearson Correlation Coefficient
corr_mat = biomarkers_df.dropna().corr()

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