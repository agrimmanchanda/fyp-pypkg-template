"""
Biomarker distributions in dataset
===========================================

Using ``seaborn`` library to visualise biomarker distributions

"""
#######################################
# Import the relevant libraries first
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

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
# Preprocessing step: obtain FBC panel
# -------------------------------------

# Obtain the biomarkers DataFrame only
biomarkers_df = df[FBC_CODES].dropna(subset=FBC_CODES)
biomarkers_df_copy = biomarkers_df.copy(deep=True)
biomarkers_data = biomarkers_df.values

######################################################
# ----------------------------------------------------
# Plot distributions and histograms for each biomarker
# ----------------------------------------------------

for col in biomarkers_df_copy.columns:
    plt.figure(figsize=(20,10))
    plt.suptitle(f'Distribution and boxplot for biomarker: {col}', 
    fontweight='bold', fontsize=25)
    
    plt.subplot(1,2,1)
    sns.distplot(biomarkers_df[col].values, bins=50, 
    kde_kws={'color': 'red','linewidth': 2, }, hist_kws={'edgecolor':'black'})
    plt.xlabel(f'{col}', fontsize=18)
    plt.ylabel('Density', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    plt.subplot(1,2,2)
    sns.boxplot(x=biomarkers_df[col])
    plt.xlabel(f'{col}', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

#######################################
# -------------------------------------
# Plot same histograms without outliers
# -------------------------------------

# Remove data outliers based on absolute Z-Score value < 3
#biomarkers_df[(np.abs(stats.zscore(biomarkers_df)) < 3).all(axis=1)]

# Remove values based on Q(1/3) +- 1.5 * IQR method
q1, q3 = biomarkers_df.quantile(0.25), biomarkers_df.quantile(0.75)
IQR = q3 - q1

# New dataframe with outlier values removed
new_biomarkers_df = biomarkers_df[~((biomarkers_df < (q1 - 1.5 * IQR)) | 
(biomarkers_df > (q3 + 1.5 * IQR))).any(axis=1)]

# Plot distribution and boxplots 
for col in biomarkers_df_copy.columns:
    plt.figure(figsize=(20,10))
    plt.suptitle(f'Distribution and boxplot for biomarker: {col}', 
    fontweight='bold', fontsize=25)
    
    plt.subplot(1,2,1)
    sns.distplot(new_biomarkers_df[col].values, bins=50, 
    kde_kws={'color': 'red','linewidth': 2, }, hist_kws={'edgecolor':'black'})
    plt.xlabel(f'{col}', fontsize=18)
    plt.ylabel('Density', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    plt.subplot(1,2,2)
    sns.boxplot(x=new_biomarkers_df[col])
    plt.xlabel(f'{col}', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
