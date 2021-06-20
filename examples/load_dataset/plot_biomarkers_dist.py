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
from sklearn import preprocessing
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
# Violin plots for raw data
# -------------------------------------

# Obtain the biomarkers DataFrame only
biomarkers_df = df[FBC_CODES].dropna(subset=FBC_CODES)

# Set figure size
plt.figure(figsize=(15,15))

# Set single title for all figures
plt.suptitle('Violin plot for raw data', 
            fontweight='bold', 
            fontsize=20)

for plot_idx, biomarker in enumerate(biomarkers_df, start=1):
    
    plt.subplot(4,4,plot_idx)
    
    sns.violinplot(data=biomarkers_df[biomarker], 
                color='skyblue',
                orient='h')
    
    plt.xticks(fontsize=12)
    plt.xlabel(f'{biomarker}', 
            fontweight='bold', 
            fontsize=12)

# Space out plots 
plt.tight_layout()
    
# Show
plt.show()

#######################################
# -------------------------------------
# Violin plots without outliers
# -------------------------------------

# Remove outliers from dataset
complete_profiles, outlier_count = remove_data_outliers(biomarkers_df)

# Set figure size
plt.figure(figsize=(15,15))

# Set single title for all figures
plt.suptitle('Violin plot for complete profiles', 
            fontweight='bold', 
            fontsize=20)

for plot_idx, biomarker in enumerate(complete_profiles, start=1):
    
    plt.subplot(4,4,plot_idx)
    
    sns.violinplot(data=complete_profiles[biomarker], 
                color='skyblue',
                orient='h')
    
    plt.xticks(fontsize=12)
    plt.xlabel(f'{biomarker}', 
            fontweight='bold', 
            fontsize=12)

# Space out plots 
plt.tight_layout()
    
# Show
plt.show()

#######################################
# -------------------------------------
# Jarque-Bera (JB) test for normality
# -------------------------------------

# Standardise data 
std_complete_profiles = preprocessing.StandardScaler().fit_transform(complete_profiles)

# Create DataFrame of standardised data
norm_profiles = pd.DataFrame(data=std_complete_profiles, columns=complete_profiles.columns)

# Dataframe to store JB Test 
norm_scores = pd.DataFrame(columns=complete_profiles.columns)

# Note: JB test is only valid on datasets with n > 2000 where n are samples
# in the dataset. The complete profile data set contains 56271 records. 

# Loop
for biomarker in complete_profiles.columns:

    # Calculate and store JB test statistic and p-value for each biomarker
    jb_test = stats.jarque_bera(complete_profiles[biomarker])
    norm_scores[biomarker] = jb_test

norm_scores.index = ['Test Statistic', 'P-Value']

norm_scores.T