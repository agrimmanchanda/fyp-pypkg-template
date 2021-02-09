"""
Plotting an interesting graph from data set
==============================

Example ``seaborn`` to visualise missing values per variable

"""
#######################################
# Import the relevant libraries first
import numpy as np 
import pandas as pd
import seaborn as sns

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

#######################################
# -------------------------------------
# Sampling and plotting
# -------------------------------------
# Then, randomly remove 5% of values from the result variable
df['result'] = df['result'].sample(frac=0.95)

#######################################
# Plot a seaborn heatmap visualising the missing values
sns.heatmap(df.isnull(), cbar=False)