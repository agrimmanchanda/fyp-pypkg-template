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
from sklearn import preprocessing
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
# Preprocessing step: normalise
# -------------------------------------

# Obtain the biomarkers DataFrame only
biomarkers_df = transform_fmt.iloc[:,2:].dropna()
biomarkers_df_copy = biomarkers_df.copy(deep=True)
biomarkers_data = biomarkers_df.values

# Normalise using minmax scaler
min_max_scaler = preprocessing.MinMaxScaler()
val_scaled = min_max_scaler.fit_transform(biomarkers_data)
biomarkers_df = pd.DataFrame(val_scaled, columns=[col for col in biomarkers_df_copy.columns])

# Can use df.melt() method
# biomarkers_dfm = biomarkers_df.melt(var_name='biomarkers')

#######################################
# -------------------------------------
# Plot histograms for each biomarker
# -------------------------------------

for col in biomarkers_df_copy.columns:
    plt.figure(figsize=(15,10))
    plt.title(f'Histogram for biomarker: {col}', fontweight='bold', fontsize=20)
    plt.xlabel('Normalised value', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    hist = biomarkers_df_copy[col].hist(bins=50)
    hist.plot(grid=True, figsize=(15,10))
