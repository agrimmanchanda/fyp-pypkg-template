"""
Experiment 1
========================================

Generating patient static profiles for patients using ``TidyWidget`` class.

"""
# Import
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# DataBlend library
from labimputer.utils.widgets import TidyWidget

# ------------------------
# Data handling 
# ------------------------

# Define FBC panel and interest biochemical markers
FBC_CODES = sorted(["EOS", "MONO", "BASO", "NEUT", "RBC", "WBC", 
                "MCHC", "MCV", "LY", "HCT", "RDW", "HGB", 
                "MCH", "PLT", "MPV", "NRBCA"])

# Define variables of interest for study
INTEREST_cols = ["_uid","dateResult", "orderCode", "result", "unit", "unitRange"]

# Define index and value parameters
index = ['_uid', 'dateResult', 'orderCode']
value = 'result'

# Set relative data path and set FBC panel list
path_data = '../resources/datasets/nhs/pathology-sample-march-may.csv'

# Read data and drop Nan _uid records
df = pd.read_csv(path_data, usecols=INTEREST_cols)

# Reset index to remove the ones from the current data set
df.reset_index(drop=True, inplace=True)

# Drop any null values from the data set
data = df.dropna()

# --------------------
# Transform
# --------------------

# Create widget
widget = TidyWidget(index=index, value=value)

# Transform (keep all)
transform, duplicated = \
    widget.transform(data, report_duplicated=True)

# Transform (keep first)
transform_first = \
    widget.transform(data, keep='first')

# Select features of interest from transformed data
data_feat = transform[["_uid", "dateResult"] + FBC_CODES]

# Save to csv file for use with subsequent experiments
# data_feat.dropna().to_csv('../resources/datasets/nhs/pathology-sample-march-may-transformed.csv')