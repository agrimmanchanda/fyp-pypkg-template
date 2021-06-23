"""
Summary report of NHS Pathology Data
===========================================

Using ``dataprep`` to create interactive report.

"""

# Import the relevant libraries first
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from dataprep.eda import create_report
import warnings
warnings.filterwarnings("ignore")
from labimputer.utils.load_dataset import *

# Set relative data path and set FBC panel list
path_data = '../resources/datasets/nhs/Transformed_First_FBC_dataset.csv'

FBC_CODES = ["EOS", "MONO", "BASO", "NEUT", "RBC", "WBC", 
                "MCHC", "MCV", "LY", "HCT", "RDW", "HGB", 
                "MCH", "PLT", "MPV", "NRBCA"]

# Read data and drop Nan _uid records
df = pd.read_csv(path_data).dropna(subset=['pid'])

df.reset_index(drop=True, inplace=True)

# Obtain the biomarkers DataFrame only
biomarkers_df = df[FBC_CODES].dropna(subset=FBC_CODES)

# Suppress output:

with suppress_stdout_stderr():
    summary = create_report(df=biomarkers_df, title="NHS Pathology Summary")

summary