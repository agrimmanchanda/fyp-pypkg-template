"""
Iterative Imputer Experiment I
===========================================

Single biomarker removal experiment.

"""
#######################################
# Import the relevant libraries first
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
from pkgname.utils.widgets import TidyWidget

#######################################
# -------------------------------------
# Data handling
# -------------------------------------
# First, let's define the data set path and relevant variables of interest

path_data = '../load_dataset/datasets/pathology-sample-march-may.csv'

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