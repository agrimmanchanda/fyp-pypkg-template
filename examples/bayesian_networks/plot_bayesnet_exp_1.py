"""
Bayesian Networks Experiment I.I
===========================================

Using the ``pgmpy`` library to learn the 
structure of Bayesian Network (BN) from the data,
estimate parameters for Conditional Probability 
Distributions (CPDs) and imputing missing values.

"""

#######################################
# -------------------------------------
# Libraries import
# -------------------------------------

# Libraries generic
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from math import isnan

# Libraries for BNs
import networkx as nx
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, BDeuScore, BicScore, BayesianEstimator
from pgmpy.inference import VariableElimination

# Libraries sklearn
from sklearn.model_selection import train_test_split

# Custom Packages
from labimputer.utils.load_dataset import remove_data_outliers
from labimputer.utils.bayes_net import get_unique_edges, get_row_from_df

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

# Obtain the biomarkers DataFrame only
raw_data = df[FBC_CODES].dropna(subset=FBC_CODES)

# Remove outliers from dataset
complete_profiles, _ = remove_data_outliers(raw_data)

# Constant variables to drop
DROP_FEATURES = ['BASO', 'NRBCA']

# Complete profiles for complete case analysis
complete_profiles = complete_profiles.drop(DROP_FEATURES, axis=1)

#######################################
# -------------------------------------
# Structure learning
# -------------------------------------

# Use the HillClimbSearch to find the best structure of the graph and use a 
# Bayesian Information Criterion (BIC) to set a score for the optimisation 
# problem being solved by HillClimbSearch. For convenience, code has been 
# commented as it takes a couple of minutes (with high CPU requirement) 
# to find the edges but the edges found are consistent for BIC and Bayesian
# Dirichlet Equivalent Uniform (BDeu).

# hc = HillClimbSearch(complete_df, scoring_method=BicScore(complete_df))
# best_model = hc.estimate()
# print(best_model.edges())

#######################################
# -------------------------------------
# Model topology 
# -------------------------------------

# edges found from structure learning
best_model = [
    ('EOS', 'LY'), 
    ('MONO', 'WBC'), 
    ('WBC', 'NEUT'), 
    #('MCHC', 'MCV'), 
    ('LY', 'MONO'), 
    ('LY', 'HCT'), 
    ('HCT', 'HGB'), 
    ('HCT', 'RBC'), 
    ('HCT', 'RDW'), 
    #('MCH', 'MCV')
]

# create an instance of Bayesian Model from pgmpy
model = BayesianModel(best_model)

# get only the variables found for BN
FEATURES = get_unique_edges(best_model)

complete_profiles = complete_profiles[FEATURES]

# get train and test split 

train, test = train_test_split(complete_profiles, shuffle=False)

#######################################
# -------------------------------------
# Visualise BN
# -------------------------------------

plt.figure(figsize=(20,10))

nx.draw(model, node_size=2000, node_color='orange', font_weight='bold', with_labels=True)

plt.show()

#######################################
# -------------------------------------
# Parameter learning
# -------------------------------------

# estimate the CPDs for each node in the BN
model.fit(train, 
    estimator=BayesianEstimator, 
    prior_type="BDeu")

# show the CPDs
for cpd in model.get_cpds():
    print(f"\n {cpd}")

#######################################
# -------------------------------------
# Generate test set data
# -------------------------------------

# test with only top 20 columns and set 10% data to NaN

test = test[:20]

tcolumns = test.columns

for col in tcolumns:
    test.loc[test.sample(frac=0.1).index, col] = np.nan

# show test dataframe with NaN values

test

#######################################
# -------------------------------------
# Inference using Variable Elimination
# -------------------------------------

# use the variable elimination algorithm
ve = VariableElimination(model)

# dataframe to return imputed test data

imp_data = pd.DataFrame()

# compute inference for each row
for row in test.itertuples():
    # get each row in dataframe as a dictionary
    rowData = get_row_from_df(row, tcolumns)

    # extract the variables with a NaN value
    nan_vars = [k for k, v in rowData.items() if isnan(v)]

    # extract the dict for observed variables i.e. ones with a value
    obs_vars = {k: v for k, v in rowData.items() if not isnan(v)}

    # save a copy of the observed variables dict
    aux = obs_vars.copy()

    # if no observed variables, then leave as is

    if len(nan_vars) > 0:

        # run BP algorithm and obtain the value for 
        # each of the missing variables

        val = ve.map_query(variables=nan_vars,
                            evidence=obs_vars, 
                            show_progress=False)
        
        # add the found values back to aux
        aux.update(val)

        # add the imputed row data to imp_data DataFrame
        imp_data = imp_data.append(aux, ignore_index=True)

        # loop again
        continue

    # otherwise just return the row as it is
    imp_data = imp_data.append(rowData, ignore_index=True)

# Show imputed test dataframe

imp_data