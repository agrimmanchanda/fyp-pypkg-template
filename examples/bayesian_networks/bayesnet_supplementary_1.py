"""
Bayesian Networks Experiment I.I
===========================================

Using the ``pgmpy`` library to learn the 
structure of Bayesian Network (BN) from the data,
estimate parameters for Conditional Probability 
Distributions (CPDs) and imputing missing values. 
This experiment uses discretisation as a 
pre-processing step to optimise inference performance.

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
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_squared_error

# Custom Packages
from labimputer.utils.load_dataset import remove_data_outliers
from labimputer.core.bayes_net import BNImputer, learn_model_structure
from labimputer.utils.bayes_net import get_unique_edges, get_score_statistics
from labimputer.utils.load_dataset import suppress_stdout

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

# create a copy of the complete data
aux = complete_profiles.copy(deep=True)

# discretise each feature into 5 uniform columns 
est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
Xt = pd.DataFrame(est.fit_transform(aux), columns=complete_profiles.columns)

# obtain train-test split to learn the structure of BN
train, _ = train_test_split(Xt, test_size=0.2, shuffle=False)

# learn the structure of BN using HillClimbSearch and BDeu score
best_model = learn_model_structure(train)

# print the best edges found
print(best_model.edges)

#######################################
# -------------------------------------
# Model topology 
# -------------------------------------

# create an instance of Bayesian Model from pgmpy
model = BNImputer(best_model.edges)

# get only the variables found for BN
FEATURES = get_unique_edges(best_model.edges)

# update the data using features found
bn_Xt = Xt[FEATURES]

# obtain train-test split to learn the structure of BN
bn_train, bn_test = train_test_split(bn_Xt, test_size=0.2, shuffle=False)

# #######################################
# # -------------------------------------
# # Visualise BN
# # -------------------------------------

plt.figure(figsize=(20,10))

nx.draw(model, node_size=2000, node_color='orange', font_weight='bold', with_labels=True)

plt.show()

#######################################
# -------------------------------------
# Parameter learning
# -------------------------------------

# estimate the CPDs for each node in the BN
model.fit(bn_train)

# show the CPDs
for cpd in model.get_cpds():
    print(f"\n {cpd}")

#######################################
# -------------------------------------
# Remove single biomarker
# -------------------------------------

bn_test = bn_test[:20]

rmse_scores = {}

for idx, col in enumerate(bn_test):

    aux = bn_test.copy()

    aux[col] = np.nan

    with suppress_stdout():
        pred = model.imputer(aux)

    ytrue = est.inverse_transform(bn_test)[:, idx]
    ypred = est.inverse_transform(pred)[:, idx]

    rmse = mean_squared_error(ytrue, ypred, squared=False)

    rmse_scores[col] = rmse

    print(f'RMSE for {col} is {rmse}')

# Save
# compendium = pd.DataFrame.from_dict(rmse_scores, orient='index')
# compendium.to_csv('datasets/bn_results.csv')

#######################################
# -------------------------------------
# Analyse scores and test results
# -------------------------------------

# read bn dataset
bn_scores = pd.read_csv('datasets/bn_results.csv', index_col=0)

# read iir dataset
iir_scores = pd.read_csv('datasets/iir_results.csv', index_col=0)

# get the mean rmse for 5 folds
mean_rmse = get_score_statistics(iir_scores, 'rmse')

# Split scores to obtain score for each estimator
split_scores = np.array_split(mean_rmse, 7)

# Stack scores horizontally for easier plotting
hsplit_scores = np.hstack((split_scores))

# Create DataFrame for mean and std dev statistics
statistics = pd.DataFrame(hsplit_scores, index=complete_profiles.columns)

# Get just mean RMSE
mean_stats = statistics.iloc[:,::2]

mean_stats = pd.concat([mean_stats, bn_scores], axis=1)

mean_stats.columns = ['lr', 'bridge', 'dt', 'sgd-ls', 'sgd-sv', 'knn', 'sir', 'BN']

# Highlighting the minimum values of last 2 columns
mean_stats.style.highlight_min(color = 'lightgreen', axis = 1)