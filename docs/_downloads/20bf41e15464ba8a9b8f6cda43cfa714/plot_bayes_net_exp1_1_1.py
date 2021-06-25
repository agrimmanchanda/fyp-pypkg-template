"""
Experiment 1: Model Evaluation
===========================================

The aim of this experiment was to remove a single feature from the data set 
and use the remaining features to predict its values to emulate a simple 
regression model with Bayesian Networks. This script has results from model evaluation.

"""

#######################################
# -------------------------------------
# Libraries import
# -------------------------------------

# Libraries generic
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import networkx as nx
import joblib

# Libraries sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split

# Regressors
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# Metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error

# Custom Packages
from labimputer.utils.load_dataset import remove_data_outliers
from labimputer.utils.iter_imp import corr_pairs, get_score_statistics, rmse, norm_rmse, rmsle, get_test_scores, nae, get_best_models, get_cvts_delta
from labimputer.core.bayes_net import BNRegressor
from labimputer.core.iter_imp import IterativeImputerRegressor, SimpleImputerRegressor
from labimputer.utils.bayes_net import get_data_statistics

#######################################
# -------------------------------------
# Data import 
# -------------------------------------

# Set relative data path and set FBC panel list
path_data = '../resources/datasets/nhs/Transformed_First_FBC_dataset.csv'

# Define FBC panel for the experiment
FBC_CODES = sorted(["EOS", "MONO", "BASO", "NEUT", "RBC", "WBC", 
                "MCHC", "MCV", "LY", "HCT", "RDW", "HGB", 
                "MCH", "PLT", "MPV", "NRBCA"])

RBC_ANALYTES = ['HCT', 'HGB', 'RBC', 'MCH', 'MCV', 'MCHC', 'RDW']
WBC_ANALYTES = ['EOS', 'MONO', 'LY', 'NEUT', 'WBC']
PLT_ANALYTES = ['PLT', 'MPV']

# Read data and drop Nan _uid records
df = pd.read_csv(path_data).dropna(subset=['pid'])

# Reset the index to easily count all test records
df.reset_index(drop=True, inplace=True)

# Obtain the biomarkers DataFrame only
raw_data = df[FBC_CODES].dropna(subset=FBC_CODES)

# Remove outliers from dataset
complete_profiles, _ = remove_data_outliers(raw_data)

# Constant variables to drop
DROP_FEATURES = ['BASO', 'NRBCA']

# Complete profiles for complete case analysis
complete_profiles = complete_profiles.drop(DROP_FEATURES, axis=1)

FBC_PANEL = complete_profiles.columns

#######################################
# -------------------------------------
# Define tuned estimators
# -------------------------------------
_TUNED_ESTIMATORS = {
    'median': SimpleImputerRegressor(
        strategy='median'
    ),
    'BN': BNRegressor(FBC_PANEL)
}

#######################################
# -------------------------------------
# Correlation matrix
# -------------------------------------

# Calculate correlation matrix using Pearson Correlation Coefficient
corr_mat = complete_profiles.corr(method='pearson')

# Show
print("\nData:")
print(complete_profiles)
print("\nCorrelation (pearson):")
print(corr_mat)

#######################################
# -------------------------------------
# Split into train-test
# -------------------------------------

SEED = 8

# Train-test split of 80:20
train_set, test_set = train_test_split(complete_profiles, shuffle=False, test_size=0.2, random_state=8)

#######################################
# -------------------------------------
# Five fold cross validation (CVTS)
# -------------------------------------

# Number of splits
n_splits = 5

# Create Kfold instance
skf = KFold(n_splits=n_splits, shuffle=False)

# Scoring
scoring = {
    'nmae': 'neg_mean_absolute_error', # MAE
    'nmse': 'neg_mean_squared_error',       # MSE
    'nrmse': 'neg_root_mean_squared_error', # RMSE
    'rmsle': make_scorer(rmsle), # RMSLE
    'norm_rmse': make_scorer(norm_rmse), # NRMSE
}

# Compendium of results
bn_results = pd.DataFrame()

# Create a list of estimators
ESTIMATORS = [
    # 'median',
    # 'BN',
]

run_eval = False

if run_eval:

    # Collect relevant scores
    test_scores = pd.DataFrame()

    # Loop over each estimator

    for biomarker in complete_profiles:

        for est in ['bn', 'median']:

            estimator = _TUNED_ESTIMATORS[est]

            if est == 'bn':
                imputer = BNRegressor(FBC_PANEL)
            else:
                imputer = estimator

            # Generate new train-test for each run
            aux_train = train_set.copy()
            aux_test = test_set.copy()

            # Define independent (X_train) and dependent (y_train) variables
            X_train = aux_train[[x for x in aux_train.columns if x != biomarker]]
            y_train = aux_train[biomarker]

            # Define same variables with test set
            X_test = aux_test[[x for x in aux_test.columns if x != biomarker]]
            y_test = aux_test[biomarker]

            # Information
            print("\n Evaluating... %s for biomarker... %s" % (est, biomarker))

            # Create pipeline
            pipe = Pipeline(steps=[ ('dis', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')),
                                    (est, imputer)],
                            verbose=True)

            # Create array for training data
            y_train_arr = y_train.to_numpy().reshape(-1, 1)

            # Discretise the scaler as appropriate
            discaler = pipe.named_steps['dis'].fit(y_train_arr)

            # Save the discretiser
            joblib.dump(discaler, f'discaler.sav')

            # Transform the data
            train_y = pd.Series(pipe.named_steps['dis'].transform(y_train_arr).flatten(), name=biomarker)

            # Fit the data on newly transformed data
            pipe.fit(X_train,train_y)

            # Predict values on X_test data
            y_pred = pipe.predict(X_test)

            # Load the earlier saved discretiser
            inscaler = joblib.load(f'discaler.sav')

            # Inverse transform data to bring back to original form
            y_pred_hat = inscaler.inverse_transform(y_pred.reshape(-1,1)).reshape(-1)

            # Flatten the array to store in csv
            y_test = y_test.to_numpy().flatten()

            # Store results in DataFrame
            if est != 'median':
                true_pred_vals = pd.DataFrame(list(zip(y_test, y_pred)),
                columns=[f'{biomarker}-{est}-true', f'{biomarker}-{est}-pred'])
            else:
                true_pred_vals = pd.Series(y_pred, name=f'{biomarker}-{est}')

            test_scores = pd.concat([test_scores, true_pred_vals], axis=1)

            test_scores.to_csv('datasets/bn_simple_test_results.csv')

#######################################
# -------------------------------------
# Plot BN structure
# -------------------------------------

# Make a copy of the training set
aux_train = train_set.copy()

# Discretise data into five bins
dis = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

# Fit transform the discretised data
Xt = pd.DataFrame(dis.fit_transform(aux_train), columns=FBC_PANEL)

# Found from training
EDGES = [
    ('HGB', 'RDW'),
    ('HCT', 'HGB'),
    ('HCT', 'RBC'),
    ('MCV', 'RBC'),
    ('MCV', 'MCHC'),
    ('RDW', 'LY'),
    ('RDW', 'MCH'),
    ('PLT', 'WBC'),
    ('WBC', 'MONO'),
    ('WBC', 'NEUT'),
    ('MCH', 'MCV'),
    ('MCH', 'MCHC'),
    ('MPV', 'PLT'),
    ('LY', 'MONO'),
    ('LY', 'NEUT'),
    ('LY', 'WBC'),
    ('LY', 'PLT'),
    ('LY', 'EOS'),
    ('RBC', 'MPV'),
    ('RBC', 'LY'),
]

# Initialise the regressor with pre-defined edges (based on previous testing)
m1 = BNRegressor(FBC_PANEL, EDGES)

# Learn the data using the edges
m1.fit(Xt)

# Plot
plt.figure(figsize=(10,8))

# Define colour map for coding
color_map = []

for node in m1:

    if node in RBC_ANALYTES:
        color_map.append('salmon')
    elif node in WBC_ANALYTES:
        color_map.append('skyblue')
    else:
        color_map.append('plum')

# Draw the graph using networkx package
nx.draw(m1, with_labels=True, 
    node_size = 3000, 
    node_color=color_map, 
    edgecolors='black', 
    font_weight='bold', 
    width=1.5)

# Show
plt.show()

############################################
# -----------------------------------------
# RMSE scores for held out test set (HOTS)
# -----------------------------------------

# Read results
hots_results = pd.read_csv(f'datasets/bn_simple_test_results.csv', index_col=0)

# Get test results that is split
test_results = get_data_statistics(hots_results, FBC_PANEL, 3)

# Assign column names
test_results.columns = ['Bayesian Network', 'Median', 'MWU Test']

# Obtain Delta value
test_results['Delta (%)'] = 100 - (100* (test_results['Bayesian Network']/test_results['Median']))

# Get average for each model
test_results.loc['Mean'] = test_results.mean()

test_results

############################################
# -----------------------------------------
# RMSE Score Plot for HOTS
# -----------------------------------------

# Plot figure
plt.figure(figsize=(20,8))

# Plot bar plot
plt1 = sns.barplot(x=test_results[:-1].index, y=test_results['Delta (%)'][:-1], color='#1f77b4')

# Set xlabel as appropriate
plt1.set_xlabel("Analyte")

# Show
plt.show()

#########################################
# ---------------------------------------
# NAE distribution for HOTS
# ---------------------------------------

# Extract data from the test results
data = np.split(hots_results.T.to_numpy(), len(hots_results.T.to_numpy())/3)

# DataFrame to store NAE results
nae_results = pd.DataFrame()

# Loop for each analyte
for idx, values in enumerate(zip(data, FBC_PANEL)):
    
    # Extract all relevant values
    y_true, y_pred, y_med = values[0][0], values[0][1], values[0][2]
    
    # Obtain NAE scores
    nae_tp, nae_tm = nae(y_true, y_pred), nae(y_true, y_med)
    
    # Obtain all NAE vals
    nae_vals = pd.DataFrame([nae_tp, 
    ['BN' for _ in range(len(nae_tp))], 
    [values[1] for _ in range(len(nae_tp))]]).T
    
    nae_vals_med = pd.DataFrame([nae_tm, 
    ['Median' for _ in range(len(nae_tm))], 
    [values[1] for _ in range(len(nae_tm))]]).T
    
    # Join with DataFrame
    join = pd.concat([nae_vals, nae_vals_med], axis=0)
    
    nae_results = nae_results.append(join)

# Set column names
nae_results.columns = ['NAE', 'Model', 'Analyte']

# Plot figure
plt.figure(figsize=(15,5))

# Create grouped boxplot 
sns.boxplot(x = nae_results['Analyte'],
        y = nae_results['NAE'],
        hue = nae_results['Model'],
        showfliers=False)

# Show
plt.show()