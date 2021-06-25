"""
Experiment 2: Model Learning
===========================================

The aim of this experiment was to remove multiple features from the data set
satisfying the Missing At Random (MAR) assumption and using the remainining 
features to predict its values to emulate an actual imputer using Bayesian
Networks.

The data was removed in proportions: 10%, 30% and 50%.

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
from labimputer.core.iter_imp import IterativeImputerRegressor, SimpleImputerRegressor
from labimputer.core.bayes_net import BNRegressor, EMImputer

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

# Use copy of the original train and test set
train_copy, test_copy = train_set.copy(), test_set.copy()

# Remove 10, 30 or 50% of values depending upon requirements
for col in train_copy.columns:
    train_copy.loc[train_set.sample(frac=0.1).index, col] = np.nan
    test_copy.loc[test_set.sample(frac=0.1).index, col] = np.nan


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

# Loop over each estimator
for i, est in enumerate(ESTIMATORS):

    # Dictionary for storing all test scores on hold
    test_scores = {}

    # Check if estimator has been defined else skip
    if est not in _TUNED_ESTIMATORS:
        continue
    
    # Select estimator
    estimator = _TUNED_ESTIMATORS[est]
    
    if est != 'median':
        imputer = IterativeImputerRegressor(estimator=estimator,
                                            min_value=0, 
                                            max_iter=10000)
    else:
        imputer = estimator

    # Loop over each analyte
    for biomarker in train_set:

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
        print("\n%s. Evaluating... %s for biomarker... %s" % (i, est, biomarker))

        # Create pipeline
        pipe = Pipeline(steps=[ ('pre', EMImputer(max_iter=10, epsilon=10)),
                                ('dis', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')),
                                (est, imputer)],
                        verbose=True)

        # Obtain scores for each fold using cross_validate
        scores = cross_validate(pipe, 
                                X_train, 
                                y_train, 
                                scoring=scoring, 
                                cv=skf, 
                                return_train_score=True, 
                                n_jobs=-1, 
                                verbose=0)

        # Fit on training set 
        pipe.fit(X_train, y_train)

        # Generate x, y test 
        y_pred = pipe.predict(X_test)

        # Compendium of all test scores
        test_scores[biomarker] = get_test_scores(y_test, y_pred)

        # Extract results
        results = pd.DataFrame(scores)
        results.index = ['%s_%s_%s' % (biomarker, est, j)
            for j in range(results.shape[0])]
        
        # Add to compendium of results
        bn_results = bn_results.append(results)

#######################################
# -------------------------------------
# Save results
# -------------------------------------

# Save
# bn_results.to_csv('datasets/bn_mult_cv_results.csv')

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

# Remove 10% of values
for col in Xt.columns:
    Xt.loc[Xt.sample(frac=0.1).index, col] = np.nan

# Initialise the regressor with pre-defined edges (based on previous testing)
m1 = BNRegressor(FBC_PANEL)

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