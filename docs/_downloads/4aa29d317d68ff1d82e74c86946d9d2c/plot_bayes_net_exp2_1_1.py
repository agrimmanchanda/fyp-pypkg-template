"""
Experiment 2: Model Evaluation
===========================================

The aim of this experiment was to remove multiple features from the data set
satisfying the Missing At Random (MAR) assumption and using the remainining 
features to predict its values to emulate an actual imputer with Bayesian
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
from labimputer.core.bayes_net import BNRegressor, BNImputer, EMImputer
from labimputer.core.iter_imp import IterativeImputerRegressor, SimpleImputerRegressor
from labimputer.utils.bayes_net import get_data_statistics, get_simple_data_stats

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
    # test_copy.loc[test_set.sample(frac=0.1).index, col] = np.nan

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

    # Run the EM imputer on training data
    em = EMImputer(max_iter=10, epsilon=0.01)

    # Fit the EM imputer
    em.fit(train_copy)

    # Get transformed data
    Xt_em = em.transform(train_copy)

    # Discretise both training and test set 
    dis = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

    # Fit on training set to prevent data leakage
    dis.fit(Xt_em)

    # Transform training set
    train_new = pd.DataFrame(dis.transform(Xt_em), columns=FBC_PANEL)

    # Test training set
    test_new = pd.DataFrame(dis.transform(test_set), columns=FBC_PANEL)

    # Initialise imputer
    bn_reg = BNImputer(FBC_PANEL)

    # Collect relevant scores
    test_scores = pd.DataFrame()

    # Loop over each biomarker
    for idx, biomarker in enumerate(FBC_PANEL):

        # Assign training
        auxtrain = train_new.copy()

        # Assign test
        auxtest = test_new.copy()

        # Split training data
        X_train = auxtrain[[x for x in auxtrain.columns if x != biomarker]]
        y_train = auxtrain[biomarker]

        # Set missing values
        for col in FBC_PANEL:
            auxtest.loc[auxtest.sample(frac=0.1).index, col] = np.nan

        # Find index of only missing values
        col_arr = auxtest.to_numpy()[:, idx]

        # Create new missing values
        nan_idx = np.argwhere(np.isnan(col_arr)).flatten()

        # Fit the training data
        bn_reg.fit(X_train, y_train)

        # Only select rows with missing values with that feature
        xtest = auxtest[auxtest[biomarker].isna()]

        # Transform and return predictions
        ypred = pd.DataFrame(bn_reg.transform(auxtest), columns=FBC_PANEL)

        # Get them back in original form
        ypred_hat = dis.inverse_transform(ypred)[:, idx]

        # Flatten to get test array
        ytest = test_set[biomarker].to_numpy().flatten()

        # Create array suitable for data storage
        true_pred_vals = pd.DataFrame(list(zip(ytest, ypred_hat)),
                columns=[f'{biomarker}-true', f'{biomarker}-pred'])
        
        # Concat the two test score types
        test_scores = pd.concat([test_scores, true_pred_vals], axis=1)

        # Save
        test_scores.to_csv('datasets/bn_mult_test_results_10.csv')



#######################################
# -------------------------------------
# Find and plot data from HOTS
# -------------------------------------

# Read data files
df1 = pd.read_csv('datasets/bn_mult_test_results_10.csv', index_col=0)
df2 = pd.read_csv('datasets/bn_mult_test_results_30.csv', index_col=0)
df3 = pd.read_csv('datasets/bn_mult_test_results_50.csv', index_col=0)
df4 = pd.read_csv('datasets/bn_simple_test_results.csv', index_col=0)

# Extract RMSE scores
hots_10 = get_simple_data_stats(df1, FBC_PANEL, 2)
hots_30 = get_simple_data_stats(df2, FBC_PANEL, 2)
hots_50 = get_simple_data_stats(df3, FBC_PANEL, 2)
median_stats = get_data_statistics(df4, FBC_PANEL, 3)

median_stats.columns = ['Best', 'Median', 'MW']

# Concatenate relevant dataframe

conc = pd.concat([hots_10, hots_30, hots_50, median_stats['Median']], axis=1)

conc.columns = ['10%', '30%', '50%', 'Median']

#######################################
# -------------------------------------
# RMSE for 10% missing on HOTS
# -------------------------------------

h10 = conc[['10%', 'Median']]

h10['Delta (%)'] = 100 - (100* (h10['10%']/h10['Median']))

h10['Model'] = ['BN (10%)' for i in range(h10.shape[0])]

h10.loc['Mean'] = h10.mean()

h10

#######################################
# -------------------------------------
# RMSE for 30% missing on HOTS
# -------------------------------------

h30 = conc[['30%', 'Median']]

h30['Delta (%)'] = 100 - (100* (h30['30%']/h30['Median']))

h30['Model'] = ['BN (30%)' for i in range(h30.shape[0])]

h30.loc['Mean'] = h30.mean()

h30

#######################################
# -------------------------------------
# RMSE for 50% missing on HOTS
# -------------------------------------

h50 = conc[['50%', 'Median']]

h50['Delta (%)'] = 100 - (100* (h50['50%']/h50['Median']))

h50['Model'] = ['BN (50%)' for i in range(h50.shape[0])]

h50.loc['Mean'] = h50.mean()

h50

#######################################
# -------------------------------------
# Delta for all missing values
# -------------------------------------

# Collect delta and respective model
pt1 = h10[['Delta (%)', 'Model']]

pt2 = h30[['Delta (%)', 'Model']]

pt3 = h50[['Delta (%)', 'Model']]

comb_df = pd.concat([pt1, pt2, pt3], axis=0)

# Plot figure
plt.figure(figsize=(16,6))

# Plot bar plot
plot_comb = sns.barplot(x=comb_df.index, y=comb_df['Delta (%)'], hue=comb_df['Model']);

# Set xlabel
plot_comb.set_xlabel("Analyte")

# Show
plt.show()

###########################################
# -----------------------------------------
# NAE distribution for all missing values
# -----------------------------------------

nae_results = pd.DataFrame()

nae_10 = np.split(df1.T.to_numpy(), len(df1.T.to_numpy())/2)
nae_30 = np.split(df2.T.to_numpy(), len(df2.T.to_numpy())/2)
nae_50 = np.split(df3.T.to_numpy(), len(df3.T.to_numpy())/2)
nae_med = np.split(df4.T.to_numpy(), len(df4.T.to_numpy())/3)

# Find for 10% panel first 
for idx, values in enumerate(zip(nae_10, FBC_PANEL)):
    
    y_true, y_pred = values[0][0], values[0][1]
    
    nae_tp = nae(y_true, y_pred)
    
    nae_vals = pd.DataFrame([nae_tp, 
    ['BN (10%)' for _ in range(len(nae_tp))], 
    [values[1] for _ in range(len(nae_tp))]]).T

    nae_results = nae_results.append(nae_vals)

# Find for 30% panel next
for idx, values in enumerate(zip(nae_50, FBC_PANEL)):
    
    y_true, y_pred = values[0][0], values[0][1]
    
    nae_tp = nae(y_true, y_pred)
    
    nae_vals = pd.DataFrame([nae_tp, 
    ['BN (30%)' for _ in range(len(nae_tp))], 
    [values[1] for _ in range(len(nae_tp))]]).T

    nae_results = nae_results.append(nae_vals)

# Find for 50% panel last
for idx, values in enumerate(zip(nae_50, FBC_PANEL)):

    y_true, y_pred = values[0][0], values[0][1]

    nae_tp = nae(y_true, y_pred)

    nae_vals = pd.DataFrame([nae_tp, 
    ['BN (50%)' for _ in range(len(nae_tp))], 
    [values[1] for _ in range(len(nae_tp))]]).T

    nae_results = nae_results.append(nae_vals)

# Find for median values 
for idx, values in enumerate(zip(nae_med, FBC_PANEL)):

    y_true, y_pred, y_med = values[0][0], values[0][1], values[0][2]

    nae_tm = nae(y_true, y_med)

    nae_meds = pd.DataFrame([nae_tm, 
    ['Median' for _ in range(len(nae_tp))], 
    [values[1] for _ in range(len(nae_tp))]]).T

    nae_results = nae_results.append(nae_meds)


nae_results.columns = ['NAE', 'Model', 'Analyte']

# Plot
plt.figure(figsize=(18,6))

# create grouped boxplot 
sns.boxplot(x = nae_results['Analyte'],
        y = nae_results['NAE'],
        hue = nae_results['Model'],
        hue_order=['BN (10%)', 'BN (30%)', 'BN (50%)', 'Median'],
        showfliers=False,
        )

# Show
plt.show()