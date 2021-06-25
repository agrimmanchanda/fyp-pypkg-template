"""
Experiment 2: Model Evaluation
===========================================

The aim of this experiment was to remove multiple features from the data set
satisfying the Missing At Random (MAR) assumption and using the remainining 
features to predict its values to emulate an actual imputer.

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
from scipy import stats

# Libraries sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
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
from labimputer.utils.iter_imp import corr_pairs, get_score_statistics, rmse, norm_rmse, rmsle, get_test_scores, nae, get_best_models, get_cvts_stats
from labimputer.core.iter_imp import IterativeImputerRegressor, SimpleImputerRegressor

#######################################
# -------------------------------------
# Define tuned estimators
# -------------------------------------
_TUNED_ESTIMATORS = {
    'lr': LinearRegression(n_jobs=-1),
    'dt': DecisionTreeRegressor(
        criterion='mse',
        splitter='best',
        max_depth=6,
        max_leaf_nodes=12,
        min_samples_leaf=8,
        min_samples_split=8,
    ),
    'rf': ExtraTreesRegressor(
        n_estimators=10,
        criterion='mse',
        max_depth=6,
        bootstrap=False,
        warm_start=False,
        n_jobs=-1,
    ),
    'svr': SGDRegressor(
        alpha=1e-4,
        epsilon=0.01,
        learning_rate='adaptive',
        loss='squared_epsilon_insensitive',
        early_stopping=True,
        warm_start=True,
    ),
    'knn': KNeighborsRegressor(
        n_neighbors=7,
        weights='distance',
        n_jobs=-1,
    ),
    'xgb': XGBRegressor(
        n_estimators=10,
        eval_metric='rmse',
        max_depth=6,
        eta=0.2,
        gamma=0.1,
    ),
    'mlp': MLPRegressor(
        alpha=1e-4,
        hidden_layer_sizes=(32,64),
        solver='adam',
        learning_rate='invscaling',
        warm_start=True,
        early_stopping=True,
    ),
    'median': SimpleImputerRegressor(
        strategy='median'
    ),
}

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

######################################
# ------------------------------------
# Obtain best RMSE scores from (CVTS)
# ------------------------------------

# Define tested methods
METHODS = [
    'LR',
    'DT',
    'RF',
    'SVR',
    'KNN',
    'MLP',
    'XGB',
    'Median',
]

# Define FBC panel for the experiment
FBC_PANEL = sorted(["EOS", "MONO", "NEUT", "RBC", "WBC", 
                "MCHC", "MCV", "LY", "HCT", "RDW", "HGB", 
                "MCH", "PLT", "MPV"])


# Read CVTS results - 10%
cvts_10 = pd.read_csv('datasets/iir_mult_cv_results_10.csv', index_col=0)

# Mean and std stats
mean_stats, std_stats = get_cvts_stats(cvts_10, FBC_PANEL)

# Find the best models for 10
BEST_MODELS_10 = get_best_models(mean_stats)

# Read CVTS results - 30%
cvts_30 = pd.read_csv('datasets/iir_mult_cv_results_30.csv', index_col=0)

# Mean and std stats
mean_stats, std_stats = get_cvts_stats(cvts_30, FBC_PANEL)

# Find the best models for 30
BEST_MODELS_30 = get_best_models(mean_stats)

# Read CVTS results - 50%
cvts_50 = pd.read_csv('datasets/iir_mult_cv_results_50.csv', index_col=0)

# Mean and std stats
mean_stats, std_stats = get_cvts_stats(cvts_50, FBC_PANEL)

# Find the best models for 50
BEST_MODELS_50 = get_best_models(mean_stats)


##############################################
# --------------------------------------------
# Model evaluation on held out test set (HOTS)
# --------------------------------------------

# Set to false to prevent running during script build
run_eval = False

if run_eval:

    # Collect relevant scores
    test_scores = pd.DataFrame()

    # Loop for each model in best models
    for biomarker, model in BEST_MODELS_10.items():

        for est in [model, 'median']:

            estimator = _TUNED_ESTIMATORS[est]

            # Select estimator
            if est != 'median':
                imputer = IterativeImputerRegressor(estimator=estimator,
                                                    min_value=0, 
                                                    max_iter=10,
                                                    verbose=2,
                                                    imputation_order='descending')
            else:
                imputer = estimator

            # Generate new train-test for each run
            aux_train = train_copy.copy()
            aux_test = test_copy.copy()

            # Define independent (X_train) and dependent (y_train) variables
            X_train = aux_train[[x for x in aux_train.columns if x != biomarker]]
            y_train = aux_train[biomarker]

            # Define same variables with test set
            X_test = aux_test[[x for x in aux_test.columns if x != biomarker]]
            y_test = aux_test[biomarker]

            nan_idx = np.argwhere(np.isnan(y_test.to_numpy())).flatten()

            # Information
            print("\n Evaluating... %s for biomarker... %s" % (est, biomarker))

            # Create pipeline
            pipe = Pipeline(steps=[ ('std', StandardScaler()),
                                    (est, imputer)],
                            verbose=True)

            # Fit on training set 
            pipe.fit(X_train, y_train)

            # Generate x, y test 
            y_pred = pipe.predict(X_test)[nan_idx]

            # Store results in DataFrame
            if est != 'median':
                true_pred_vals = pd.DataFrame(list(zip(y_test, y_pred)),
                columns=[f'{biomarker}-{est}-true', f'{biomarker}-{est}-pred'])
            else:
                true_pred_vals = pd.Series(y_pred, name=f'{biomarker}-{est}')

            test_scores = pd.concat([test_scores, true_pred_vals], axis=1)

            test_scores.to_csv('datasets/iir_mult_test_results_10.csv')

##########################################
# ----------------------------------------
# RMSE for held out test set (HOTS) - 10%
# ----------------------------------------

# Generate simple test results
df = pd.read_csv('datasets/iir_mult_test_results_10.csv', index_col=0)

# Split the model and median scores for each analyte
split_data = np.split(df.T.to_numpy(), len(df.T.to_numpy())/3)

# DataFrame for RMSE
data = pd.DataFrame()

# DataFrame for NAE
nae_results = pd.DataFrame()

# Iterate through the predicted and median scores
for idx, values in enumerate(zip(split_data, FBC_PANEL)):
    
    # Extract the true and predicted values
    y_true, y_pred, y_med = values[0][0], values[0][1], values[0][2]
    
    # Obtain the RMSE scores
    rmse_tp, rmse_tm = rmse(y_true, y_pred), rmse(y_true, y_med)

    # Obtain NAE scores
    nae_tp, nae_tm = nae(y_true, y_pred), nae(y_true, y_med)
    
    nae_vals = pd.DataFrame([nae_tp, 
    ['Best (10%)' for _ in range(len(nae_tp))], 
    [values[1] for _ in range(len(nae_tp))]]).T
    
    nae_vals_med = pd.DataFrame([nae_tm, 
    ['Median' for _ in range(len(nae_tm))], 
    [values[1] for _ in range(len(nae_tm))]]).T
    
    # Join the RMSE results
    join_rmse = pd.concat([pd.Series(rmse_tp), pd.Series(rmse_tm)], axis=1)

    # Join the NAE results
    join_nae = pd.concat([nae_vals, nae_vals_med], axis=0)
    
    # Append
    data = data.append(join_rmse)
    nae_results = nae_results.append(join_nae)


# Create column names and set index
data.columns, data.index = ['Best', 'Median'], FBC_PANEL

# Define delta column
data['Delta (%)'] = 100 - (100* (data['Best']/data['Median']))

# Set model type
data['Model'] = ['Best (10%)' for i in range(data.shape[0])] 

# Get mean scores for each model
data.loc['Mean'] = data.mean()

data

##########################################
# ----------------------------------------
# RMSE for HOTS - 30%
# ----------------------------------------

# Generate simple test results
df1 = pd.read_csv('datasets/iir_mult_test_results_30.csv', index_col=0)

# Split the model and median scores for each analyte
split_data1 = np.split(df1.T.to_numpy(), len(df1.T.to_numpy())/3)

# DataFrame for data
data1 = pd.DataFrame()

# Iterate through the predicted and median scores
for idx, values in enumerate(zip(split_data1, FBC_PANEL)):
    
    # Extract the true and predicted values
    y_true, y_pred, y_med = values[0][0], values[0][1], values[0][2]
    
    # Obtain the RMSE scores
    rmse_tp, rmse_tm = rmse(y_true, y_pred), rmse(y_true, y_med)

    # Obtain NAE scores
    nae_tp, nae_tm = nae(y_true, y_pred), nae(y_true, y_med)
    
    nae_vals = pd.DataFrame([nae_tp, 
    ['Best (30%)' for _ in range(len(nae_tp))], 
    [values[1] for _ in range(len(nae_tp))]]).T
    
    nae_vals_med = pd.DataFrame([nae_tm, 
    ['Median' for _ in range(len(nae_tm))], 
    [values[1] for _ in range(len(nae_tm))]]).T
    
    # Join the RMSE results
    join_rmse = pd.concat([pd.Series(rmse_tp), pd.Series(rmse_tm)], axis=1)

    # Join the NAE results
    join_nae = pd.concat([nae_vals, nae_vals_med], axis=0)
    
    # Append
    data1 = data1.append(join_rmse)
    nae_results = nae_results.append(join_nae)

# Create column names and set index
data1.columns, data1.index = ['Best', 'Median'], FBC_PANEL

# Define delta column
data1['Delta (%)'] = 100 - (100* (data1['Best']/data1['Median']))

# Set model type
data1['Model'] = ['Best (30%)' for i in range(data1.shape[0])] 

# Get mean scores for each model
data1.loc['Mean'] = data1.mean()

data1

##########################################
# ----------------------------------------
# RMSE for for HOTS - 50%
# ----------------------------------------

# Generate simple test results
df2 = pd.read_csv('datasets/iir_mult_test_results_50.csv', index_col=0)

# Split the model and median scores for each analyte
split_data2 = np.split(df2.T.to_numpy(), len(df2.T.to_numpy())/3)

# DataFrame for data
data2 = pd.DataFrame()

# Iterate through the predicted and median scores
for idx, values in enumerate(zip(split_data2, FBC_PANEL)):
    
    # Extract the true and predicted values
    y_true, y_pred, y_med = values[0][0], values[0][1], values[0][2]
    
    # Obtain the RMSE scores
    rmse_tp, rmse_tm = rmse(y_true, y_pred), rmse(y_true, y_med)

    # Obtain NAE scores
    nae_tp, nae_tm = nae(y_true, y_pred), nae(y_true, y_med)
    
    nae_vals = pd.DataFrame([nae_tp, 
    ['Best (50%)' for _ in range(len(nae_tp))], 
    [values[1] for _ in range(len(nae_tp))]]).T
    
    nae_vals_med = pd.DataFrame([nae_tm, 
    ['Median' for _ in range(len(nae_tm))], 
    [values[1] for _ in range(len(nae_tm))]]).T
    
    # Join the RMSE results
    join_rmse = pd.concat([pd.Series(rmse_tp), pd.Series(rmse_tm)], axis=1)

    # Join the NAE results
    join_nae = pd.concat([nae_vals, nae_vals_med], axis=0)
    
    # Append
    data2 = data2.append(join_rmse)
    nae_results = nae_results.append(join_nae)

# Create column names and set index
data2.columns, data2.index = ['Best', 'Median'], FBC_PANEL

# Define delta column
data2['Delta (%)'] = 100 - (100 * (data2['Best']/data2['Median']))

# Set model type
data2['Model'] = ['Best (50%)' for i in range(data2.shape[0])] 

# Get mean scores for each model
data2.loc['Mean'] = data2.mean()

data2

##################################################
# ------------------------------------------------
# Comparison of Delta to Simple Median Imputation
# ------------------------------------------------

# Select all Delta and Model part of data
pt1 = data[['Delta (%)', 'Model']][:-1]
pt2 = data1[['Delta (%)', 'Model']][:-1]
pt3 = data2[['Delta (%)', 'Model']][:-1]

# Combined all Delta scores together
comb_df = pd.concat([pt1, pt2, pt3], axis=0)

# Figure
plt.figure(figsize=(16,6))

# Plot combined Delta scores
plot_comb = sns.barplot(x=comb_df.index, y=comb_df['Delta (%)'], hue=comb_df['Model'])

# Set the x label
plot_comb.set_xlabel("Analyte")

# Show
plt.show()


###################################
# ---------------------------------
# NAE distribution for HOTS
# ---------------------------------

nae_results.columns = ['NAE', 'Model', 'Analyte']

# Plot the figure
plt.figure(figsize=(20,8))

# Create grouped boxplot 
sns.boxplot(x = nae_results['Analyte'],
        y = nae_results['NAE'],
        hue = nae_results['Model'],
        showfliers=False
        )

# Show
plt.show()