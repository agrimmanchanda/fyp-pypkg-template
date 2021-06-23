"""
Experiment 1: Model Evaluation
===========================================

The aim of this experiment was to remove a single feature from the data set 
and use the remaining features to predict its values to emulate a simple 
regression model. This script has results from model evaluation.

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
from labimputer.utils.iter_imp import corr_pairs, get_score_statistics, rmse, norm_rmse, rmsle, get_test_scores, nae, get_best_models
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
        max_depth=8,
        max_leaf_nodes=15,
        min_samples_leaf=8,
        min_samples_split=8,
    ),
    'rf': ExtraTreesRegressor(
        n_estimators=100,
        criterion='mse',
        max_depth=8,
        bootstrap=False,
        warm_start=False,
        n_jobs=-1,
    ),
    'svr': SGDRegressor(
        alpha=1e-4,
        epsilon=0.05,
        learning_rate='adaptive',
        loss='squared_loss',
        early_stopping=True,
        warm_start=True,
    ),
    'knn': KNeighborsRegressor(
        n_neighbors=8,
        weights='distance',
        n_jobs=-1,
    ),
    'xgb': XGBRegressor(
        n_estimators=100,
        eval_metric='rmse',
        max_depth=10,
        eta=0.2,
        gamma=0.1,
    ),
    'mlp': MLPRegressor(
        alpha=1e-4,
        hidden_layer_sizes=32,
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

#######################################
# -------------------------------------
# Five fold cross validation (CVTS)
# -------------------------------------

# Number of splits
n_splits = 5

# Create Kfold instance
skf = KFold(n_splits=n_splits, shuffle=False)

###############################################################
# -------------------------------------------------------------
# Obtain best RMSE scores from cross validation test set (CVTS)
# -------------------------------------------------------------

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

# Read CVTS results
cvts = pd.read_csv('datasets/iir_simple_cv_results.csv', index_col=0)

# Get mean and variance of RMSE scores
all_scores = get_score_statistics(cvts, metric='rmse')

# Split scores to obtain score for each estimator
split_scores = np.array_split(all_scores, 8)

# Stack scores horizontally for easier plotting
hsplit_scores = np.hstack((split_scores))

# Create DataFrame for mean and std dev statistics
statistics = pd.DataFrame(hsplit_scores, index=FBC_PANEL)

# Split mean and std dev statistics
mean_stats, std_stats = statistics.iloc[:,::2], statistics.iloc[:,1::2]

# Rename columns to match algorithms
mean_stats.columns, std_stats.columns = METHODS, METHODS

# Select best models based on best CVTS score for each analyte
BEST_MODELS = get_best_models(mean_stats)

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
    for biomarker, model in BEST_MODELS.items():

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
            pipe = Pipeline(steps=[ ('std', StandardScaler()),
                                    (est, imputer)],
                            verbose=True)

            # Fit on training set 
            pipe.fit(X_train, y_train)

            # Generate x, y test 
            y_pred = pipe.predict(X_test)

            # Store results in DataFrame
            if est != 'median':
                true_pred_vals = pd.DataFrame(list(zip(y_test, y_pred)),
                columns=[f'{biomarker}-{est}-true', f'{biomarker}-{est}-pred'])
            else:
                true_pred_vals = pd.Series(y_pred, name=f'{biomarker}-{est}')

            test_scores = pd.concat([test_scores, true_pred_vals], axis=1)

            test_scores.to_csv('datasets/iir_simple_test_results.csv')



#######################################
# -------------------------------------
# RMSE for held out test set (HOTS)
# -------------------------------------

# Generate simple test results
df = pd.read_csv('datasets/iir_simple_test_results.csv', index_col=0)

# Split the model and median scores for each analyte
split_data = np.split(df.T.to_numpy(), len(df.T.to_numpy())/3)

# DataFrame for data
data = pd.DataFrame()

# Iterate through the predicted and median scores
for idx, values in enumerate(zip(split_data, FBC_PANEL)):
    
    # Extract the true and predicted values
    y_true, y_pred, y_med = values[0][0], values[0][1], values[0][2]
    
    # Obtain the RMSE scores
    rmse_tp, rmse_tm = rmse(y_true, y_pred), rmse(y_true, y_med)
    
    # Join the results
    join = pd.concat([pd.Series(rmse_tp), pd.Series(rmse_tm)], axis=1)
    
    # Append
    data = data.append(join)

# Create column names and set index
data.columns, data.index = ['Best', 'Median'], FBC_PANEL

# Define delta column
data['Delta (%)'] = 100 - (100* (data['Best']/data['Median']))

# Get the best models from CVTS
data['Model'] = BEST_MODELS.values()

# Get mean scores for each model
data.loc['Mean'] = data.mean()

data

##############################
# ----------------------------
# Mann Whitney U-test for HOTS
# ----------------------------

# Create empty DataFrame
mwu_df = pd.DataFrame()

# Loop through results
for idx, values in enumerate(zip(split_data, FBC_PANEL)):
    
    # Extract true and median scores
    y_true, y_pred, y_med = values[0][0], values[0][1], values[0][2]
    
    # Carry out MWU test
    mwu_tp, mwu_tm = stats.mannwhitneyu(y_true, y_pred)[1], stats.mannwhitneyu(y_true, y_med)[1]

    # Join relevant cores
    join = pd.concat([pd.Series(mwu_tp), pd.Series(mwu_tm)], axis=1)
    
    # Append relevant scores
    mwu_df = mwu_df.append(join)

# Display the p-values for best and median
mwu_df.columns, mwu_df.index = ['Best: p-value', 'Median: p-value'], FBC_PANEL

mwu_df

############################
# --------------------------
# NAE distribution for HOTS
# --------------------------

# Create new DataFrame
nae_results = pd.DataFrame()

# Loop through data
for idx, values in enumerate(zip(split_data, FBC_PANEL)):
    
    # Get the predicted and median scores
    y_true, y_pred, y_med = values[0][0], values[0][1], values[0][2]
    
    # Get the NAE scores for predicted and median
    nae_tp, nae_tm = nae(y_true, y_pred), nae(y_true, y_med)
    
    # Get the best scores
    nae_vals = pd.DataFrame([nae_tp, 
    ['Best' for _ in range(len(nae_tp))], 
    [values[1] for _ in range(len(nae_tp))]]).T
    
    # Get the median scores
    nae_vals_med = pd.DataFrame([nae_tm, 
    ['Median' for _ in range(len(nae_tm))], 
    [values[1] for _ in range(len(nae_tm))]]).T
    
    # Join scores
    join = pd.concat([nae_vals, nae_vals_med], axis=0)
    
    # Append
    nae_results = nae_results.append(join)

# Set columns
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