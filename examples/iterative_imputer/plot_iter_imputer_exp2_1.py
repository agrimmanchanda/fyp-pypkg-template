"""
Experiment 2: Model Learning
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
from labimputer.utils.iter_imp import corr_pairs, get_score_statistics, rmse, norm_rmse, rmsle, get_test_scores, nae, get_best_models, get_cvts_stats, get_cvts_delta
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
        loss='squared_epsilon_insensitive',
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
iir_results = pd.DataFrame()

# Create a list of estimators
ESTIMATORS = [
    # 'lr',
    # 'dt',
    # 'rf',
    # 'svr',
    # 'knn',
    # 'mlp',
    # 'xgb',
    # 'median',
]

# Concat scores for each CVTS run
test_data = pd.DataFrame()

# Loop over each estimator
for i, est in enumerate(ESTIMATORS):

    # Dictionary for storing all test scores on hold
    test_scores = {}

    # Check if estimator has been defined else skip
    if est not in _TUNED_ESTIMATORS:
        continue
    
    # Select estimator
    estimator = _TUNED_ESTIMATORS[est]
    
    # Select imputer type
    if est != 'median':
        imputer = IterativeImputerRegressor(estimator=estimator,
                                            min_value=0, 
                                            max_iter=10,
                                            verbose=2,
                                            imputation_order='descending')
    else:
        imputer = estimator

    # Loop over each analyte
    for biomarker in train_set:

        # Generate new train-test for each run
        aux_train = train_copy.copy()
        aux_test = test_copy.copy()

        # Define independent (X_train) and dependent (y_train) variables
        X_train = aux_train[[x for x in aux_train.columns if x != biomarker]]
        y_train = aux_train[biomarker]

        # Define same variables with test set
        X_test = aux_test[[x for x in aux_test.columns if x != biomarker]]
        y_test = aux_test[biomarker]

        # Information
        print("\n%s. Evaluating... %s for biomarker... %s" % (i, est, biomarker))

        # Create pipeline
        pipe = Pipeline(steps=[ ('std', StandardScaler()),
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
        iir_results = iir_results.append(results)
    
    # Concatenate scores for the estimator to all other test scores
    test_data = pd.concat([test_data, pd.Series(test_scores, name=est)], axis=1)

#######################################
# -------------------------------------
# Save results
# -------------------------------------

# Save
# iir_results.to_csv('datasets/iir_mult_cv_results_10.csv')
# test_data.to_csv('datasets/iir_mult_test_results_10.csv')

#######################################
# -------------------------------------
# Analysis of results from CVTS - 10%
# -------------------------------------

# Read CVTS results
cvts_10 = pd.read_csv('datasets/iir_mult_cv_results_10.csv', index_col=0)

mean_stats, std_stats = get_cvts_stats(cvts_10, FBC_PANEL)

BEST_MODELS_10 = get_best_models(mean_stats)

print("Mean CVTS RMSE statistics (lowest score highlighted in green)")

# Highlighting the minimum values of last 2 columns
mean_stats.style.highlight_min(color = 'lightgreen', 
                       axis = 1)

#######################################
# -------------------------------------
# Plotting CVTS scores - 10%
# -------------------------------------

# Define figure size
plt.figure(figsize=(20,40))

# Create new mean_stats df without the final row
mean_stats_plot = mean_stats.head(mean_stats.shape[0] - 1)

# Loop for each plot
for idx, (biomarker, scores) in enumerate(mean_stats_plot.iterrows(), start=1):
    plt.subplot(7,2,idx)
    plt.title(f'RMSE for {biomarker}',
    fontweight='bold',
    fontsize=14)
    cmap = ['green' if (x == min(scores)) else 'royalblue' for x in scores]
    scores.plot.barh(grid=True,
                xerr=list(std_stats.loc[biomarker, :]),
                align='center',
                color=cmap)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('RMSE Score', fontsize=16)

# Space plots out
plt.tight_layout()

# Show
plt.show()

#########################################################
# -------------------------------------------------------
# Compare best CVTS with Simple Median Imputation - 10%
# -------------------------------------------------------

# Get Delta scores for 50% missing
cvts_best_10 = get_cvts_delta(mean_stats, BEST_MODELS_10)

# Plot figure
plt.figure(figsize=(15,5))

# Create barplot
plot = sns.barplot(x=cvts_best_10.index, y=cvts_best_10['$\Delta$ (%)'], hue=cvts_best_10['Model'], dodge=False)

# Set xlabel as appropriate
plot.set_xlabel("Analyte")

# Show
plt.show()

#######################################
# -------------------------------------
# Analysis of results from CVTS - 30%
# -------------------------------------

# Read CVTS results
cvts_30 = pd.read_csv('datasets/iir_mult_cv_results_30.csv', index_col=0)

mean_stats, std_stats = get_cvts_stats(cvts_30, FBC_PANEL)

BEST_MODELS_30 = get_best_models(mean_stats)

print("Mean CVTS RMSE statistics (lowest score highlighted in green)")

# Highlighting the minimum values of last 2 columns
mean_stats.style.highlight_min(color = 'lightgreen', 
                       axis = 1)

########################################
# -------------------------------------
# Plotting CVTS scores - 30%
# -------------------------------------

# Define figure size
plt.figure(figsize=(20,40))

# Create new mean_stats df without the final row
mean_stats_plot = mean_stats.head(mean_stats.shape[0] - 1)

# Loop for each plot
for idx, (biomarker, scores) in enumerate(mean_stats_plot.iterrows(), start=1):
    plt.subplot(7,2,idx)
    plt.title(f'RMSE for {biomarker}',
    fontweight='bold',
    fontsize=14)
    cmap = ['green' if (x == min(scores)) else 'royalblue' for x in scores]
    scores.plot.barh(grid=True,
                xerr=list(std_stats.loc[biomarker, :]),
                align='center',
                color=cmap)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('RMSE Score', fontsize=16)

# Space plots out
plt.tight_layout()

# Show
plt.show()

#########################################################
# -------------------------------------------------------
# Compare best CVTS with Simple Median Imputation - 30%
# -------------------------------------------------------

# Get Delta scores for 50% missing
cvts_best_30 = get_cvts_delta(mean_stats, BEST_MODELS_30)

# Plot figure
plt.figure(figsize=(15,5))

# Create barplot
plot = sns.barplot(x=cvts_best_30.index, y=cvts_best_30['$\Delta$ (%)'], hue=cvts_best_30['Model'], dodge=False)

# Set xlabel as appropriate
plot.set_xlabel("Analyte")

# Show
plt.show()


#######################################
# -------------------------------------
# Analysis of results from CVTS - 50%
# -------------------------------------

# Read CVTS results
cvts_50 = pd.read_csv('datasets/iir_mult_cv_results_50.csv', index_col=0)

mean_stats, std_stats = get_cvts_stats(cvts_50, FBC_PANEL)

BEST_MODELS_50 = get_best_models(mean_stats)

print("Mean CVTS RMSE statistics (lowest score highlighted in green)")

# Highlighting the minimum values of last 2 columns
mean_stats.style.highlight_min(color = 'lightgreen', 
                       axis = 1)


########################################
# -------------------------------------
# Plotting CVTS scores - 50%
# -------------------------------------

# Define figure size
plt.figure(figsize=(20,40))

# Create new mean_stats df without the final row
mean_stats_plot = mean_stats.head(mean_stats.shape[0] - 1)

# Loop for each plot
for idx, (biomarker, scores) in enumerate(mean_stats_plot.iterrows(), start=1):
    plt.subplot(7,2,idx)
    plt.title(f'RMSE for {biomarker}',
    fontweight='bold',
    fontsize=14)
    cmap = ['green' if (x == min(scores)) else 'royalblue' for x in scores]
    scores.plot.barh(grid=True,
                xerr=list(std_stats.loc[biomarker, :]),
                align='center',
                color=cmap)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('RMSE Score', fontsize=16)

# Space plots out
plt.tight_layout()

# Show
plt.show()

#########################################################
# ------------------------------------------------------
# Compare best CVTS with Simple Median Imputation - 50%
# ------------------------------------------------------

# Get Delta scores for 50% missing
cvts_best_50 = get_cvts_delta(mean_stats, BEST_MODELS_50)

# Plot figure
plt.figure(figsize=(15,5))

# Create barplot
plot = sns.barplot(x=cvts_best_50.index, y=cvts_best_50['$\Delta$ (%)'], hue=cvts_best_50['Model'], dodge=False)

# Set xlabel as appropriate
plot.set_xlabel("Analyte")

# Show
plt.show()