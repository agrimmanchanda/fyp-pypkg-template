"""
Iterative Imputer Experiment I.I
===========================================

Single biomarker removal using ``sklearn``
methods only.

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

# Libraries sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

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
from pkgname.utils.load_dataset import remove_data_outliers
from pkgname.utils.iter_imp import corr_pairs, get_score_statistics
from pkgname.core.iter_imp import IterativeImputerRegressor, SimpleImputerRegressor

#######################################
# -------------------------------------
# Define tuned estimators
# -------------------------------------
_TUNED_ESTIMATORS = {
    'lr': LinearRegression(),
    'bridge': BayesianRidge(
        alpha_1=1e-05,
        alpha_2=1e-07,
        lambda_1=1e-07,
        lambda_2=1e-05,
    ),
    'dt': DecisionTreeRegressor(
        criterion='mse',
        splitter='best',
        max_depth=8,
        max_leaf_nodes=15,
        min_samples_leaf=8,
        min_samples_split=8,
    ),
    'etr': ExtraTreesRegressor(
        n_estimators=100,
        criterion='mse',
        bootstrap=False,
        warm_start=False,
        n_jobs=-1,
    ),
    'sgd-ls': SGDRegressor(
        alpha=1e-4,
        epsilon=0.05,
        learning_rate='adaptive',
        loss='squared_loss',
        early_stopping=True,
        warm_start=True,
    ),
    'sgd-sv': SGDRegressor(
        alpha=1e-4,
        epsilon=0.01,
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
    'xgb': XGBRegressor(),
    'mlp': MLPRegressor(
        alpha=1e-4,
        hidden_layer_sizes=32,
        solver='adam',
        learning_rate='invscaling',
        warm_start=True,
        early_stopping=True,
    ),
    'sir': SimpleImputerRegressor(
        strategy='median'
    ),
}

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
# Obtain evaluation scores
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
    #'norm_rmse': make_scorer(norm_rmse) # NRMSE
}

# Compendium of results
iir_results = pd.DataFrame()

# Create a list of estimators
ESTIMATORS = [
    # 'lr',
    # 'bridge',
    # 'dt',
    # 'etr',
    # 'sgd-ls',
    # 'sgd-sv',
    # 'knn',
    # 'xgb',
    # 'sir',
]

# For each estimator
for i, est in enumerate(ESTIMATORS):

    data = pd.DataFrame()

    # Check if estimator has been defined else skip
    if est not in _TUNED_ESTIMATORS:
        continue
    
    estimator = _TUNED_ESTIMATORS[est]
    
    if estimator != 'sir':
        imputer = IterativeImputerRegressor(estimator=estimator)
    else:
        imputer = estimator

    for biomarker in complete_profiles:

        aux = complete_profiles.copy(deep=True)
        X = aux[[x for x in aux.columns if x != biomarker]]
        y = aux[biomarker]

        # Information
        print("\n%s. Evaluating... %s for biomarker... %s" % (i, est, biomarker))

        # Create pipeline
        pipe = Pipeline(steps=[ ('std', StandardScaler()),
                                (est, imputer)],
                        verbose=True)

        # Obtain scores for each fold using cross_validate
        scores = cross_validate(pipe, 
                                X, 
                                y, 
                                scoring=scoring, 
                                cv=skf, 
                                return_train_score=True, 
                                n_jobs=-1, 
                                verbose=0)
        
        # Extract results
        results = pd.DataFrame(scores)
        results.index = ['%s_%s_%s' % (biomarker, est, j)
            for j in range(results.shape[0])]
        
        # Add to compendium and data
        iir_results = iir_results.append(results)
        data = data.append(results)
        # data.to_csv(f'datasets/iir_{est}.csv')


#######################################
# -------------------------------------
# Save results
# -------------------------------------

# Save
# iir_results.to_csv('datasets/iir_results.csv')


#######################################
# -------------------------------------
# Analyse scores and test results
# -------------------------------------

# Create a list of estimators
METHODS = [
    'lr',
    'bridge',
    'dt',
    'etr',
    'sgd-ls',
    'sgd-sv',
    'knn',
    'mlp',
    # 'xgb',
    'sir',
]


compendium = pd.read_csv('datasets/iir_results.csv', index_col=0)

# Get mean and variance of RMSE scores
all_scores = get_score_statistics(compendium, 'rmse')

# Split scores to obtain score for each estimator
split_scores = np.array_split(all_scores, len(METHODS))

# Stack scores horizontally for easier plotting
hsplit_scores = np.hstack((split_scores))

# Create DataFrame for mean and std dev statistics
statistics = pd.DataFrame(hsplit_scores, index=complete_profiles.columns)

# Split mean and std dev statistics
mean_stats, std_stats = statistics.iloc[:,::2], statistics.iloc[:,1::2]

# Rename columns to match algorithms
mean_stats.columns, std_stats.columns = METHODS, METHODS

print("Mean RMSE Statistics: ")

# Highlighting the minimum values of last 2 columns
mean_stats.style.highlight_min(color = 'lightgreen', 
                       axis = 1)

#######################################
# -------------------------------------
# Plot results
# -------------------------------------

plt.figure(figsize=(20,40))

# Set single title for all figures
# plt.suptitle('Iterative Imputer RMSE scores for complete profiles', 
#             fontweight='bold', 
#             fontsize=12)


for idx, (biomarker, scores) in enumerate(mean_stats.iterrows(), start=1):
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