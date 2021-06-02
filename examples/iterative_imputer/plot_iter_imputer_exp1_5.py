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
from pkgname.utils.load_dataset import remove_data_outliers
from pkgname.utils.iter_imp import corr_pairs, get_score_statistics, rmse, norm_rmse, rmsle, get_test_scores
from pkgname.core.iter_imp import IterativeImputerRegressor, SimpleImputerRegressor

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
        n_neighbors=8,
        weights='distance',
        n_jobs=-1,
    ),
    'xgb': XGBRegressor(
        n_estimators=100,
        eval_metric='rmse',
        max_depth=8,
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

FBC_CODES = sorted(["EOS", "MONO", "BASO", "NEUT", "RBC", "WBC", 
                "MCHC", "MCV", "LY", "HCT", "RDW", "HGB", 
                "MCH", "PLT", "MPV", "NRBCA"])

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
# Split into train-test
# -------------------------------------

SEED = 8

train_set, test_set = train_test_split(complete_profiles, shuffle=False, test_size=0.2, random_state=8)

for col in train_set.columns:
    train_set.loc[train_set.sample(frac=0.1).index, col] = np.nan
    test_set.loc[test_set.sample(frac=0.1).index, col] = np.nan


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
    'rmsle': make_scorer(rmsle), # RMSLE
    'norm_rmse': make_scorer(norm_rmse), # NRMSE
}

# Compendium of results
iir_results = pd.DataFrame()

# Create a list of estimators
ESTIMATORS = [
    'lr',
    # 'dt',
    # 'rf',
    # 'svr',
    # 'knn',
    # 'mlp',
    # 'xgb',
    'median',
]

test_data = pd.DataFrame()

# For each estimator
for i, est in enumerate(ESTIMATORS):

    test_scores = {}

    # Check if estimator has been defined else skip
    if est not in _TUNED_ESTIMATORS:
        continue
    
    estimator = _TUNED_ESTIMATORS[est]
    
    if est != 'median':
        imputer = IterativeImputerRegressor(estimator=estimator,
                                            min_value=0, 
                                            max_iter=10000)
    else:
        imputer = estimator

    for biomarker in train_set:

        aux_train = train_set.copy()
        aux_test = test_set.copy()

        X_train = aux_train[[x for x in aux_train.columns if x != biomarker]]
        y_train = aux_train[biomarker]

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

        test_scores[biomarker] = get_test_scores(y_test, y_pred)

        # Extract results
        results = pd.DataFrame(scores)
        results.index = ['%s_%s_%s' % (biomarker, est, j)
            for j in range(results.shape[0])]
        
        # Add to compendium of results
        iir_results = iir_results.append(results)
    
    # Concatenate scores for the estimator to all other test scores
    test_data = pd.concat([test_data, pd.Series(test_scores, name=est)], axis=1)

    print(test_data)