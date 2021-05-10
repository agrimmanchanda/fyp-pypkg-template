"""
Iterative Imputer Experiment I.II
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

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa

# Libraries sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Regressors
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor

# Metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error

# Custom Packages
from pkgname.utils.load_dataset import *
from pkgname.utils.iter_imp import *
from pkgname.core.iter_imp import IterativeImputerRegressor, SimpleImputerRegressor


#######################################
# -------------------------------------
# Define parameter grids
# -------------------------------------
param_grid_lr = {}
param_grid_ridge = {
    'ridge__alpha': [x / 10 for x in range(1, 11)],
}

param_grid_bridge = {
    'bridge__alpha_1': [1e-5, 1e-6, 1e-7],
    'bridge__alpha_2': [1e-5, 1e-6, 1e-7],
    'bridge__lambda_1': [1e-5, 1e-6, 1e-7],
    'bridge__lambda_2': [1e-5, 1e-6, 1e-7],
}
param_grid_iir = {
    'iir__estimator': [
        BayesianRidge()
    ]
}

param_grid_sir = {
    'sir__strategy': [
        'mean',
        'median'
    ]
}

param_grid_rfr = {
    'rfr__n_estimators': [10, 50]
}

_DEFAULT_PARAM_GRIDS = {
    'lr': param_grid_lr,
    'ridge': param_grid_ridge,
    'bridge': param_grid_bridge,
    'iir': param_grid_iir,
    'rfr': param_grid_rfr,
    'sir': param_grid_sir,
}

_DEFAULT_ESTIMATORS = {
    'lr': LinearRegression(),
    'ridge': Ridge(),
    'bridge': BayesianRidge(),
    'iir': IterativeImputerRegressor(),
    'rfr': RandomForestRegressor(),
    'sir': SimpleImputerRegressor(),
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
# Grid Search (with just regressor)
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

# Parameter Grid
param_grid = {}

# Compendium of results
compendium = pd.DataFrame()

# Create a list of estimators
ESTIMATORS = [
    'lr',
    #'ridge',
    #'bridge',
    # 'iir',
    # 'sir',
]

# For each estimator
for i, est in enumerate(ESTIMATORS):

    # Basic checks
    if est not in _DEFAULT_ESTIMATORS:
        continue
    if est not in _DEFAULT_PARAM_GRIDS:
        continue

    for biomarker in complete_profiles:

        aux = complete_profiles.copy(deep=True)
        X = aux[[x for x in aux.columns if x != biomarker]]
        y = aux[biomarker]

        # Information
        print("\n%s. Evaluating... %s for biomarker... %s" % (i, est, biomarker))

        # Create pipeline
        pipe = Pipeline(steps=[ ('std', StandardScaler()),
                                (est, _DEFAULT_ESTIMATORS[est])],
                        verbose=True)

        # Create grid search (another option is RandomSearchCV)
        grid = GridSearchCV(pipe, param_grid=_DEFAULT_PARAM_GRIDS[est],
                            cv=skf, scoring=scoring,
                            return_train_score=True, verbose=0,
                            refit=False, n_jobs=-1)

        # Fit grid search
        grid.fit(X, y)

        # Extract results
        results = pd.DataFrame(grid.cv_results_)
        results.index = ['%s_%s_%s' % (est, j, biomarker)
            for j in range(results.shape[0])]
        
        # Add to compendium
        compendium = compendium.append(results)


#######################################
# -------------------------------------
# Show and save
# -------------------------------------

# # Show grid search scores
# print("\n\nGrid Search result:")
# print(compendium.T)

# Save
compendium.to_csv('datasets/compendium.csv')