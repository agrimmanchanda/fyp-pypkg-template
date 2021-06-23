"""
Experiment 1 and 2: Hyperparameter tuning
===========================================

Hyperparameter tuning using ``sklearn`` 
``GridSearchCV``. 

"""

#######################################
# -------------------------------------
# Libraries import
# -------------------------------------

# Libraries generic
import numpy as np
import pandas as pd
import sklearn

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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor

# Metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error

# Custom Packages
from labimputer.utils.load_dataset import remove_data_outliers
from labimputer.utils.iter_imp import corr_pairs, get_score_statistics
from labimputer.core.iter_imp import IterativeImputerRegressor, SimpleImputerRegressor


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

param_grid_dt = {
    'dt__criterion': ["mse", "mae"],
    'dt__max_depth': [8, 12],
    'dt__min_samples_split': [8, 12],
    'dt__min_samples_leaf': [8, 12],
    'dt__max_leaf_nodes': [10, 15],
}

param_grid_etr = {
    'etr__n_estimators': [x*10 for x in range (1, 11)],
    'etr__criterion': ["mse", "mae"],
    'etr__max_depth': [8, 12],
    'etr__min_samples_split': [8, 12],
    'etr__bootstrap': [False, True],
    'etr__warm_start': [False, True]
}

param_grid_sgd = {
    'sgd__loss': ["squared_loss", 
                "huber", 
                "epsilon_insensitive",
                "squared_epsilon_insensitive"],
    'sgd__alpha': [1e-2, 1e-3, 1e-4],
    'sgd__epsilon': [0.01, 0.05, 0.1],
    'sgd__learning_rate': ["optimal", "invscaling", "adaptive"],
    'sgd__early_stopping': [False, True],
    'sgd__warm_start': [False, True]
}

param_grid_knn = {
    'knn__n_neighbors': [2, 5, 8],
    'knn__weights': ["uniform", "distance"],
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
    'dt': param_grid_dt,
    'etr': param_grid_etr,
    'sgd': param_grid_sgd,
    'knn': param_grid_knn,
    'sir': param_grid_sir,
}

_DEFAULT_ESTIMATORS = {
    'lr': LinearRegression(),
    'ridge': Ridge(),
    'bridge': BayesianRidge(),
    'iir': IterativeImputerRegressor(),
    'rfr': RandomForestRegressor(),
    'dt': DecisionTreeRegressor(),
    'etr': ExtraTreesRegressor(),
    'sgd': SGDRegressor(max_iter=2000),
    'knn': KNeighborsRegressor(),
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
    # 'lr',
    #'ridge',
    #'bridge',
    # 'iir',
    # 'dt',
    # 'etr',
    # 'sgd',
    # 'knn',
    # 'sir',
]

# For each estimator
for i, est in enumerate(ESTIMATORS):

    data = pd.DataFrame()

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
        data = data.append(results)
        data.to_csv(f'datasets/{est}.csv')


#######################################
# -------------------------------------
# Show and save
# -------------------------------------

# # Show grid search scores
# print("\n\nGrid Search result:")
# print(compendium.T)

# Save
# compendium.to_csv('datasets/compendium.csv')