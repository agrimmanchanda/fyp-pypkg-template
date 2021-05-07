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
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer

# Regressors
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor

# Metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error


# ------------------------
# Helper methods
# ------------------------
def rmse(y_true, y_pred, **kwargs):
    return mean_squared_error(y_true, y_pred, squared=True, **kwargs)


class IterativeImputerRegressor(IterativeImputer):

    def fit(self, X, y, **kwargs):
        """

        The IterativeImputer needs to have the column to
        predict (y) in the X, as it is considered an
        imputer and not a regressor.
        """
        # Cannot reshape a series
        if isinstance(y, pd.Series):
            y = y.to_numpy().reshape(-1, 1)

        # Fit with all data
        super().fit(np.hstack((X, y)), **kwargs)

    def predict(self, X):
        """Adding predict (required in pipelines)

        This is just a hack. Since we want to predict the 4th column
        with add a new column with empty values so that the iterative
        imputer fills it. Be careful, this might crash as it might be
        a very sensitive hack.
        """
        # Copy just for safety (might not be necessary)
        aux = np.copy(X)
        # Create empty column
        c = np.empty(shape=aux.shape[0]).reshape(-1, 1)
        c[:] = np.nan
        # Add column with empty values
        aux = np.hstack((aux, c))
        # Make prediction (transform)
        aux = self.transform(aux)
        # Return predictions (last column)
        return aux[:, -1]


"""
   Note that the transform method in IterativeImputer is stochastic. Thus,
   if random_state is not fixed, repeated calls, or permuted input, will 
   yield different results.
"""

# ------------------------
# Constants
# ------------------------
param_grid_lr = {}
param_grid_ridge = {
    'ridge__alpha': [1, 2]
}
param_grid_bridge = {}
param_grid_iir = {
    'iir__estimator': [
        LinearRegression(),
        BayesianRidge()
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
}

_DEFAULT_ESTIMATORS = {
    'lr': LinearRegression(),
    'ridge': Ridge(),
    'bridge': BayesianRidge(),
    'iir': IterativeImputerRegressor(),
    'rfr': RandomForestRegressor(),
}

# ------------------------
# Create data
# ------------------------
# Create features
s1 = np.random.randint(low=1, high=100, size=100)
s2 = s1+12
s3 = s1**3
s4 = np.log(s1)

# Create DataFrame
df = pd.DataFrame(
    data=np.stack([s1, s2, s3, s4]).T,
    columns=['s1', 's2', 's3', 's4']
)

# Show
print("\nData:")
print(df)
print("\nCorrelation (pearson):")
print(df.corr())

# ---------------------------------
# Grid search (with just regressor)
# ---------------------------------
# Show valid scorers
print("\nValid scorers:")
print(sorted(sklearn.metrics.SCORERS.keys()))

# Number of splits
n_splits = 10

# Don't shuffle the folds to ensure that all iteration get
# the same partition between train and test data and
# therefore the results are comparably 100% (paired)

# Create Kfold instance
skf = KFold(n_splits=n_splits, shuffle=False)

# Note you can define your own metrics by defining your own
# function receiving y_true, y_pred and any other arguments
# and wrapping it with the make_scorer function.

# Scoring
scoring = {
    'nmse': 'neg_mean_squared_error',       # MSE
    'nrmse': 'neg_root_mean_squared_error', # RMSE
    'rmse': make_scorer(rmse)               # Create any metric!
}

# Parameter Grid
param_grid = {}

# Create X and y
X = df[['s1', 's2', 's3']]
y = df['s4']

# Compendium of results
compendium = pd.DataFrame()

# For each estimator
for i, est in enumerate(['lr', 'bridge', 'iir']):

    # Basic checks
    if est not in _DEFAULT_ESTIMATORS:
        continue
    if est not in _DEFAULT_PARAM_GRIDS:
        continue

    # Information
    print("\n%s. Evaluating... %s" % (i, est))

    # Note you could apply prior transformations to the data through
    # the pipeline if needed. For instance you can apply the scaler
    # before estimating if needed.

    # Create pipeline
    pipe = Pipeline(steps=[ #('std', StandardScaler()),
                            (est, _DEFAULT_ESTIMATORS[est])],
                    verbose=True)

    # Create grid search (another option is RandomSearchCV)
    grid = GridSearchCV(pipe, param_grid=_DEFAULT_PARAM_GRIDS[est],
                        cv=skf, scoring=scoring,
                        return_train_score=True, verbose=0,
                        refit=False, n_jobs=1)

    # Fit grid search
    grid.fit(X, y)

    # Extract results
    results = pd.DataFrame(grid.cv_results_)
    results.index = ['%s_%s' % (est, j)
        for j in range(results.shape[0])]

    # Add to compendium
    compendium = compendium.append(results)


# ---------------
# Show and save
# ---------------
# Show grid search scores
print("\n\nGrid Search result:")
print(compendium.T)

# Comment
print("""\nNote:\n"""
      """The results using directly the Regressor object and the \n"""
      """results obtained with the hacked IterativeImputerRegressor\n"""
      """are the same.\n""")

# Save
compendium.to_csv('compendium.csv')