# Iterative Imputer Class Methods

# Generic Methods
import numpy as np
import pandas as pd
import sklearn

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer # noqa
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
from sklearn.metrics import mean_squared_log_error


class IterativeImputerRegressor(IterativeImputer):

    def fit(self, X, y, **kwargs):
        """
        Concatenate two columns to treat each
        X as the entire dataset inclusive of the
        target column.
        """
        # Cannot reshape a series
        if isinstance(y, pd.Series):
            y = y.to_numpy().reshape(-1, 1)

        # Fit with all data
        super().fit(np.hstack((X, y)), **kwargs)

    def predict(self, X):
        """Adding predict (required in pipelines)

        A new column is created and imputed with all
        NaN values which is then concatenated with the
        dataset and passed to transform method of the 
        base class to impute. We are only interested 
        in the imputed column results so the last 
        column is returned. 
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

class SimpleImputerRegressor(SimpleImputer):

    def fit(self, X, y, **kwargs):
        """
        Concatenate two columns to treat each
        X as the entire dataset inclusive of the
        target column.
        """
        # Cannot reshape a series
        if isinstance(y, pd.Series):
            y = y.to_numpy().reshape(-1, 1)

        # Fit with all data
        super().fit(np.hstack((X, y)), **kwargs)

    def predict(self, X):
        """Adding predict (required in pipelines)

        A new column is created and imputed with all
        NaN values which is then concatenated with the
        dataset and passed to transform method of the 
        base class to impute. We are only interested 
        in the imputed column results so the last 
        column is returned.
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

class CustomIterativeImputer(IterativeImputer):
    
    def get_nan_features(self, X):
        
        return X.columns[X.isna().any()].tolist()

    def fit(self, X, **kwargs):
        """
        Fit the data on the input dataset.
        """
        
        nan_features = self.get_nan_features(X)
        
        aux = np.copy(X)
        
        super().fit_transform(aux)
        
        return self

    def predict(self, X):
        """Adding predict

        Transform the data using round robin fashion
        and round the results to 2 d.p. as that is the
        precision of the raw data set. 

        """
        # Copy just for safety
        aux = np.copy(X)
        
        # Get columns of original DataFrame
        cols = X.columns
        
        # Return numpy array with transformations
        imputed_data_ = np.abs(self.transform(aux))
        
        # Round values to 2 d.p 
        imputed_data = np.around(imputed_data_, 2)
        
        # Convert back to DataFrame
        return_data = pd.DataFrame(data=imputed_data, 
                                  columns=cols)
    
        # Return pandas with same shape as input
        return imputed_data