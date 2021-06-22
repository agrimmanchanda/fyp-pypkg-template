# Load Dataset Helper Methods

# Libraries 
import pandas as pd
import numpy as np
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
import sys, os

# Function to remove outliers based using Q(1/3) -+ 1.5 * IQR
def remove_data_outliers(df, coeff=1.5, tol=0):
    """Utility function to remove data outliers from the data set using Z-Score method.

    Args:
        df (DataFrame): Input transformed data set.
        coeff (float, optional): Scaling coefficient for outlier removal strategy. Defaults to 1.5.
        tol (int, optional): Feature tolerance on how many features with NaN values may exist. Defaults to 0.

    Returns:
        Data set (DataFrame): Output data set without outlier values.
        Outliers per variable (Series): Number of outliers remaining per variable.
    """

    # Calculate IQR = Q3 - Q1
    q1, q3 = df.quantile(.25), df.quantile(.75)
    IQR = q3 - q1
    lower_bound = q1 - (coeff * IQR)
    upper_bound = q3 + (coeff * IQR)
    
    # Set the outliers to NaN
    df[(df < lower_bound) | (df > upper_bound)] = np.nan
    
    # Drop NaN values with option for adjusting threshold
    df = df.dropna(how='any', thresh = df.shape[1] - tol)
    
    # Store number of outliers per column
    outlier_count = df.isnull().sum(axis = 0)
    
    # Return complete profiles and outliers per column
    return df, outlier_count

# Code snippet from: shorturl.at/gAOR2
@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

# Code snippet from: shorturl.at/lquDN
@contextmanager
def suppress_stdout():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout