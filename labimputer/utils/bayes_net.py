# Bayesian Network Helper Methods

# Libraries 
import pandas as pd
import numpy as np 
import itertools
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from scipy import stats

# Function to get unique variables in model edges
def get_unique_edges(model_edges):

    # create a list of edge1 and edge2
    edge1, edge2 = [x[0] for x in model_edges], [x[1] for x in model_edges]

    # concatenate the list and use a set to get unique elements
    uniq_edges = list(set(edge1+edge2))

    # return the edges as a (alphabetically sorted) list
    return sorted(uniq_edges)

# Function to get each dataframe row as dictionary
def get_row_from_df(row, columns):

    # create a dict for each row
    sample = {}

    # create a dict with feature and corresponding value
    for idx, col in enumerate(columns, start=1):
        sample[col] = row[idx]

    # return the dict containing data for that row
    return sample

# Function to get score statistics for each compendium
def get_score_statistics(df, metric, folds=5, combined=True):

    # Use 5-fold CV
    FOLDS = folds
    
    # get score for specified metric
    if metric == 'rmse':
        scores = df.test_nrmse.values # RMSE
    elif metric == 'mse':
        scores = df.test_nmse.values # MSE
    else:
        scores = df.test_nmae.values # MAE
    
    # Take absolute value 
    scores = scores * -1
    
    mean_scores, std_scores = [], []
    
    # Loop over all scores using folds
    for i in range(0, len(scores), FOLDS):
        mean = np.mean(scores[i:i+FOLDS])
        std = np.std(scores[i:i+FOLDS])
        mean_scores.append(mean)
        std_scores.append(std)
    
    # Return combined or separate scores
    if combined:
        return list(zip(mean_scores, std_scores))
    else:
        return mean_scores, std_scores

# Function to return the data statistics
def get_data_statistics(df, panel, num):
    
    data = np.split(df.T.to_numpy(), len(df.T.to_numpy())/num)

    rmse_dict = {}
    
    rmse_dict2 = {}
    
    mw_test = {}

    for idx, values in enumerate(zip(data, panel)):

        y_true, y_pred, y_med = values[0][0], values[0][1], values[0][2]

        rmse_tp, rmse_tm = rmse(y_true, y_pred), rmse(y_true, y_med)

        rmse_dict[values[1]] = rmse_tp
        rmse_dict2[values[1]] = rmse_tm
        mw_test[values[1]] = stats.mannwhitneyu(y_true, y_pred)[1]
        
    return pd.concat([pd.Series(rmse_dict), pd.Series(rmse_dict2), pd.Series(mw_test)], axis=1)


# Function to return the data statistics for only true and predicted
def get_simple_data_stats(df, panel, num):
    
    data = np.split(df.T.to_numpy(), len(df.T.to_numpy())/num)

    rmse_dict = {}

    for idx, values in enumerate(zip(data, panel)):

        y_true, y_pred = values[0][0], values[0][1]

        rmse_tp = rmse(y_true, y_pred)

        rmse_dict[values[1]] = rmse_tp
        
    return pd.DataFrame.from_dict(rmse_dict, orient='index')

# RMSE for grisearchCV
def rmse(y_true, y_pred, **kwargs):
    """ Returns the RMSE scores

    Args:
        y_true (array-like): Array of true values.
        y_pred (array-like): Array of predicted values.

    Returns:
        rmse (float): Returns the RMSE for given data.
    """

    return mean_squared_error(y_true, y_pred, squared=False, **kwargs)

# NRMSE for gridsearchCV
def norm_rmse(y_true, y_pred, **kwargs):
    """ Returns the normalised RMSE score

    Args:
        y_true (array-like): Array of true values.
        y_pred (array-like): Array of predicted values.

    Returns:
        nrmse (float): Returns the NRMSE for given data.
    """

    # Calculate RMSE score
    score = rmse(y_true, y_pred, **kwargs)

    # Calculate spread of data
    spread = max(y_true) - min(y_pred)
    if spread != 0:
        return score/spread
    else:
        return score

# RMSLE for gridsearchCV
def rmsle(y_true, y_pred, **kwargs):
    """ Returns the RMSLE scores

    Args:
        y_true (array-like): Array of true values.
        y_pred (array-like): Array of predicted values.

    Returns:
        rmsle (float): Returns the RMSLE for given data.
    """

    return np.sqrt(mean_squared_log_error(y_true, y_pred, **kwargs))

def nae(y_true, y_pred):
    """ Returns the NAE scores

    Args:
        y_true (array-like): Array of true values.
        y_pred (array-like): Array of predicted values.

    Returns:
        nae (float): Returns the NAE for given data.
    """
    
    return np.abs(y_pred - y_true)/(max(y_true) - min(y_true))