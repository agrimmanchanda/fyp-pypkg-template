# Iterative Imputer Helper Methods

# Libraries 
import pandas as pd
import numpy as np 
import itertools
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

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

# Function to get highest correlation pairs 
def corr_pairs(df):
    """Returns the correlation pairs for highest correlated variables

    Args:
        df (DataFrame): Input data to function.

    Returns:
        pairs (int): Output data with top five correlation pairs.
    """

    # Code adapted from:
    # shorturl.at/izF04
    df1 = pd.DataFrame([[i, j, df.corr().loc[i,j]] 
    for i,j in list(itertools.combinations(df.corr(), 2))],
    columns=['var1', 'var2','corr'])    
    
    pairs = df1.sort_values(by='corr',ascending=False).head(5)
    
    return pairs


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

def get_metric_scores(true, pred, metric):
    """ Returns the score for requested metric

    Args:
        true (array-like): Array of true values.
        pred (array-like): Array of predicted values.
        metric (string): Name of metric to return score.

    Returns:
        score: Returns the requested score.
    """

    # Check that they are the same shape
    assert len(true) == len(pred)
    
    if metric == 'RMSE':
        return mean_squared_error(true, pred, squared=False)

    elif metric == 'RMSLE':
        return np.sqrt(mean_squared_log_error(true, pred))

    elif metric == 'NRMSE':
        rmse = mean_squared_error(true, pred, squared=False)
        return rmse/(max(pred) - min(pred))

    else:
        return 0

def get_test_scores(true, pred):
    """ Returns the RMSE, RMSLE and NRMSE score

    Args:
        true (array-like): Array of true values.
        pred (array-like): Array of predicted values.

    Returns:
        scores (list of int): List of three scoring metrics.
    """

    rmse_score = rmse(true, pred)

    rmsle_score = rmsle(true, pred)

    nrmse_score = norm_rmse(true, pred)

    return [rmse_score, rmsle_score, nrmse_score]

def get_score_statistics(df, metric, folds=5, combined=True):

    """Gets the score statistics for cross validation training set

    Args:
        df (DataFrame): Input data containing results.
        metric (string): Metric to return score statistics for.
        folds (int): Number of folds chosen in cross validation. Default at 5.
        combined (bool): Return statistics together or separate. Default at True.

    Returns:
        statistics (list or tuple): Returns list of mean and standard statistics.
    """

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


def get_best_models(df):
    """Returns the best model for each feature

    Args:
        df (DataFrame): Input data containing containing results for each feature.

    Returns:
        best_models (Dict): Returns the best model for each feature as a dict of strings.
    """

    # Keep dictionary of all the best models
    best_models = {}

    # Loop over each biomarker and value
    for biomarker, value in df.iterrows():
        
        # Find the best model for each biomarker
        best_models[biomarker] = value.idxmin()

    # Return 
    return best_models

def get_cvts_stats(df, panel):
    """Returns the CVTS statistics

    Args:
        df (DataFrame): Input data containing the CVTS scores.
        panel (List of strings): The laboratory panel of interest

    Returns:
        statistics (Tuple): Returns the mean and standard deviation scores.
    """

    # Get mean and variance of RMSE scores
    all_scores = get_score_statistics(df, metric='rmse')

    # Split scores to obtain score for each estimator
    split_scores = np.array_split(all_scores, 8)

    # Stack scores horizontally for easier plotting
    hsplit_scores = np.hstack((split_scores))

    # Create DataFrame for mean and std dev statistics
    statistics = pd.DataFrame(hsplit_scores, index=panel)

    # Split mean and std dev statistics
    mean_stats, std_stats = statistics.iloc[:,::2], statistics.iloc[:,1::2]

    # Rename columns to match algorithms
    mean_stats.columns, std_stats.columns = METHODS, METHODS

    # Find the mean RMSE score for each method
    mean_stats.loc["Mean"] = mean_stats.mean()

    return mean_stats, std_stats

def get_cvts_delta(df, model):
    """Returns the Delta metric which measures improvement on RMSE.

    Args:
        df (DataFrame): Input data containing the CVTS scores.
        model (Dict): Dictionary containing best mode for each feature.

    Returns:
        cvts_best_df (DataFrame): Returns the Delta for best models. 
    """

    # Create dictionary for CVTS models
    cvts_models = {}

    # Loop for each biomarker
    for biomarker, value in df.iterrows():
        
        # Keep record of lowest value for each model
        cvts_models[biomarker] = value.min()

    # Store best df scores in DataFrame
    cvts_best_df = pd.DataFrame.from_dict(cvts_models, orient='index')

    # Create a new column for simple median imputation scores
    cvts_best_df['Median'] = df['Median']

    # Define columns of best models
    cvts_best_df.columns = ['Best', 'Median']

    # Define Delta Metric measure
    cvts_best_df['$\Delta$ (%)'] = 100 - (100* (cvts_best_df['Best']/cvts_best_df['Median']))

    # Create a column to display best model results
    cvts_best_df['Model'] = model.values()

    # Create a row to diplay mean for best and median
    cvts_best_df.loc['Mean'] = cvts_best_df.mean()

    # Discount the mean row
    cvts_best_df = cvts_best_df[:-1]

    # Return
    return cvts_best_df
