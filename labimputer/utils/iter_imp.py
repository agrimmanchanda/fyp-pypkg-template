# Iterative Imputer Helper Methods

# Libraries 
import pandas as pd
import numpy as np 
import itertools
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

# Function to get highest correlation pairs 
def corr_pairs(df):

    # Code adapted from:
    # shorturl.at/izF04
    df1 = pd.DataFrame([[i, j, df.corr().loc[i,j]] 
    for i,j in list(itertools.combinations(df.corr(), 2))],
    columns=['var1', 'var2','corr'])    
    
    pairs = df1.sort_values(by='corr',ascending=False).head(5)
    
    return pairs


# RMSE for grisearchCV
def rmse(y_true, y_pred, **kwargs):

    return mean_squared_error(y_true, y_pred, squared=False, **kwargs)

# NRMSE for gridsearchCV
def norm_rmse(y_true, y_pred, **kwargs):
    score = rmse(y_true, y_pred, **kwargs)

    spread = max(y_true) - min(y_pred)
    if spread != 0:
        return score/spread
    else:
        return score

# RMSLE for gridsearchCV
def rmsle(y_true, y_pred, **kwargs):

    return np.sqrt(mean_squared_log_error(y_true, y_pred, **kwargs))

def nae(y_true, y_pred):
    
    return np.abs(y_pred - y_true)/(max(y_true) - min(y_true))

def get_metric_scores(true, pred, metric):

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

    rmse_score = rmse(true, pred)

    rmsle_score = rmsle(true, pred)

    nrmse_score = norm_rmse(true, pred)

    return [rmse_score, rmsle_score, nrmse_score]

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


def get_best_models(df):

    best_models = {}

    for biomarker, value in mean_stats.iterrows():

        best_models[biomarker] = value.idxmin()
        
    return best_models