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

    spread = max(y_pred) - min(y_pred)
    if spread != 0:
        return score/spread
    else:
        return score

# RMSLE for gridsearchCV
def rmsle(y_true, y_pred, **kwargs):

    return np.sqrt(mean_squared_log_error(y_true, y_pred, **kwargs))

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