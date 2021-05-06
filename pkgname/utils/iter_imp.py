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

def get_metric_scores(true, pred, metric):

    # Check that they are the same shape
    assert len(true) == len(pred)
    
    if metric == 'RMSE':
        return mean_squared_error(true, pred, squared=False)

    elif metric == 'RMSLE':
        return np.sqrt(mean_squared_log_error(true, pred))

    else:
        return 0