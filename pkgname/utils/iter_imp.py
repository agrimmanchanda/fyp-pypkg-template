# Iterative Imputer Helper Methods

# Libraries 
import pandas as pd
import numpy as np 
import itertools

# Function to get highest correlation pairs 
def corr_pairs(df):

    # Code adapted from:
    # shorturl.at/izF04
    df1 = pd.DataFrame([[i, j, df.corr().loc[i,j]] 
    for i,j in list(itertools.combinations(df.corr(), 2))],
    columns=['var1', 'var2','corr'])    
    
    pairs = df1.sort_values(by='corr',ascending=False).head(5)
    
    return pairs