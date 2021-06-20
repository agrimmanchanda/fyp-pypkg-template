# Bayesian Network Helper Methods

# Libraries 
import pandas as pd
import numpy as np 
import itertools

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