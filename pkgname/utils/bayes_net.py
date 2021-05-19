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