"""
Initial results of prototyping with IterativeImputer
============================

"""

# Libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
import os
import pathlib

# -------------------------------
# Create configuration from data
# -------------------------------
# Current path
curr_path = pathlib.Path(__file__).parent.absolute()

path_data = '{0}/dataset/{1}'.format(
    curr_path, 'iter_imp_data_raw.csv')

# Load dataset
og_df = pd.read_csv(f'{path_data}')
new_df = og_df.copy()
new_lst = [col for col in new_df]

# Main script:
for j in range(1,5):

    # randomly remove 20, 40, 60 and 80% of values from each column
    for col in new_df.columns:
        new_df.loc[new_df.sample(frac=0.2*j).index, col] = np.nan

    # IterativeImputer with ExtraTreesRegressor 
    imp_it = IterativeImputer(max_iter=100, estimator=ExtraTreesRegressor(n_estimators=10, random_state=0))

    # use the regressor to fit and transform on the same data set
    new_df = imp_it.fit_transform(new_df)

    # create a dataframe from the array identical to original data set
    new_df = pd.DataFrame(data=new_df[0:,0:], index=[i for i in range(new_df.shape[0])], columns=[col for col in og_df.columns])

    x_labels = [col for col in new_df]

    actual_lst = []
    pred_lst = []

    # create a (zipped) list to compare values for each variable
    for col in og_df.columns:
        actual_lst.append(list(og_df[col]))
        pred_lst.append(list(new_df[col]))

    comb_lst = zip(actual_lst, pred_lst)

    # calculate the MAE for each variable
    rms_lst = []
    for i in comb_lst:
        rms = mean_absolute_error(i[0],i[1])
        rms_lst.append(rms)

    # plot the graphs and set appropriate labels
    plt.figure()
    plt.bar(x_labels, rms_lst)
    plt.title('Graph showing MAE error for %i percent missing values' % (20*j))
    plt.xlabel('Percentage of missing values (per code): %i' % (20*j))
    plt.ylabel('MAE')
    plt.xticks(new_lst, rotation='vertical')

    # save each plot (.png) in examples/IterativeImputer/outputs/
    filename = f"{20*j}.png"
    path = "examples/IterativeImputer/outputs"
    fullpath = os.path.join(path, filename)
    plt.savefig(filename)
