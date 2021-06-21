"""
Iterative Imputer Supplementary Script
===========================================

Provides all supplementary code for experiments.

"""

#######################################
# -------------------------------------
# Libraries import
# -------------------------------------

# Libraries generic
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Libraries sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Regressors
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# Metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error

# Custom Packages
from labimputer.utils.load_dataset import remove_data_outliers
from labimputer.utils.iter_imp import corr_pairs, get_score_statistics, rmse, norm_rmse, rmsle, get_test_scores, get_best_models, nae
from labimputer.core.iter_imp import IterativeImputerRegressor, SimpleImputerRegressor

df = pd.read_csv('iir_simple_cv_results.csv', index_col=0)

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

FBC_CODES = sorted(["EOS", "MONO", "NEUT", "RBC", "WBC", 
                "MCHC", "MCV", "LY", "HCT", "RDW", "HGB", 
                "MCH", "PLT", "MPV"])

# Get mean and variance of RMSE scores
all_scores = get_score_statistics(df, metric='rmse')

# Split scores to obtain score for each estimator
split_scores = np.array_split(all_scores, 8)

# Stack scores horizontally for easier plotting
hsplit_scores = np.hstack((split_scores))

# Create DataFrame for mean and std dev statistics
statistics = pd.DataFrame(hsplit_scores, index=FBC_CODES)

# Split mean and std dev statistics
mean_stats, std_stats = statistics.iloc[:,::2], statistics.iloc[:,1::2]

# Rename columns to match algorithms
mean_stats.columns, std_stats.columns = METHODS, METHODS

print("Mean RMSE Statistics: ")

# Highlighting the minimum values of last 2 columns
mean_stats.style.highlight_min(color = 'lightgreen', 
                       axis = 1)

mean_stats.loc["Mean"] = mean_stats.mean()

BEST_MODELS = get_best_models(mean_stats)

best_cv_models = {}

for biomarker, value in mean_stats.iterrows():
    
    best_cv_models[biomarker] = value.min()
    
new_df = pd.DataFrame.from_dict(best_cv_models, orient='index')

new_df['Median'] = mean_stats['Median']

new_df.columns = ['Best', 'Median']

new_df['$\Delta$ (%)'] = 100 - (100* (new_df['Best']/new_df['Median']))

new_df['Model'] = BEST_MODELS.values()

new_df.loc['Mean'] = new_df.mean()



plt.figure(figsize=(15,5))

new_df = new_df[:-1]

plot = sns.barplot(x=new_df.index, y=new_df['$\Delta$ (%)'], hue=new_df['Model'], dodge=False)

plot.set_xlabel("Analyte")

#plt.axhline(new_df['$\Delta$ (%)'].mean(), ls='--', color='r', label='Mean')

plt.show()


## Test results 

df1 = pd.read_csv('iir_simple_test_results_1.csv', index_col=0)

df1

data = np.split(df1.T.to_numpy(), len(df1.T.to_numpy())/3)

d = pd.DataFrame()

for idx, values in enumerate(zip(data, FBC_CODES)):
    
    y_true, y_pred, y_med = values[0][0], values[0][1], values[0][2]
    
    rmse_tp, rmse_tm = rmse(y_true, y_pred), rmse(y_true, y_med)
    
    join = pd.concat([pd.Series(rmse_tp), pd.Series(rmse_tm)], axis=1)
    
    d = d.append(join)
    
d.columns, d.index = ['Best', 'Median'], FBC_CODES

d['$\Delta$ (%)'] = 100 - (100* (d['Best']/d['Median']))

d['Model'] = BEST_MODELS.values()

d.loc['Mean'] = d.mean()

for idx, values in enumerate(zip(data, FBC_CODES)):
    
    y_true, y_pred, y_med = values[0][0], values[0][1], values[0][2]
    
    t1 = stats.mannwhitneyu(y_true, y_med)
    
    print(values[1], t1)


d = pd.DataFrame()

for idx, values in enumerate(zip(data, FBC_CODES)):
    
    y_true, y_pred, y_med = values[0][0], values[0][1], values[0][2]
    
    nae_tp, nae_tm = nae(y_true, y_pred), nae(y_true, y_med)
    
    nae_vals = pd.DataFrame([nae_tp, ['Best' for _ in range(len(nae_tp))], [values[1] for _ in range(len(nae_tp))]]).T
    
    nae_vals_med = pd.DataFrame([nae_tm, ['Median' for _ in range(len(nae_tm))], [values[1] for _ in range(len(nae_tm))]]).T
    
    join = pd.concat([nae_vals, nae_vals_med], axis=0)
    
    d = d.append(join)
    
d.columns = ['NAE', 'Model', 'Analyte']

plt.figure(figsize=(15,5))

# create grouped boxplot 
sns.boxplot(x = d['Analyte'],
        y = d['NAE'],
        hue = d['Model'],
        showfliers=False
        )

plt.show()