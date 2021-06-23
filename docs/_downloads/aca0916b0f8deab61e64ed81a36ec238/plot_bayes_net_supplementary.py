"""
Bayesian Network Supplementary Script
===========================================

Provides all supplementary code for experiments.

"""

# #######################################
# # -------------------------------------
# # Libraries import
# # -------------------------------------

# # Libraries generic
# import numpy as np
# import pandas as pd
# import sklearn
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats

# # Libraries sklearn
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_validate
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

# # Regressors
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import BayesianRidge
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import ExtraTreesRegressor
# from sklearn.linear_model import SGDRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.neural_network import MLPRegressor
# from xgboost import XGBRegressor

# # Metrics
# from sklearn.metrics import make_scorer
# from sklearn.metrics import mean_squared_error

# # Custom Packages
# from labimputer.utils.load_dataset import remove_data_outliers
# from labimputer.utils.iter_imp import corr_pairs, get_score_statistics, rmse, norm_rmse, rmsle, get_test_scores, get_best_models, nae
# from labimputer.core.iter_imp import IterativeImputerRegressor, SimpleImputerRegressor
# from labimputer.core.bayes_net import BNRegressor, BNImputer

# METHODS = [
#     'LR',
#     'DT',
#     'RF',
#     'SVR',
#     'KNN',
#     'MLP',
#     'XGB',
#     'Median',
# ]

# FBC_CODES = sorted(["EOS", "MONO", "NEUT", "RBC", "WBC", 
#                 "MCHC", "MCV", "LY", "HCT", "RDW", "HGB", 
#                 "MCH", "PLT", "MPV"])


# ## Test results 

# df1 = pd.read_csv('ML_mult_test_results_30.csv', index_col=0)

# df1

# data = np.split(df1.T.to_numpy(), len(df1.T.to_numpy())/3)

# d = pd.DataFrame()

# for idx, values in enumerate(zip(data, FBC_CODES)):
    
#     y_true, y_pred, y_med = values[0][0], values[0][1], values[0][2]
    
#     rmse_tp, rmse_tm = rmse(y_true, y_pred), rmse(y_true, y_med)
    
#     join = pd.concat([pd.Series(rmse_tp), pd.Series(rmse_tm)], axis=1)
    
#     d = d.append(join)
    
# d.columns, d.index = ['Best', 'Median'], FBC_CODES

# d['$\Delta$ (%)'] = 100 - (100* (d['Best']/d['Median']))

# d.loc['Mean'] = d.mean()

# for idx, values in enumerate(zip(data, FBC_CODES)):
    
#     y_true, y_pred, y_med = values[0][0], values[0][1], values[0][2]
    
#     t1 = stats.mannwhitneyu(y_true, y_med)
    
#     print(values[1], t1)


# d = pd.DataFrame()

# for idx, values in enumerate(zip(data, FBC_CODES)):
    
#     y_true, y_pred, y_med = values[0][0], values[0][1], values[0][2]
    
#     nae_tp, nae_tm = nae(y_true, y_pred), nae(y_true, y_med)
    
#     nae_vals = pd.DataFrame([nae_tp, ['Best' for _ in range(len(nae_tp))], [values[1] for _ in range(len(nae_tp))]]).T
    
#     nae_vals_med = pd.DataFrame([nae_tm, ['Median' for _ in range(len(nae_tm))], [values[1] for _ in range(len(nae_tm))]]).T
    
#     join = pd.concat([nae_vals, nae_vals_med], axis=0)
    
#     d = d.append(join)
    
# d.columns = ['NAE', 'Model', 'Analyte']

# plt.figure(figsize=(15,5))

# # create grouped boxplot 
# sns.boxplot(x = d['Analyte'],
#         y = d['NAE'],
#         hue = d['Model'],
#         showfliers=False
#         )

# plt.show()