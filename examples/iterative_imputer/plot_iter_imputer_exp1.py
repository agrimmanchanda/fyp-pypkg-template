"""
Iterative Imputer Experiment I
===========================================

Single biomarker removal experiment.

"""
#######################################
# Import the relevant libraries first
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
from pkgname.utils.widgets import TidyWidget

#######################################
# -------------------------------------
# Data handling
# -------------------------------------
# First, let's define the data set path and relevant variables of interest

path_data = '../load_dataset/datasets/pathology-sample-march-may.csv'

FBC_codes = ["EOS", "MONO", "BASO", "NEUT", "RBC", "WBC", 
                "MCHC", "MCV", "LY", "HCT", "RDW", "HGB", 
                "MCH", "PLT", "MPV", "NRBCA"]

INTEREST_cols = ["_uid", "orderCode", "result", "dateResult"]

#############################

#######################################
# Next, import only variables of interest and FBC panel results
df = pd.read_csv(path_data, usecols=INTEREST_cols)

df = df.loc[df['orderCode'].isin(FBC_codes)]

df = df.dropna() # drop records of patients with NaN _uid

df.reset_index(drop=True, inplace=True)

# Define function to set pid (patient ID) sorted by datetime

def change_pid_datetime_format(df):
    df['pid'] = df['_uid'].str.extract('(\d+)').astype(int)

    pid_col = df.pop('pid')

    df.insert(0, 'pid', pid_col)

    df.drop('_uid', inplace=True, axis=1)

    df.sort_values(by=['pid', 'dateResult'], inplace=True)

    return df

# Function to get highest correlation pairs 

def corr_pairs(df):

    # Code adapted from: 
    # https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
    df1 = pd.DataFrame([[i, j, df.corr().loc[i,j]] for i,j in list(itertools.combinations(df.corr(), 2))],columns=['var1', 'var2','corr'])    
    
    pairs = df1.sort_values(by='corr',ascending=False).head(5)
    
    return pairs

#######################################
# -------------------------------------
# Transform data using TidyWidget
# -------------------------------------

# Parameters
index = ['_uid', 'dateResult', 'orderCode']
value = 'result'

# Create widget
widget = TidyWidget(index=index, value=value)

# Transform (keep all)
transform, duplicated = \
    widget.transform(df, report_duplicated=True)

# Set pid for each patient and sort accordingly
transform_fmt = change_pid_datetime_format(transform)

# Transform (keep first)
transform_first = \
    widget.transform(df, keep='first')

# Set pid for each patient and sort accordingly
transform_first_fmt = change_pid_datetime_format(transform_first)

#######################################
# -------------------------------------
# Split data into input and output
# -------------------------------------

# Obtain the biomarkers DataFrame only
biomarkers_df = transform_fmt.iloc[:,2:].dropna()
biomarkers_original_df_copy = biomarkers_df.copy()

biomarkers_data = biomarkers_df.values

# Obtain highest correlation pairs Pearson Correlation Coefficient
biomarkers_highest_corr = corr_pairs(biomarkers_df)

# Find biomarkers that are to be dropped from the dataset
biomarkers_to_drop = np.unique(biomarkers_highest_corr[['var1', 'var2']].values)

print("\nBiomarkers with high correlations: ", biomarkers_to_drop)

# TODO: should biomarkers be completely discarded from dataset?

#######################################
# -------------------------------------
# Preprocessing step: normalise
# -------------------------------------

# Define min-max scaler and normalise dataset
min_max_scaler = preprocessing.MinMaxScaler()
val_scaled = min_max_scaler.fit_transform(biomarkers_data)
biomarkers_df = pd.DataFrame(val_scaled)
biomarkers_copy_df = biomarkers_df.copy(deep=True)

#######################################
# -------------------------------------
# Iterative Imputer
# -------------------------------------

# dictionary to store mse scores for each biomarker
mse_scores = {}

for biomarker in biomarkers_df.columns:

    # Use a clean copy of the normalised data set
    bm_copy_df = biomarkers_df.copy(deep=True)

    # Randomly remove 50% of values and set to NaN
    #bm_copy_df.loc[bm_copy_df.sample(frac=0.5).index, biomarker] = np.nan

    # Strictly set every second value of the biomarker to NaN
    col_pos = bm_copy_df.columns.get_loc(biomarker)
    bm_copy_df.iloc[1::2, col_pos] = np.nan

    # Define imputer 
    imputer = IterativeImputer()

    # Fit on the dataset
    biomarker_tansformed_data = imputer.fit_transform(bm_copy_df)

    # Make dataframe of imputed data
    imputed_data = pd.DataFrame(data=biomarker_tansformed_data, index=[i for i in range(biomarker_tansformed_data.shape[0])], columns=[col for col in biomarkers_df.columns])

    # Compute true and obtain real value
    val_pred = imputed_data[biomarker].values
    val_true = biomarkers_copy_df[biomarker].values

    # Calculate MSE scores from the true and predicted values
    mse_score = mean_squared_error(val_true, val_pred)

    # Store it in the mse_scores dict
    mse_scores[biomarker] = mse_score

# Create DataFrame of the dictionary
mse_df = pd.DataFrame(mse_scores.items(), columns=['Biomarker', 'MSE Score'])

#######################################
# -------------------------------------
# Plotting MSE Scores
# -------------------------------------
cols = [col for col in biomarkers_original_df_copy.columns]
ax = mse_df.plot.bar(x='Biomarker', y='MSE Score', rot=0, grid=True, legend=False)
ax.set_title('MSE Scores for 50 percent missing biomarkers', fontdict={'fontsize': 15, 'fontweight': 'bold'})
ax.set_ylabel('MSE Score', fontsize=12)
ax.set_xlabel('Biomarker', fontsize=12)
ax.set_xticklabels(cols, rotation = 45, fontsize=12)
plt.show()
