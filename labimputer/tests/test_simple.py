import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pkgname.core.iter_imp import IterativeImputerRegressor
from joblib import load


# Helper methods

def get_x_y_data():

    df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))

    X, y = df[['B', 'C', 'D']], df['A']

    return X, y

def train_linear_regressor(X,y):

    model = LinearRegression()

    model.fit(X,y)

    return model

def train_iterative_imputer_regressor(X,y, estimator=LinearRegression()):

    model = IterativeImputerRegressor(estimator=estimator)

    model.fit(X,y)

    return model


# ================================================
#                      TEST SUITE 
# ================================================

def test_iterative_imputer_linear_regressor_fit():

    X, Y = get_x_y_data()

    model_lr = train_linear_regressor(X, Y)

    coeff_lr = model_lr.coef_

    model_iir = train_iterative_imputer_regressor(X,Y)

    coeff_iir = model_iir.estimator.fit(X,Y).coef_

    assert all([a == b for a, b in zip(coeff_lr, coeff_iir)])


def test_iterative_imputer_linear_regressor_predict():

    X, _ = get_x_y_data()

    model_lr = load('lr.sav')

    model_iir = load('iir.sav')

    y_pred = model_lr.predict(X)

    y_pred_iir = model_iir.predict(X)

    assert all([a == b for a, b in zip(y_pred, y_pred_iir)])


def test_iterative_imputer_linear_regressor_fit_predict():

    X, Y = get_x_y_data()

    model_lr = train_linear_regressor(X, Y)

    X_test = pd.DataFrame(np.random.randint(0,100,size=(100, 3)), columns=list('BCD'))

    y_pred = model_lr.predict(X_test)

    model_iir = train_iterative_imputer_regressor(X,Y)

    y_pred_iir = model_iir.predict(X_test)
    
    assert all([a == b for a, b in zip(y_pred, y_pred_iir)])