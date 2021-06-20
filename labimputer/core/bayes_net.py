# Libraries generic
import pandas as pd
import numpy as np

# Libraries pgmpy
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, BicScore

# Libraries sklearn
from sklearn.base import BaseEstimator, RegressorMixin

def learn_model_structure(data):

    hc = HillClimbSearch(data, scoring_method=BicScore(data))
    
    best_model = hc.estimate()

    return best_model

class BNImputer(BayesianModel):
    
    def fit(self, data, **kwargs):
        
        from pgmpy.estimators import BayesianEstimator
        
        estimator = BayesianEstimator
        
        super().fit(data, estimator, **kwargs)
        
    
    def get_model_adjmat(self):

        nodes, edges = self.nodes,self.edges

        zeros = np.zeros((len(nodes), len(nodes)))

        adjmat = pd.DataFrame(data=zeros, index=nodes, columns=nodes)

        for edge in edges:
            adjmat.loc[edge] = 1

        return adjmat.astype('bool')
    
    def get_model_cdf(self):
        
        return super().check_model()
    
    
    # Function to get unique variables in model edges
    def _get_unique_edges(self):
        
        model_edges = self.edges

        # create a list of edge1 and edge2
        edge1, edge2 = [x[0] for x in model_edges], [x[1] for x in model_edges]

        # concatenate the list and use a set to get unique elements
        uniq_edges = list(set(edge1+edge2))

        # return the edges as a (alphabetically sorted) list
        return sorted(uniq_edges)
        
     
    def impute(self, df):
        
        data = pd.DataFrame()

        for idx, row in df.iterrows():

            is_nan = row.isnull().values.any()

            if is_nan:
                obs_vars = row.dropna().to_frame().T
                aux = obs_vars.copy()
                ypred = self.predict(obs_vars)
                ret = pd.concat([aux.reset_index(drop=True), ypred.reset_index(drop=True)], axis=1)
                data = data.append(ret, ignore_index=True)
            else:
                data = data.append(row, ignore_index=True)
                print(f"{idx}, no nan found")

        unique_edges = self._get_unique_edges()

        data = data[unique_edges]

        return data

    def _predict_values(self, row):

        pred_data = pd.DataFrame()

        is_nan = row.isnull().values.any()

        if is_nan:
            obs_vars = row.dropna().to_frame().T
            aux = obs_vars.copy()
            ypred = self.predict(obs_vars)
            ret = pd.concat([aux.reset_index(drop=True), ypred.reset_index(drop=True)], axis=1)
            pred_data = pred_data.append(ret, ignore_index=True)
        else:
            pred_data = pred_data.append(row, ignore_index=True)

        unique_edges = self._get_unique_edges()
        
        pred_data = pred_data[unique_edges]

        return pred_data


    def imputer(self, df):

        from joblib import Parallel, delayed

        pred_values = pd.DataFrame()

        pred_values = pred_values.append(
                    Parallel(n_jobs=-1)
                        (delayed(
                            self._predict_values)(r) 
                            for _,r in df.iterrows()
                        ), 
                        ignore_index=True
                    )

        return pred_values


class BNRegressor(BayesianModel, BaseEstimator, RegressorMixin):
    
    def __init__(self, PANEL):
        
        self.PANEL = PANEL
        
        super().__init__(self)
        
    def _learn_model_structure(self, X,y=None):
        
        if y is not None:
            print(f'lstr {type(X)}')
            data = pd.concat([X,y], axis=1)
        else:
            data = X

        hc = HillClimbSearch(data, scoring_method=BicScore(data))

        best_model = hc.estimate()
        
        bn = BayesianModel(best_model.edges)
        
        super().__init__(bn.edges)
    
    def _fit_model(self,
           X, y=None,
           **kwargs):
        
        print("Entering _fit_model...")
        
        from pgmpy.estimators import BayesianEstimator
        
        estimator = BayesianEstimator
        
        if y is not None:
            data = pd.concat([X,y], axis=1)
        else:
            data = X
        
        data = data[self.PANEL]
        
        print(f"\n {data}")
        
        super().fit(data, estimator, **kwargs)
    
    def predict(self, X):
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.obs_vars)
        
        aux = X.copy()
        
        ypred = super().predict(X)
        
        return ypred.values
    
    def fit(self, X, y=None):
        
        self.obs_vars = []
        
        if y is not None:
            self.col_drop = y.name
            self.obs_vars = set(self.PANEL) - set([y.name])
            y = y.reset_index(drop=True)
        
        if isinstance(X, np.ndarray):
            
            X = pd.DataFrame(X, columns=self.obs_vars)
        
        if len(self.edges) > 0 and len(self.nodes) > 0:
            self._fit_model(X,y)
        else:
            self._learn_model_structure(X,y)
            self._fit_model(X,y)
        
        return self