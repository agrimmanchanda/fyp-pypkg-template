# Libraries generic
import pandas as pd
import numpy as np
from scipy.stats import norm

# Libraries pgmpy
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, BicScore

# Libraries sklearn
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

class BNImputer(BayesianModel, BaseEstimator, TransformerMixin):
    
    def __init__(self, panel, edges=[]):
        """Initialises the panel and any passed edges.

        Args:
            panel (list of str): Laboratory panel of interest.
            edges (list of tuples, optional): Any edges passed as list of tuples. Defaults to [].
        """
        
        super().__init__(edges)

        self.obs_vars = []

        self.panel = panel
        
    def _learn_model_structure(self, X,y=None):
        """Learn the model structure using Hill Climb Algorithm. 
        Use the HillClimbSearch to find the best structure of the graph and use a Bayesian Information Criterion (BIC) 
        to set a score for the optimisation problem being solved by HillClimbSearch. 
        For convenience, code has been commented as it takes a couple of minutes (with high CPU requirement) 
        to find the edges but the edges found are consistent for BIC and BayesianDirichlet Equivalent Uniform (BDeu).

        Args:
            X (DataFrame): Training data as a DataFrame.
            y (Series, optional): Target data. Defaults to None.
        """
        
        # Combine data together if there is a Series.
        if y is not None:
            data = pd.concat([X,y], axis=1)
        else:
            data = X

        # Perform HillClimbSearch using BIC method.
        hc = HillClimbSearch(data, scoring_method=BicScore(data))

        # Estimate the best model.
        best_model = hc.estimate()
        
        # Initialise the Bayesian Model using the found edges.
        bn = BayesianModel(best_model.edges)
        
        super().__init__(bn.edges)
    
    def _param_learn(self, X, y=None, **kwargs):
        """Perform parameter learning using Bayesian statistics method.

        Args:
            X (DataFrame): Training data as a DataFrame.
            y (Series, optional): Target data. Defaults to None.
        """
        
        from pgmpy.estimators import BayesianEstimator
        
        estimator = BayesianEstimator

        # Concatenate training data with target data
        if y is not None:
            data = pd.concat([X,y], axis=1)
        else:
            data = X

        # Bayesian Dirichlet equivalent uniform (BDeu)
        prior_type = 'BDeu'
        
        # Set data in right format for learning
        data = data[self.panel]
        
        # Pass to BayesianModel parent class for learning
        super().fit(data, estimator, prior_type, **kwargs)
        
    def _impute_values(self, row):
        """Utility function to impute missing values based on provided evidence.

        Args:
            row (Dict): Dictionary containing values from each record.

        Returns:
            pred_data (DataFrame): Returns a complete data set with no missing values.
        """

        # Initialise predicted data
        pred_data = pd.DataFrame()

        # Check if any nan values
        is_nan = row.isnull().values.any()

        if is_nan:
            # Find all observed variables
            obs_vars = row.dropna().to_frame().T
            # Keep copy
            aux = obs_vars.copy()
            # Predict values for all missing values
            ypred = self.predict(obs_vars)
            # Concat with all observed variables
            ret = pd.concat([aux.reset_index(drop=True), ypred.reset_index(drop=True)], axis=1)
            # Join with DataFrame to return
            pred_data = pred_data.append(ret, ignore_index=True)
        else:
            pred_data = pred_data.append(row, ignore_index=True)

        # Check that all data is included
        unique_edges = self._get_unique_edges()

        # Order return data in the order of input data
        pred_data = pred_data[unique_edges]

        # Return
        return pred_data
    
    def transform(self, X, y=None):
        """Transformer which actually imputes missing data

        Args:
            X (DataFrame): Training data as a DataFrame.
            y (Series, optional): Target data. Defaults to None.

        Returns:
            data (array-like): Returns the data set with imputed values.
        """
        
        # Check type of input data and change to DataFrame as that is supported
        # format for pgmpy functions
        if isinstance(X, np.ndarray):
            
            cols = len(self.obs_vars)
            
            X = X.reshape(len(X), cols)

            X = pd.DataFrame(X, columns=self.obs_vars)
    
        # Check for any NaN values in the data set 
        nan_check = X.isnull().values.any()
        

        if nan_check:
            
            # Library to perform parallel execution
            from joblib import Parallel, delayed

            # DataFrame to store missing values
            pred_values = pd.DataFrame()

            # For each row pass it to _impute_values function
            pred_values = pred_values.append(
                        Parallel(n_jobs=-1)
                            (delayed(
                                self._impute_values)(r) 
                                for _,r in X.iterrows()
                            ), 
                            ignore_index=True
                        )
            # Return imputed DataFrame as a numpy to satisfy Scikit-learn API.
            return pred_values.to_numpy()
    
        else:
            
            return X.to_numpy()
            
    def fit(self, X, y=None):
        """Fit the data as necessary

        Args:
            X (DataFrame): Training data as a DataFrame.
            y (Series, optional): Target data. Defaults to None.

        Returns:
            self: Returns self object.
        """
        
        # Initialise observed values to panel
        self.obs_vars = self.panel
        
        # Extract the name of the target variable 
        if y is not None:
            self.col_drop = y.name
            self.obs_vars = set(self.panel) - set([y.name])
            y = y.reset_index(drop=True)
        
        # Convert input data into DataFrame if no value present.
        if isinstance(X, np.ndarray):
            
            cols = len(self.obs_vars)
            
            X = X.reshape(len(X), cols)

            X = pd.DataFrame(X, columns=self.obs_vars)
        
        # If edges are already provided then just perform parameter
        # learning else perform structure + parameter learning.
        if len(self.edges) > 0 and len(self.nodes) > 0:
            self._param_learn(X,y)
        else:
            self._learn_model_structure(X,y)
            self._param_learn(X,y)
        
        return self

    def get_params(self, deep=True):
        """ Returns the edges and used to satisfy Scikit-learn API.

        Args:
            deep (bool, optional): Scikit-learn API requirements. Defaults to True.

        Returns: 
            params (Dict): Returns the initialised parameters as Dict.
        """

        return{
            "edges": self.edges
        }

    def set_params(self, **parameters):
        """ Set the relevant parameters in the class.

        Returns:
            self: Return self object.
        """

        for parameter, value in parameters:

            setattr(self, parameter, value)

        return self

    def model_to_adjmat(self):
        """ Returns the Adjacency Matrix for the Bayesian Network.

        Returns:
            matrix (DataFrame): Return the Adjacency Matrix as boolean.
        """

        # Find nodes and edges
        nodes, edges = self.nodes,self.edges

        # Initialise the array
        zeros = np.zeros((len(nodes), len(nodes)))

        # Initialise adjacency matrix
        adjmat = pd.DataFrame(data=zeros, index=nodes, columns=nodes)

        # If an edge exist then make that entry in DataFrame 1.
        for edge in edges:
            adjmat.loc[edge] = 1

        # Return
        return adjmat.astype('bool')

    def check_model_prob(self):
        """ Function checks if all probabilities in CPD add to 1.

        Returns:
            bool: True if all probabilities add to 1 else returns False.
        """
        
        return super().check_model()

    def _get_unique_edges(self):
        """Function to get unique variables in model edges

        Returns:
            edges (list of str): Returns the edges in sorted order.
        """
        
        model_edges = self.edges

        # create a list of edge1 and edge2
        edge1, edge2 = [x[0] for x in model_edges], [x[1] for x in model_edges]

        # concatenate the list and use a set to get unique elements
        uniq_edges = list(set(edge1+edge2))

        # return the edges as a (alphabetically sorted) list
        return sorted(uniq_edges)


class BNRegressor(BayesianModel, BaseEstimator, RegressorMixin):
    
    def __init__(self, panel, edges=[]):
        """Initialises the panel and any passed edges.

        Args:
            panel (list of str): Laboratory panel of interest.
            edges (list of tuples, optional): Any edges passed as list of tuples. Defaults to [].
        """
        
        super().__init__(edges)

        self.obs_vars = []

        self.panel = panel
        
    def _learn_model_structure(self, X,y=None):
        """Learn the model structure using Hill Climb Algorithm. 
        Use the HillClimbSearch to find the best structure of the graph and use a Bayesian Information Criterion (BIC) 
        to set a score for the optimisation problem being solved by HillClimbSearch. 
        For convenience, code has been commented as it takes a couple of minutes (with high CPU requirement) 
        to find the edges but the edges found are consistent for BIC and BayesianDirichlet Equivalent Uniform (BDeu).

        Args:
            X (DataFrame): Training data as a DataFrame.
            y (Series, optional): Target data. Defaults to None.
        """
        
        # Combine data together if there is a Series.
        if y is not None:
            data = pd.concat([X,y], axis=1)
        else:
            data = X

        # Perform HillClimbSearch using BIC method.
        hc = HillClimbSearch(data, scoring_method=BicScore(data))

        # Estimate the best model.
        best_model = hc.estimate()
        
        # Initialise the Bayesian Model using the found edges.
        bn = BayesianModel(best_model.edges)
        
        super().__init__(bn.edges)
    
    def _param_learn(self, X, y=None, **kwargs):
        """Perform parameter learning using Bayesian statistics method.

        Args:
            X (DataFrame): Training data as a DataFrame.
            y (Series, optional): Target data. Defaults to None.
        """
        
        from pgmpy.estimators import BayesianEstimator
        
        estimator = BayesianEstimator

        # Concatenate training data with target data
        if y is not None:
            data = pd.concat([X,y], axis=1)
        else:
            data = X

        # Bayesian Dirichlet equivalent uniform (BDeu)
        prior_type = 'BDeu'
        
        # Set data in right format for learning
        data = data[self.panel]
        
        # Pass to BayesianModel parent class for learning
        super().fit(data, estimator, prior_type, **kwargs)
    
    def predict(self, X):
        """Returns the predicted value as per Scikit-learn API.

        Args:
            X (DataFrame): Test data without the predictor variable.

        Returns:
            prediction (array-like): Returns the 1D Numpy array.
        """
        
        # Make the input array into DataFrame
        if isinstance(X, np.ndarray):
            
            cols = len(self.obs_vars)
            
            X = X.reshape(len(X), cols)

            X = pd.DataFrame(X, columns=self.obs_vars)

        # Copy the input data
        aux = X.copy()

        # Collect the values for predicted variable
        ypred = super().predict(X)
        
        # Return as 1D array.
        return ypred.values.reshape(-1)
    
    def fit(self, X, y=None):
        """Fit the data as necessary

        Args:
            X (DataFrame): Training data as a DataFrame.
            y (Series, optional): Target data. Defaults to None.

        Returns:
            self: Returns self object.
        """
        
        # Initialise observed values to panel
        self.obs_vars = self.panel
        
        # Extract the name of the target variable 
        if y is not None:
            self.col_drop = y.name
            self.obs_vars = set(self.panel) - set([y.name])
            y = y.reset_index(drop=True)
        
        # Convert input data into DataFrame if no value present.
        if isinstance(X, np.ndarray):
            
            cols = len(self.obs_vars)
            
            X = X.reshape(len(X), cols)

            X = pd.DataFrame(X, columns=self.obs_vars)
        
        # If edges are already provided then just perform parameter
        # learning else perform structure + parameter learning.
        if len(self.edges) > 0 and len(self.nodes) > 0:
            self._param_learn(X,y)
        else:
            self._learn_model_structure(X,y)
            self._param_learn(X,y)
        
        return self

    def get_params(self, deep=True):
        """ Returns the edges and used to satisfy Scikit-learn API.

        Args:
            deep (bool, optional): Scikit-learn API requirements. Defaults to True.

        Returns: 
            params (Dict): Returns the initialised parameters as Dict.
        """

        return{
            "edges": self.edges
        }

    def set_params(self, **parameters):
        """ Set the relevant parameters in the class.

        Returns:
            self: Return self object.
        """

        for parameter, value in parameters:

            setattr(self, parameter, value)

        return self

    def model_to_adjmat(self):
        """ Returns the Adjacency Matrix for the Bayesian Network.

        Returns:
            matrix (DataFrame): Return the Adjacency Matrix as boolean.
        """

        # Find nodes and edges
        nodes, edges = self.nodes,self.edges

        # Initialise the array
        zeros = np.zeros((len(nodes), len(nodes)))

        # Initialise adjacency matrix
        adjmat = pd.DataFrame(data=zeros, index=nodes, columns=nodes)

        # If an edge exist then make that entry in DataFrame 1.
        for edge in edges:
            adjmat.loc[edge] = 1

        # Return
        return adjmat.astype('bool')

    def check_model_prob(self):
        """ Function checks if all probabilities in CPD add to 1.

        Returns:
            bool: True if all probabilities add to 1 else returns False.
        """
        
        return super().check_model()

    def _get_unique_edges(self):
        """Function to get unique variables in model edges

        Returns:
            edges (list of str): Returns the edges in sorted order.
        """
        
        model_edges = self.edges

        # create a list of edge1 and edge2
        edge1, edge2 = [x[0] for x in model_edges], [x[1] for x in model_edges]

        # concatenate the list and use a set to get unique elements
        uniq_edges = list(set(edge1+edge2))

        # return the edges as a (alphabetically sorted) list
        return sorted(uniq_edges)



class EMImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, max_iter = 50, epsilon = 0.01):
        """Initialises the imputer

        Args:
            max_iter (int, optional): Maximum number of iterations. Defaults to 50.
            epsilon (float, optional): Tolerance level for percentage change. Defaults to 0.01.
        """
        super().__init__()
        self.max_iter = max_iter
        self.epsilon = epsilon

    def fit(self, X, y=None):
        """Fit the imputer.

        Args:
            X (DataFrame): Training data as a DataFrame.
            y (Series, optional): Target data. Defaults to None.

        Returns:
            self: Returns self object.
        """

        return self
    
    def _EM_imputer(self, X, y=None):

        # Find all nan entries in array
        nan_idx = np.argwhere(np.isnan(X))
        
        for x, y in nan_idx:
            
            # Find all values for that feature
            feat = X[:, y]
            
            # Step 1: Initialisation
            mu, std = np.nanmean(feat), np.nanstd(feat)
            
            # Update the value for that feature
            X[x, y] = np.random.normal(loc=mu, scale=std)
            
            check_conv = False
            
            iter_no = 1

            last_vals = []

            while iter_no <= self.max_iter and not check_conv:
                
                # Step 2: Expectation step
                mu, std = np.nanmean(feat), np.nanstd(feat)
                
                # Step 3: Maximisation step
                X[x, y] = np.random.normal(loc=mu, scale=std)

                last_vals.append(X[x, y])

                # Step 4: Convergence check
                relative_change = pd.Series(last_vals).pct_change().iloc[-1]

                # Check convergence condition
                check_conv = np.abs(relative_change) < self.epsilon and not np.isnan(relative_change)

                # Increment
                iter_no += 1

        # Return the imputed array values    
        return X

    def transform(self, X, y=None):
        """Transforms the data and imputes missing values.

        Args:
            X (DataFrame): Training data as a DataFrame.
            y (Series, optional): Target data. Defaults to None.

        Returns:
            data (array-like): Returns the data set with imputed values.
        """
        
        # Conver to numpy is not array
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        
        # Check if any missing values
        nan_check = np.isnan(X).any()
        
        if nan_check:
            return self._EM_imputer(X)
        else:
            return X


def em_prefill(X, epsilon=0.01, max_iter=50):
    """Function to prefill values using the EM algorith.

    Args:
        X (array-like data): Input (incomplete) data that needs to be imputed.
        epsilon (float, optional): The tolerance level to which changes can occur. Defaults to 0.01.
        max_iter (int, optional): Number of maximum iterations per impute variable. Defaults to 50.

    Returns:
        data (array-like): Imputed data set that contains no missing values.
    """
        
    # Find all nan entries in array
    nan_idx = np.argwhere(np.isnan(X))
    
    for x, y in nan_idx:
        
        # Find all values for that feature
        feat = X[:, y]
        
        # Step 1: Initialisation
        mu, std = np.nanmean(feat), np.nanstd(feat)
        
        # Update the value for that feature
        X[x, y] = np.random.normal(loc=mu, scale=std)
        
        check_conv = False
        
        iter_no = 1

        last_vals = []

        while iter_no <= max_iter and not check_conv:
            
            # Step 2: Expectation step
            mu, std = np.nanmean(feat), np.nanstd(feat)
            
            # Step 3: Maximisation step
            X[x, y] = np.random.normal(loc=mu, scale=std)

            last_vals.append(X[x, y])

            # Step 4: Convergence check
            relative_change = pd.Series(last_vals).pct_change().iloc[-1]

            # Check convergence condition
            check_conv = np.abs(relative_change) < epsilon and not np.isnan(relative_change)

            # Increment
            iter_no += 1

    # Return the imputed array values    
    return X