import pandas as pd
import xgboost as xgb
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.model_selection import  GridSearchCV        
# import sys
# sys.path.append('src/')
#import numpy as np

"""
class that contains the forecaster
"""

class Forecaster:
    def __init__(self, name = 'forecaster', data_set = None, model = None):
        self.name = name            
        self.train_set = None       
        self.val_set = None
        self.test_set = None
        self.model = model          
        self.model_name = None      
        self.predictions = None
        self.mean_train = None
        self.sd_train = None
        self.best_mode_name = None
        self.y_column = None
        self.models_available = ['linear', 'ridge', 'lasso', 'xgboost','random_forest']
        self.metrics_available = ['rmse', 'mae', 'mape']
        self.data_set = data_set
        

    def preprocess(self, data_set):
        """
        function that drops the na values and duplicates and convert the index to datetime
        """
        data_set = data_set.dropna(axis=0, how='any')
        data_set = data_set.drop_duplicates()
        return data_set


    def ingest(self, data,  y_column = None):
        """
        function that ingests the data, set the index and y column
        param: data: dataframe
        param: index_column: column to set as index
        param: y_column: column to set as y column
        """
        self.data_set = data
        if y_column not in data.columns:
            raise ValueError('y_column not in data columns')
        else:
            self.y_column = y_column
        self.data_set = self.preprocess(self.data_set)
    
    
    def split_train_test(self, train_perc = 0.7, val_perc = 0.15):
        """
        function that splits the data_set variable into train, validation and test set
        param: train_perc: percentage of data to be used as train set
        param: val_perc: percentage of data to be used as validation set
        """
        #sort the data set by index
        self.data_set = self.data_set.sort_index()
        train_size = int(len(self.data_set) * train_perc)
        val_size = int(len(self.data_set) * val_perc)
        #split the data set into train, validation and test set
        self.train_set = self.data_set.iloc[:train_size].copy()
        self.val_set = self.data_set.iloc[train_size:train_size+val_size].copy()
        self.test_set = self.data_set.iloc[train_size+val_size:].copy()
        #calculate the mean and standard deviation of the train set for future normalization
        self.mean_train = self.train_set.mean()
        self.sd_train = self.train_set.std()

    
    def normalize(self, data_set):
        """
        function that normalizes the data set using the mean and standard deviation of the train set
        that for avoiding data leakage to the validation and test set
        param: data_set: dataframe to normalize
        return: normalized dataframe
        """
        
        means = self.mean_train
        sds = self.sd_train
        data_set = pd.DataFrame(index=data_set.index, data=data_set)
        #for each column in the data set, if the column is in the mean and standard deviation index, normalize the column
        for col in data_set.columns:
            if col in means.index:
                data_set[col] = (data_set[col] - means[col]) / sds[col]
            else:
                raise ValueError('Column not in means.index or sds.index')
        return data_set
        

    def denormalize(self, data_set):
        """
        function that denormalizes the data set using the mean and standard deviation of the train set
        that for avoiding data leakage to the validation and test set
        param: data_set: dataframe to denormalize
        return: denormalized dataframe
        """
        means = self.mean_train
        sds = self.sd_train
        for col in data_set.columns:
            if col in means.index:
                data_set[col] = data_set[col] * sds[col] + means[col]
            else:
                raise ValueError('Column not in means.index or sds.index')
        return data_set


    def set_parameters(self, parameters):
        """
        function that sets the parameters of the model
        """
        if self.model is None:
            raise ValueError('Model is None')
        else:
            self.model.set_params(**parameters)
            print(self.model.get_params())
            

    def train(self,  model, params = None):
        """
        function that set the model and train it using the given parameters
        param: model: model to train 
        param: params: parameters of the model
        """

        self.split_train_test()
        self.train_set = self.normalize(self.train_set)
        #set the model
        self.model = model
        self.y_column = self.y_column
        #set the x and y variables
        y = self.train_set[self.y_column]
        x = self.train_set.drop(self.y_column, axis=1)

        if model == 'linear':
            self.model = linear_model.LinearRegression()    
        elif model == 'ridge':
            self.model = linear_model.Ridge(alpha = 0.05)
        elif model == 'lasso':
            self.model = linear_model.Lasso(alpha=0.1)
        elif model == 'random_forest':
            self.model = RandomForestRegressor(random_state = 0)
        elif model == 'xgboost':
            self.model = xgb.XGBRegressor(random_state = 0)
        else:
            raise ValueError('Unknown model: ' + model)
        
        if params is not None:
            self.set_parameters(parameters= params)
    
        self.model_name = model
        self.model.fit(x, y)
        self.train_set = self.denormalize(self.train_set)
        return self.model
    

    def forecast(self, data_set):
        """
        function that makes the forecast using the model in the class and the given data
        param: x: data to make the forecast
        return: predictions
        """
        x = data_set.copy()
        #normalize the input data
        x = self.normalize(x)
        # make predictions
        self.predictions = self.model.predict(x)
        self.predictions = pd.DataFrame(index=x.index, data=self.predictions, columns=[self.y_column])
        #denormalize the predictions
        self.predictions = self.denormalize(self.predictions)
        return self.predictions

    
    def backtest(self, metric = 'rmse', set_type = 'test_set'):
        """
        function that runs the model on the testset and compare the prediction with the actuals
        return the chosen metric and the predictions
        """
        #set the x and y variables depending on the set_type
        if set_type == 'test_set':
            x =  self.test_set
        elif set_type == 'val_set':
            x = self.val_set
        elif set_type == 'train_set':
            x = self.train_set

        y = x[[self.y_column]]
        x = x.drop([self.y_column], axis=1)
        #make the forecast
        self.predictions = self.forecast(x)
        #calculate the metric
        metric_value = self.calculate_metric(self.predictions, y, metric)
        return metric_value, self.predictions
    

    def calculate_metric(self, predictions, actuals, metric= 'rmse'):
        """
        function that calculates the metric of the predictions vs the actuals
        param: predictions: predictions of the model
        param: actuals: actuals of the data
        param: metric: metric to calculate
        return: metric value
        """
        #set the metric to lower case
        metric = metric.lower()
        #check if the metric is available
        if metric in self.metrics_available:
            if metric == 'rmse':
                metric_value = ((predictions - actuals) ** 2).mean() ** 0.5    
            elif metric == 'mape':
                metric_value = abs(predictions - actuals) / actuals.mean()
            elif metric == 'mae':
                metric_value = abs(predictions - actuals).mean()
            else:
                raise ValueError('Unknown metric: ' + metric)
            metric_value = float(metric_value)
            metric_value = round(metric_value, 2)
        else:
            raise ValueError('Unknown metric: ' + metric)

        return metric_value

    
    def best_model(self, metric = 'rmse'):
        """
        function that runs all the available models in the class  with the default parameters and returns the best model and the metric value
        param: metric: metric to calculate
        return: best model and metric value
        """
        metric = metric.lower()
        
        #set best metric value infinity
        best_metric = float('inf')
        #set best model to None
        best_model = None

        #for each model in the models available, train the model, make the forecast and calculate the metric
        for model in self.models_available:
            self.train(model)
            actuals = self.test_set[[self.y_column]].copy()
            x = self.test_set.drop([self.y_column], axis=1)
            y = self.forecast(x)
            metric_value = self.calculate_metric(y, actuals, metric)
            #if the metric value is better than the best metric value, set the best metric value to the metric value and the best model to the model
            if metric_value < best_metric:
                best_metric = metric_value
                best_model = model
            print(f"{model}: {metric_value}")
        #set the model to the best model
        self.model_name = best_model
        self.model = self.train(best_model)
        return best_model, best_metric

    
    def plot_predictions(self, backtest = False, set_type = 'test_set'):
        """
        function that plots the predictions or/and the actuals
        param: backtest: boolean to plot the actuals
        """
        plt.figure(figsize = (15,6))
        #plot the predictions
        plt.plot(self.predictions, label='predictions')
        #if backtest is True, plot the actuals
        if backtest:
            if set_type == 'test_set':
                plt.plot(self.test_set[[self.y_column]], label='actuals')
            elif set_type == 'val_set':
                plt.plot(self.val_set[[self.y_column]], label='actuals')
            elif set_type == 'train_set':
                plt.plot(self.train_set[[self.y_column]], label='actuals')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Load (MW)')
        plt.grid(True)
        plt.tight_layout()
        plt.title(self.model_name)
        plt.show(block=True, )


    def get_model_params(self):
        """
        function that returns the parameters of the model
        return: parameters of the model
        """
        return self.model.get_params()
    
    
    def feature_selection_rfe(self, num_features = 5):
        """
        function that runs a feature selection through recursive feature elimination
        return: best metric value
        """
        rfe = RFE(self.model, num_features)
        rfe.fit(self.val_set.drop([self.y_column], axis=1), self.val_set[self.y_column])
        print(rfe.ranking_)
        return rfe.ranking_

    
    def hyper_parameter(self, model, params_dict):
        """
        function that runs a grid search for the given model and parameters in the validation set
        param: model: model to run the grid search
        param: params_dict: dictionary of parameters to run the grid search
        return: best metric value
        """
        #grid search
        tuned_model = GridSearchCV(self.model, params_dict, scoring='neg_mean_squared_error', cv = 5,  verbose = 1)
        #normalize the validation set
        self.val_set = self.normalize(self.val_set)
        #fit the model
        tuned_model.fit(self.val_set.drop([self.y_column], axis=1), self.val_set[self.y_column])
        self.model = tuned_model.best_estimator_
        #denormalize the validation set
        self.val_set = self.denormalize(self.val_set)
        #calculate the metric
        metric_val_set, preds_on_val_set = self.backtest()
        return metric_val_set