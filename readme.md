The Forecaster Library

Welcome to the Forecaster Library! This library is designed to develop and evaluate forecasting models. It is extensible, so that you can add new models, features, and metrics. It is also designed to be easy to use, so that you can quickly train and evaluate models.

## Files

- **Forecaster.py**: Contains the Forecaster class, which is the main class of the library. It contains the methods to ingest, preprocess, train, backtest, forecast, evaluate, plot, select features and  hyperparameter models.
- **caiso_case.ipynb**: Contains an example of how to use the library to train, backtest and forecast a model, through an use case.
- **data/caiso_system_load.csv**: Contains the dataset used in the example.
- **src/feature_eng.py**: Contains functions for adding features to the dataset.

## Features

- **Ingest**: Ingest data from pandas dataframe
- **Preprocess**: Preprocess data, dropping rows with empty varlues
- **Train**: Train a model on the train set of the dataset
- **Backtest**: Backtest the model on the train set, dev set or test set (parameter)
- **Forecast**: Forecast the target variable 
- **Evaluate**: Evaluate the model using a metric (rmse, mae, mape)
- **Plot**: Plot the forecast and the actual values
- **Feature selection**: Select the best features using recursive feature elimination, RFE
- **Hyperparameter tuning**: Tune the hyperparameters of the model using grid search

## Assumptions

1. The data are timeseries with a datetime as the index
2. For simplicity rows with any empty value will be removed
3. Using rmse as the metric to evaluate the model
4. For simplicity train set was defined as the oldest portion of the dataset, validataion set as the second oldest portion of the dataset and test set as the most recent portion of the dataset.

## Forecasting models

- **Linear**: A linear regressor from scikit-learn
- **Ridge**: A ridge regressor from scikit-learn
- **Lasso**: A lasso regressor from scikit-learn
- **RandomForest**: A random forest regressor from scikit-learn
- **XGBoost**: A gradient boosting regressor from XGBoost

## Example of usage

"""read the data"""
caiso_data = pd.read_csv('data/caiso_system_load.csv')

"""obtain the data to predict and data to train"""
data_to_predict = caiso_data.loc[caiso_data['interval_start_time'] >= '2023-07-31 00:00:00']

"""obtain the dataset to train"""
caiso_dataset = caiso_data.loc[caiso_data['interval_start_time'] < '2023-07-31 00:00:00']

"""create a forecaster object with the dataset"""
caiso_case = Forecaster(name = 'caiso_forecaster',data_set=caiso_dataset)

"""remove empty rows, duplicates"""
caiso_case.preprocess()

"""train a linear model"""
caiso_case.train(method = 'linear')

"""backtest the model and obtain the rmse"""
rsme, preds= caiso_case.backtest()
print('The rmse is: ', rsme)

"""forecast using data_to_predict as input"""
caiso_case.forecast(data_to_predict)

## Libraries used

- Pandas 1.1.3
- Xgboost 1.2.1
- Scikit-learn 0.23.2ks
- Matplotlib 3.3.2
- seaborn 0.11.0

## Future work
- Add Tranformers and LSTM models
- Add holidays as a feature

## Acknowledgements

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)

## 🚀 About Me

I'm Luis Vizcaya, Machine Learning engineer, AI entusiastic, and a lifelong learner.