import pandas as pd

def add_heating_cooling_degrees(data_set, temperature_cols , keep_temperature = False, temperature_reference = 65):
    """
    function that adds the heating and cooling degrees to the data set
    param: temperature_cols: columns of the temperature
    param: keep_temperature: boolean to keep the temperature columns
    param: temperature_reference: reference temperature to calculate the heating and cooling degrees
    
    ## TIP: even when the formal formula states 65 degrees as the reference, in my experience that temperature can be tuned
    for improving the performance of the model. 
    
    ## TIP: Even when the forecaster class is defined as a general class, this method was created for the case of Caiso load 
    forecasting.

    """
    for temperature in temperature_cols:
        if temperature not in data_set.columns:
            raise ValueError('Temperature column not in data set')
        else:
            data_set['cooling_degrees_'+temperature] = -data_set[temperature].copy() + temperature_reference
            data_set['cooling_degrees_'+temperature] = data_set['cooling_degrees_'+temperature].clip(lower=0)
            data_set['heating_degrees_'+temperature] = data_set[temperature].copy() - temperature_reference
            data_set['heating_degrees_'+temperature] = data_set['heating_degrees_'+temperature].clip(lower=0)
            
            if not keep_temperature:
                data_set = data_set.drop([temperature], axis=1)
    return data_set


def add_weekday_hour(data_set):
    """
    function that adds the weekday-hour column to the data set from the index
    """
    data_set.index = pd.to_datetime(data_set.index)
    data_set['hour'] = data_set.index.hour
    data_set['weekday'] = data_set.index.weekday
    data_set['month'] = data_set.index.weekday
    data_set['weekday-hour'] = data_set['weekday'].astype(str) + '-' + data_set['hour'].astype(str)
    data_set = data_set.drop(['hour', 'weekday'], axis=1)
    return data_set
    
    
def create_dummy_variables(data_set, column):
    """
    function that creates dummy variables from a column
    param: column: column to create dummy variables
    return: data set with the dummy variables
    """
    dummies = pd.get_dummies(data_set[column],prefix = column+'_')
    data_set = pd.concat([data_set, dummies], axis=1)
    data_set = data_set.drop([column], axis=1)
    return data_set

def add_month(data_set):
    """
    function that adds the month column to the data set from the index
    """
    #data_set.index = pd.to_datetime(data_set.index)
    data_set['month_feature'] = data_set.index.month
    return data_set