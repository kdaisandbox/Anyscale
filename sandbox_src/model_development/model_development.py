import pandas as pd
from darts.models import (
    AutoARIMA,
    LightGBMModel
)
import numpy as np


class ModelDevelopment:
    
    def __init__(self, config):
        self.PREDICTION_HORIZON = config['model_development']['prediction_horizon']
        self.PREDICTION_FREQUENCY = config['model_development']['prophet']['prediction_frequency']
        self.PREDICTION_LGBM = config['model_development']['lgbm']['prediction_horizon']
        self.AUTOREG_LAG_30DAYS_TRAINING = config['model_development']['lgbm']['autoreg_lag_30days_training']
        self.AUTOREG_LAG_14DAYS_TRAINING = config['model_development']['lgbm']['autoreg_lag_14days_training']
        self.START_P = config['model_development']['autoarima']['START_P']
        self.START_Q = config['model_development']['autoarima']['START_Q']
        self.MAX_P = config['model_development']['autoarima']['MAX_P']
        self.MAX_Q = config['model_development']['autoarima']['MAX_Q']
        self.PREDICTION_FREQUENCY_ARIMA = config['model_development']['autoarima']['prediction_frequency']
        

  
    def predict_arima(self, train):
        """
        Makes 30 days of prediction using Auto-ARIMA
        :param train: A darts series representing training timeseries data
        :return prediction_autoarima: A darts series representing Auto-ARIMA prediction
        """
        
        #Initalize Auto-ARIMA Model
        autoarima_model = AutoARIMA(start_p = self.START_P, start_q = self.START_Q,
                                    max_p = self.MAX_P, max_q = self.MAX_Q, frequency=self.PREDICTION_FREQUENCY_ARIMA, error_action="ignore")
        #Train the model
        autoarima_model.fit(series = train)
        #Predict disk storage for the upcoming 30 days
        prediction_autoarima = autoarima_model.predict(n = self.PREDICTION_HORIZON)
        
        return prediction_autoarima
    
    
    def predict_lgbm_30(self, train):
        """
        Makes 1 day of prediction using LGBM. Retrains the model with shifted data after each prediction for 29 times more to make 30 days prediction. Lag = 15
        :param train: A darts series representing training timeseries data
        :return predictions: A darts series representing LGBM prediction
        """
        #Initalize Autoregressive LightGBM Model with 15 days lag
        lgbm_model = LightGBMModel(lags = self.AUTOREG_LAG_30DAYS_TRAINING)
        #Train the model
        lgbm_model.fit(series = train)
        #Predict disk storage for the upcoming 30 days
        prediction_lgbm = lgbm_model.predict(n = self.PREDICTION_LGBM) 
        #Create prediction timeseries
        predictions = prediction_lgbm
        #Shift time series with the prediction
        train1 = train.append(prediction_lgbm)[1:]
        
        #Repeat process for 30 days prediction
        for c in range(29):
            lgbm_model = LightGBMModel(lags = self.AUTOREG_LAG_30DAYS_TRAINING)
            lgbm_model.fit(series = train1)
            prediction_lgbm = lgbm_model.predict(n = self.PREDICTION_LGBM)
            #Append new predictions to timeseries
            predictions = predictions.append(prediction_lgbm)
            train1 = train1.append(prediction_lgbm)[1:]                          
    
        return predictions
    
    
    def predict_lgbm_14(self, train):
        """
        Makes 1 day of prediction using LGBM. Retrains the model with shifted data after each prediction for 29 times more to make 30 days prediction. Lag = 7
        :param train: A darts series representing training timeseries data
        :return predictions: A darts series representing LGBM prediction
        """
        #Initalize Autoregressive LightGBM Model with 7 days lag
        lgbm_model = LightGBMModel(lags = self.AUTOREG_LAG_14DAYS_TRAINING)
        #Train the model
        lgbm_model.fit(series = train)
        #Predict disk storage for the upcoming 30 days
        prediction_lgbm = lgbm_model.predict(n = self.PREDICTION_LGBM) 
        #Create prediction timeseries
        predictions = prediction_lgbm
        #Shift time series with the prediction
        train1 = train.append(prediction_lgbm)[1:]
        
        #Repeat process for 30 days prediction
        for c in range(29):
            lgbm_model = LightGBMModel(lags = self.AUTOREG_LAG_14DAYS_TRAINING)
            lgbm_model.fit(series = train1)
            prediction_lgbm = lgbm_model.predict(n = self.PREDICTION_LGBM)
            #Append new predictions to timeseries
            predictions = predictions.append(prediction_lgbm)
            train1 = train1.append(prediction_lgbm)[1:]
                                    
        return predictions

