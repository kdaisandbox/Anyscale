from fbprophet import Prophet
import pandas as pd
from darts.models import (
    AutoARIMA,
    LightGBMModel
)
import numpy as np
#from unidecode import unidecode
from sktime.forecasting.compose import make_reduction, TransformedTargetForecaster
from sktime.forecasting.model_selection import ExpandingWindowSplitter, ForecastingGridSearchCV
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.detrend import Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.series.detrend import Deseasonalizer

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
        

    def check_downtrend(self, train_set):
        X = (np.arange(30)+1).reshape(-1, 1)
        y = train_set
        lin_reg = LinearRegression()
        lin_reg.fit(X,y)
        pred = lin_reg.predict(np.array([31,32]).reshape(-1, 1))
       
        return pred[0] > pred[-1]
    

    def create_forecaster_w_detrender_deseasonalizer(self, sp, degree=1):
        # creating forecaster with LightGBM
        regressor = lgb.LGBMRegressor()
        forecaster = TransformedTargetForecaster(
            [   # ("deseasonalize", Deseasonalizer(model="additive", sp=sp)),
                ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=degree))),
                (
                    "forecast",
                    make_reduction(regressor, strategy="recursive"),
                ),
            ]
        )

        return forecaster


    def find_detrend_and_deseasonality(self, train_set, param_grid):
        cv = ExpandingWindowSplitter(initial_window=int(len(train_set) * 0.8))
        forecaster = self.create_forecaster_w_detrender_deseasonalizer(5, 5)
        gscv = ForecastingGridSearchCV(
            forecaster, strategy="refit", cv=cv, param_grid=param_grid
        )
        gscv.fit(train_set)
        return gscv.best_params_, gscv


    def predict_prophet(self, train):
        """
        Makes 30 days of prediction using Prophet
        :param train: A pandas dataframe representing training data in required Prophet input df
        :return prediction_prophet: A pandas dataframe representing Prophet prediction
        """
        
        #Initalize Prophet
        prophet_model = Prophet()
        #Train the model
        prophet_model.fit(train)
        #Create 30 days prediction
        future = prophet_model.make_future_dataframe(periods = self.PREDICTION_HORIZON, freq = self.PREDICTION_FREQUENCY, include_history=False)
        prediction_prophet = prophet_model.predict(future)[["ds", "yhat"]]
        
        #prophet_model.plot(prediction_prophet)
    
        return prediction_prophet
    
    
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

