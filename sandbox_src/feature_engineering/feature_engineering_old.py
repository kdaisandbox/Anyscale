import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.metrics.metrics import mae, mse
from darts.models import (
    AutoARIMA,
    LightGBMModel,
)
from fbprophet import Prophet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

class ClusterFeatureEngine:

    def __init__(self, config):
        self.config = config
        
    # Helper function to create outut df for clustering
    def create_model_error_df(self, diskname, lgbm_7_model_mae_mean, lgbm_7_model_mae_std, prophet_7_model_mae_mean, prophet_7_model_mae_std, lgbm_30_model_mae_mean, lgbm_30_model_mae_std, prophet_30_model_mae_mean, prophet_30_model_mae_std, arima_30_model_mae_mean, arima_30_model_mae_std):
        return pd.DataFrame({'diskname':[diskname], 'lgbm_7_model_mae_mean':[lgbm_7_model_mae_mean], 'lgbm_7_model_mae_std':[lgbm_7_model_mae_std], 'prophet_7_model_mae_mean':[prophet_7_model_mae_mean], 'prophet_7_model_mae_std':[prophet_7_model_mae_std], 'lgbm_30_model_mae_mean':[lgbm_30_model_mae_mean], 'lgbm_30_model_mae_std':[lgbm_30_model_mae_std], 'prophet_30_model_mae_mean':[prophet_30_model_mae_mean], 'prophet_30_model_mae_std':[prophet_30_model_mae_std], 'arima_30_model_mae_mean':[arima_30_model_mae_mean], 'arima_30_model_mae_std':[arima_30_model_mae_std]})
        
    def cluster_disks(self, df):
        
        NUMBER_OF_DIFFERENT_WINDOW_SIZES = 2
        data = []
        
        for i in range(len(df.columns)): 
            diskname = df.columns[i]
            disk_series = df.iloc[:, i]
            
            

            if len(disk_series) > 0:
            
                disk_series = disk_series.round(decimals=5)
                disk_series = disk_series.iloc[-75:] 

                # Iterate over day lags (last 7 or 30days)
                for j in range(NUMBER_OF_DIFFERENT_WINDOW_SIZES):


                    # If the length of the number of training days = 7
                    if j == 0:
                        # Create sliding windows for train and test
                        length = len(disk_series)
                        window_size = 15

                        lgbm_7_mae_list = []
                        prophet_7_mae_list = []

                        for k in range(window_size, length - 30):
                            train = TimeSeries.from_series(disk_series.iloc[k - window_size:k-1])
                            test = TimeSeries.from_series(disk_series.iloc[k:k + 30])
                            prophet_train = pd.DataFrame(data={'ds': disk_series.iloc[k - window_size + 7:k - 1].index,
                                                               'y': disk_series.iloc[k - window_size + 7:k - 1].values})

                            prophet_test = disk_series.iloc[k:k + 30]

                            ####MODEL TRAINING####

                            # Initalize Autoregressive LightGBM Model with 7 days lag
                            lgbm_model = LightGBMModel(lags=[-7, -6, -5, -4, -3, -2, -1])
                            # Train the model
                            lgbm_model.fit(series=train)
                            # Predict disk storage for the upcoming 30 days
                            prediction_lgbm = lgbm_model.predict(n=1)
                            # Create prediction timeseries
                            prediction_lgbm_7 = prediction_lgbm
                            # Shift time series with the prediction
                            train1 = train.append(prediction_lgbm)[1:]

                            # Repeat process for 30 days prediction
                            for c in range(29):
                                lgbm_model = LightGBMModel(lags=[-7, -6, -5, -4, -3, -2, -1])
                                lgbm_model.fit(series=train1)
                                prediction_lgbm = lgbm_model.predict(n=1)
                                # Append new predictions to timeseries
                                prediction_lgbm_7 = prediction_lgbm_7.append(prediction_lgbm)
                                train1 = train1.append(prediction_lgbm)[1:]

                            # Evaluate model using MAE for the sliding windows
                            lgbm_mae = mae(actual_series=test, pred_series=prediction_lgbm_7[1:])
                            # print('MAE: %f' % lgbm_mae)
                            lgbm_7_mae_list.append(lgbm_mae)

                            # Facebook Prophet Model
                            prophet_model = Prophet()
                            prophet_model.fit(prophet_train)
                            future = prophet_model.make_future_dataframe(periods=31, freq="d", include_history=False)
                            prediction_prophet_7 = prophet_model.predict(future)
                            # Evaluate model using MAE for the sliding windows
                            prophet_mae = mean_absolute_error(prophet_test, prediction_prophet_7.iloc[1:, 1])
                            # print('MAE: %f' % prophet_mae)
                            prophet_7_mae_list.append(prophet_mae)

                        lgbm_7_model_mae_mean = np.mean(lgbm_7_mae_list)
                        lgbm_7_model_mae_std = np.std(lgbm_7_mae_list)
                        prophet_7_model_mae_mean = np.mean(prophet_7_mae_list)
                        prophet_7_model_mae_std = np.std(prophet_7_mae_list)

                    # If length of training days = 30
                    elif j == 1:


                        # Create sliding windows for train and test
                        length = len(disk_series)
                        window_size = 31

                        lgbm_30_mae_list = []
                        autoarima_30_mae_list = []
                        prophet_30_mae_list = []

                        for k in range(window_size, length - 30):
                            train = TimeSeries.from_series(disk_series.iloc[k - window_size:k - 1])
                            test = TimeSeries.from_series(disk_series.iloc[k:k + 30])
                            prophet_train = pd.DataFrame(data={'ds': disk_series.iloc[k - window_size:k - 1].index,
                                       'y': disk_series.iloc[k - window_size:k - 1].values})
                            prophet_test = disk_series.iloc[k:k + 30]

                            ####MODEL TRAINING####
                            # Autoregressive LightGBM Model
                            # Initalize Autoregressive LightGBM Model with 15 days lag
                            lgbm_model = LightGBMModel(lags=[-15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1])
                            # Train the model
                            lgbm_model.fit(series=train)
                            # Predict disk storage for the upcoming 30 days
                            prediction_lgbm = lgbm_model.predict(n=1)
                            # Create prediction timeseries
                            prediction_lgbm_30 = prediction_lgbm
                            # Shift time series with the prediction
                            train1 = train.append(prediction_lgbm)[1:]

                            # Repeat process for 30 days prediction
                            for c in range(29):
                                lgbm_model = LightGBMModel(lags=[-15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1])
                                lgbm_model.fit(series=train1)
                                prediction_lgbm = lgbm_model.predict(n=1)
                                # Append new predictions to timeseries
                                prediction_lgbm_30 = prediction_lgbm_30.append(prediction_lgbm)
                                train1 = train1.append(prediction_lgbm)[1:]

                            # Evaluate model using MAE for the sliding windows
                            lgbm_mae = mae(actual_series=test, pred_series=prediction_lgbm_30[1:])
                            # print('MAE: %f' % lgbm_mae)
                            lgbm_30_mae_list.append(lgbm_mae)

                            # Facebook Prophet Model
                            prophet_model = Prophet()
                            prophet_model.fit(prophet_train)
                            future = prophet_model.make_future_dataframe(periods=31, freq="d", include_history=False)
                            prediction_prophet_30 = prophet_model.predict(future)
                            # Evaluate model using MAE for the sliding windows
                            prophet_mae = mean_absolute_error(prophet_test, prediction_prophet_30.iloc[1:, 1])
                            # print('MAE: %f' % prophet_mae)
                            prophet_30_mae_list.append(prophet_mae)

                            # Auto-ARIMA Model
                            autoarima_model = AutoARIMA(start_p=1, start_q=1, max_p=7, max_q=7)
                            autoarima_model.fit(series=train)
                            prediction_autoarima_30 = autoarima_model.predict(n=30)
                            # Evaluate model using MAE for the sliding windows
                            autoarima_30_mae = mae(actual_series=test, pred_series=prediction_autoarima_30[1:])
                            # print('MAE: %f' % disk_mae)
                            autoarima_30_mae_list.append(autoarima_30_mae)

                        lgbm_30_model_mae_mean = np.mean(lgbm_30_mae_list)
                        lgbm_30_model_mae_std = np.std(lgbm_30_mae_list)
                        prophet_30_model_mae_mean = np.mean(prophet_30_mae_list)
                        prophet_30_model_mae_std = np.std(prophet_30_mae_list)
                        arima_30_model_mae_mean = np.mean(autoarima_30_mae_list)
                        arima_30_model_mae_std = np.std(autoarima_30_mae_list)

            
            data.append([diskname, lgbm_7_model_mae_mean,lgbm_7_model_mae_std, prophet_7_model_mae_mean, prophet_7_model_mae_std, lgbm_30_model_mae_mean, lgbm_30_model_mae_std, prophet_30_model_mae_mean, prophet_30_model_mae_std, arima_30_model_mae_mean, arima_30_model_mae_std])
            

        errors_df = pd.DataFrame(data, columns=["diskname", "lgbm_7_model_mae_mean", "lgbm_7_model_mae_std", "prophet_7_model_mae_mean", 
                                          "prophet_7_model_mae_std", "lgbm_30_model_mae_mean", "lgbm_30_model_mae_std", 
                                          "prophet_30_model_mae_mean", "prophet_30_model_mae_std", "arima_30_model_mae_mean",
                                          "arima_30_model_mae_std"])

        return errors_df
        

        
        
    def split_into_sames_falses(self,df):
        std_df = df[['lgbm_7_model_mae_std','lgbm_30_model_mae_std','prophet_7_model_mae_std','prophet_30_model_mae_std','arima_30_model_mae_std']]
        mean_df=df[['lgbm_7_model_mae_mean','lgbm_30_model_mae_mean','prophet_7_model_mae_mean','prophet_30_model_mae_mean','arima_30_model_mae_mean']]
        df_means_stds=pd.DataFrame()
        df_means_stds['lowest_mean'] = mean_df.idxmin(axis=1)
        df_means_stds['lowest_std'] = std_df.idxmin(axis=1)
        df_means_stds['lowest_mean'] = df_means_stds['lowest_mean'].str[:-9]
        df_means_stds['lowest_std'] = df_means_stds['lowest_std'].str[:-8]
        comparison_column = np.where(df_means_stds["lowest_mean"] == df_means_stds["lowest_std"], True, False)
        df_means_stds["equal"] = comparison_column
        mean_df["is_same"]=df_means_stds["equal"]
        std_df["is_same"]=df_means_stds["equal"]
        only_sames=df_means_stds[df_means_stds.equal]
        only_falses=df_means_stds[(df_means_stds.equal==False)]
        processing_index=only_falses.index
        the_list=[]
        first_models=[]
        second_models=[]
        for i in processing_index:
            if only_falses["lowest_mean"][i]== "lgbm_7_model":
                mean_df["lgbm_7_model_mae_mean"][i]=float(10000)
            elif only_falses["lowest_mean"][i]== "lgbm_30_model":
                mean_df["lgbm_30_model_mae_mean"][i]=float(10000)
            elif only_falses["lowest_mean"][i]== "prophet_7_model":
                mean_df["prophet_7_model_mae_mean"][i]=float(10000)
            elif only_falses["lowest_mean"][i]== "prophet_30_model":
                mean_df["prophet_30_model_mae_mean"][i]=float(10000)
            elif only_falses["lowest_mean"][i]== "arima_30_model":
                mean_df["arima_30_model_mae_mean"][i]=float(10000)

        mean_df['lgbm_7_model_mae_mean'] = pd.to_numeric(mean_df['lgbm_7_model_mae_mean'])
        mean_df['lgbm_30_model_mae_mean'] = pd.to_numeric(mean_df['lgbm_30_model_mae_mean'])
        mean_df['prophet_7_model_mae_mean'] = pd.to_numeric(mean_df['prophet_7_model_mae_mean'])
        mean_df['prophet_30_model_mae_mean'] = pd.to_numeric(mean_df['prophet_30_model_mae_mean'])
        mean_df['arima_30_model_mae_mean'] = pd.to_numeric(mean_df['arima_30_model_mae_mean'])

        mean_df=mean_df.drop(["is_same"], axis=1)

        df_means_stds['second_lowest_mean'] = mean_df.idxmin(axis=1)
        df_means_stds['second_lowest_mean'] = df_means_stds['second_lowest_mean'].str[:-9]

        only_sames=df_means_stds[df_means_stds.equal]
        only_falses=df_means_stds[(df_means_stds.equal==False)]

        return only_sames, only_falses



    def model_decider_different(self,only_falses, df):
        processing_index=only_falses.index
        the_list=[]
        first_models=[]
        second_models=[]
        third_models=[]
        current_mean=[]
        current_stds=[]
        for i in processing_index:
            if only_falses["lowest_mean"][i]== "lgbm_7_model":
                the_mean1= df["lgbm_7_model_mae_mean"][i]
                the_std1= df["lgbm_7_model_mae_std"][i]
                model_1="lgbm_7_model"
            elif only_falses["lowest_mean"][i]== "lgbm_30_model":
                the_mean1= df["lgbm_30_model_mae_mean"][i]
                the_std1= df["lgbm_30_model_mae_std"][i]
                model_1="lgbm_30_model"
            elif only_falses["lowest_mean"][i]== "prophet_7_model":
                the_mean1= df["prophet_7_model_mae_mean"][i]
                the_std1= df["prophet_7_model_mae_std"][i]
                model_1="prophet_7_model"
            elif only_falses["lowest_mean"][i]== "prophet_30_model":
                the_mean1= df["prophet_30_model_mae_mean"][i]
                the_std1= df["prophet_30_model_mae_std"][i]
                model_1="prophet_30_model"
            elif only_falses["lowest_mean"][i]== "arima_30_model":
                the_mean1= df["arima_30_model_mae_mean"][i]
                the_std1= df["arima_30_model_mae_std"][i]
                model_1="arima_30_model"
            #the_interval_1=st.t.interval(0.95, 300, loc=the_mean1, scale=the_std1)
            if only_falses["second_lowest_mean"][i]== "lgbm_7_model":
                the_mean2= df["lgbm_7_model_mae_mean"][i]
                the_std2= df["lgbm_7_model_mae_std"][i]
                model_2="lgbm_7_model"
            elif only_falses["second_lowest_mean"][i]== "lgbm_30_model":
                the_mean2= df["lgbm_30_model_mae_mean"][i]
                the_std2= df["lgbm_30_model_mae_std"][i]
                model_2="lgbm_30_model"
            elif only_falses["second_lowest_mean"][i]== "prophet_7_model":
                the_mean2= df["prophet_7_model_mae_mean"][i]
                the_std2= df["prophet_7_model_mae_std"][i]
                model_2="prophet_7_model"
            elif only_falses["second_lowest_mean"][i]== "prophet_30_model":
                the_mean2= df["prophet_30_model_mae_mean"][i]
                the_std2= df["prophet_30_model_mae_std"][i]
                model_2="prophet_30_model"
            elif only_falses["second_lowest_mean"][i]== "arima_30_model":
                the_mean2= df["arima_30_model_mae_mean"][i]
                the_std2= df["arima_30_model_mae_std"][i]
                model_2="arima_30_model"
            if only_falses["lowest_std"][i]== "lgbm_7_model":
                the_mean3= df["lgbm_7_model_mae_mean"][i]
                the_std3= df["lgbm_7_model_mae_std"][i]
                model_3="lgbm_7_model"
            elif only_falses["lowest_std"][i]== "lgbm_30_model":
                the_mean3= df["lgbm_30_model_mae_mean"][i]
                the_std3= df["lgbm_30_model_mae_std"][i]
                model_3="lgbm_30_model"
            elif only_falses["lowest_std"][i]== "prophet_7_model":
                the_mean3= df["prophet_7_model_mae_mean"][i]
                the_std3= df["prophet_7_model_mae_std"][i]
                model_3="prophet_7_model"
            elif only_falses["lowest_std"][i]== "prophet_30_model":
                the_mean3= df["prophet_30_model_mae_mean"][i]
                the_std3= df["prophet_30_model_mae_std"][i]
                model_3="prophet_30_model"
            elif only_falses["lowest_std"][i]== "arima_30_model":
                the_mean3= df["arima_30_model_mae_mean"][i]
                the_std3= df["arima_30_model_mae_std"][i]
                model_3="arima_30_model"
            #the_interval_2=st.t.interval(0.95, 300, loc=the_mean2, scale=the_std2)
            score_1=the_mean1*the_std1
            score_2=the_mean2*the_std2
            score_3=the_mean3*the_std3
            first_models.append(score_1)
            second_models.append(score_2)
            third_models.append(score_3)
            if score_1<=score_2 and score_1<=score_3:
                the_list.append(model_1)
                current_mean.append(the_mean1)
                current_stds.append(the_std1)
            elif score_2<=score_1 and score_2<=score_3:
                the_list.append(model_2)
                current_mean.append(the_mean2)
                current_stds.append(the_std2)
            else:
                the_list.append(model_3)
                current_mean.append(the_mean3)
                current_stds.append(the_std3)
        only_falses["Prefered model"]=the_list
        only_falses["First Score"]=first_models
        only_falses["Second Score"]=second_models
        only_falses["Third Score"]=third_models
        only_falses["CurrentMean"]=current_mean
        only_falses["CurrentStds"]=current_stds
        only_falses.drop(["equal"], axis=1, inplace=True)

        return only_falses

    def model_decider_sames(self,only_sames, df):    
        sames_index=only_sames.index
        the_same_means=[]
        the_same_stds=[]
        the_scoress=[]
        for i in sames_index:
            the_mean1=-100000000
            the_std1=-1000000000
            if only_sames["lowest_mean"][i]== "lgbm_7_model":
                the_mean1= df["lgbm_7_model_mae_mean"][i]
                the_std1= df["lgbm_7_model_mae_std"][i]
                model_1="lgbm_7_model"
            elif only_sames["lowest_mean"][i]== "lgbm_30_model":
                the_mean1= df["lgbm_30_model_mae_mean"][i]
                the_std1= df["lgbm_30_model_mae_std"][i]
                model_1="lgbm_30_model"
            elif only_sames["lowest_mean"][i]== "prophet_7_model":
                the_mean1= df["prophet_7_model_mae_mean"][i]
                the_std1= df["prophet_7_model_mae_std"][i]
                model_1="prophet_7_model"
            elif only_sames["lowest_mean"][i]== "prophet_30_model":
                the_mean1= df["prophet_30_model_mae_mean"][i]
                the_std1= df["prophet_30_model_mae_std"][i]
                model_1="prophet_30_model"
            elif only_sames["lowest_mean"][i]== "arima_30_model":
                the_mean1= df["arima_30_model_mae_mean"][i]
                the_std1= df["arima_30_model_mae_std"][i]
                model_1="arima_30_model"
            the_same_means.append(the_mean1)
            the_same_stds.append(the_std1)
            the_scoress.append(the_mean1*the_std1)
        only_sames["CurrentAvgMae"]=the_same_means
        only_sames["CurrentStdMae"]=the_same_stds
        only_sames["Scores"]=the_scoress

        return only_sames

    def create_model_mapping_table(self,only_falses, only_sames, df):    
        models_selected=[]
        average_maes=[]
        average_stds=[]
        for i in df.index:
            if i in only_sames.index:
                models_selected.append(only_sames["lowest_mean"][i])
                average_maes.append(only_sames["CurrentAvgMae"][i])
                average_stds.append(only_sames["CurrentStdMae"][i])
            else:
                models_selected.append(only_falses["Prefered model"][i])
                average_maes.append(only_falses["CurrentMean"][i])
                average_stds.append(only_falses["CurrentStds"][i])
        new_df=pd.DataFrame()
        new_df["disks"]= df.diskname
        new_df["Models_Selected"] = models_selected
        new_df["CurrentAvgMae"] = [round(num, 5) for num in average_maes]
        new_df["CurrentStdMae"] = [round(num, 5) for num in average_stds]

        model_ids=[]
        for row,x in new_df.iterrows():
            the_model_name=x.Models_Selected
            if the_model_name == "arima_30_model":
                model_id=1
            elif the_model_name=="lgbm_7_model":
                model_id=2
            elif the_model_name=="lgbm_30_model":
                model_id=3
            elif the_model_name=="prophet_7_model":
                model_id=4
            elif the_model_name=="prophet_30_model":
                model_id=5
            model_ids.append(model_id)
        new_df["ModelId"]=model_ids
        new_df.drop("Models_Selected", axis=1, inplace=True)
        #new_df['CIKey'] = new_df.disks.apply(lambda x: x[0])
        new_df['DiskName'] = new_df.disks.apply(lambda x: int(x))
        new_df.drop("disks", axis=1, inplace=True)
        
        
        return new_df

    def get_model_selection(self,df):
        only_sames, only_falses=self.split_into_sames_falses(df)
        only_falses=self.model_decider_different(only_falses, df)
        only_sames=self.model_decider_sames(only_sames, df)
        
        return self.create_model_mapping_table(only_falses, only_sames, df)