import pandas as pd
from darts import TimeSeries

class DataPreprocess:

    def convert_series2darts(self, series):
        """
        Converts timeseries data into the dart's timeseries format
        :param series: A pandas series representing timeseries data
        :return: A dart's series representing timeseries data
        """
        return TimeSeries.from_series(series)

    def convert_series2prophetdf(self, series):
        """
        Converts timeseries data into the required input format of Prophet
        :param series: A pandas timeseries data
        :return: A pandas dataframe representing the required input format for Prophet
        """
        return pd.DataFrame(data= {'ds': series.index, 'y': series.values})
   
    
 