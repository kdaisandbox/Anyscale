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
