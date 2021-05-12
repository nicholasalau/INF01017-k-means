import math
import pandas as pd
import random
import numpy as np
random.seed(123)


class DataFrame(object):
    _data_frame = None
    _header = []

    def __init__(self, data_frame):
        self._data_frame = data_frame

    def get_dataframe(self):
        return self._data_frame

    def get_attributes(self):
        return self._data_frame.columns.values

    def get_values(self):
        return self._data_frame.values
        

    def get_random_instances(self, n):
        return random.sample(range(0, len(self.get_values())), n)

    def get_column(self, column):
        return self._data_frame[column]

    def pre_process(self, column, categoric, values_mapping=None):
        df = self._data_frame.copy()

        most_occured_value = df[column].value_counts()[:1].index.tolist()[0] # used in case of NaN
        column_values = df[column].copy()

        # Treatment of NaN
        for index, value in enumerate(df[column]):
            try:
                if math.isnan(value):
                    column_values[index+1] = most_occured_value
            except:
                pass

        # Save the new dataframe (without NaN)
        df[column] = column_values

        if categoric == True:
            # Map the categoric values
            df[column] = df[column].map(values_mapping)
            
            df[column] = pd.DataFrame(df[column].values, columns=[column]).values
            self._data_frame = df
        else:
            print("sxee")

    



