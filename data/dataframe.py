import math
import pandas as pd
import random
import numpy as np
random.seed(666)


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

    



