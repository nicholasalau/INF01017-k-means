import math
import pandas as pd
import random
import numpy as np

class DataFrame(object):
    _data_frame = None
    _header = []

    def __init__(self, data_frame):
        self._data_frame = data_frame

    def get_dataframe(self):
        return self._data_frame

    def get_attributes(self):
        return self._data_frame.columns.values


