import unittest
import pandas as pd
import numpy as np
from utils import fill_null

df = pd.read_csv('test.csv')


class Test(unittest.TestCase):
    def test_dataframe_shape(self):
        self.assertEquals(150, df.shape[0])  # 150 Records
        self.assertEquals(5+1, df.shape[1])  # 5 Columns + index Column

    def test_count_of_null_values(self):
        count = 0
        for i in list(df.columns):
            for j in df[i].to_list():
                if pd.isna(j) or j in [None, np.nan, '', '?']:
                    count += 1
        self.assertAlmostEquals(6, count)

    def test_filling_null_values_function(self):
        self.assertAlmostEquals(0, fill_null(
            df, unittest=True, filler=15).isnull().sum().sum())  # Checking Filling null Values with a specific values that got from the UI inputs
