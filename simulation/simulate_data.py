from __future__ import division
from math import log, exp

import numpy as np
import pandas as pd
import scipy.io as sio
from numpy.random.mtrand import uniform
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

pd.set_option('display.width', 1000)
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000


class SimulatedData:
    def __init__(self,
                 average_death=5,
                 num_var = 5,
                 v1 = (1, 2, 3, 4, 5),
                 v2 = (1, 2, 4, 3, 5),
                 ):
        self.average_death = average_death
        self.num_var = num_var
        self.v1 = v1
        self.v2 = v2


    def generate_data(self, events=400, censor=20):
        np.random.seed(321)
        N = events + censor
        X = uniform(low=-1, high=1, size=[N, self.num_var])

        risk1 = self._linear_H(X, self.v1)
        risk1 = risk1 - np.mean(risk1)
        risk2 = self._linear_H(X, self.v2)
        risk2 = risk2 - np.mean(risk2)

        death_time1 = np.zeros((N, 1))
        death_time2 = np.zeros((N, 1))
        p_death = self.average_death * np.ones((N, 1))
        for i in range(N):
            death_time1[i] = np.random.exponential(p_death[i]) / exp(risk1[i])
            death_time2[i] = np.random.exponential(p_death[i]) / exp(risk2[i])

        end_time_idx = events
        censoring1 = np.ones((N, 1))
        end_time1 = np.sort(death_time1.flatten())[end_time_idx]
        death_time1[death_time1 > end_time1] = end_time1
        censoring1[death_time1 == end_time1] = 0

        censoring2 = np.ones((N, 1))
        end_time2 = np.sort(death_time1.flatten())[end_time_idx]
        death_time2[death_time2 > end_time2] = end_time2
        censoring2[death_time2 == end_time2] = 0

        dataset = {'X':X.astype(np.float32), 'E1':censoring1.astype(np.int32), 'T1':death_time1.astype(np.float32),
                   'E2': censoring2.astype(np.int32), 'T2': death_time2.astype(np.float32)}
        return dataset

    def _linear_H(self,x, b):
        risk = np.dot(x, b)
        return risk
