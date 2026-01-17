import multiprocessing # can use multiprocessing to make code go brrrrrrrrr
import numpy as np
import pandas as pd

class BaseEnsembler:

    def __init__(self, num_trees = 10):
        '''
        TODO: Initialize the BaggingEnsembler with the required hyperparameters
        '''
        self.num_trees = num_trees
        print(f"Fitting with {num_trees} trees")

    def fit(self, X, y):
        '''
        TODO: Implement the fit method for Bagging Ensembler
        '''
        raise NotImplementedError("Function not implemented")

    def predict(self, X):
        raise NotImplementedError("Function not implemented")
