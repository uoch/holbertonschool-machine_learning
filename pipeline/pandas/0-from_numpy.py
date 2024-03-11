#!/usr/bin/env python3
"""pipelines"""
import numpy as np
import pandas as pd


def from_numpy(array):
    """converts a numpy array to a dataframe"""
    return pd.DataFrame(array)