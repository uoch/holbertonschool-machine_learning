#!/usr/bin/env python3
import pandas as pd


def from_file(filename, delimiter):
    """Loads data from a file into a DataFrame"""
    df = pd.read_csv(filename, delimiter=delimiter)
    return df