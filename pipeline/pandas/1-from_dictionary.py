#!/usr/bin/env python3
import pandas as pd


def from_dictionary(data):
    """Converts a dictionary to a DataFrame"""
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    data = {"First": [0.0, 0.5, 1.0, 1.5],
            "Second": ["one", "two", "three", "four"]}
    label = ["A", "B", "C", "D"]
    df = from_dictionary(data)
    df.index = label
