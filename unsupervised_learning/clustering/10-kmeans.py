#!/usr/bin/env python3
"""clustering"""
import sklearn.cluster


def kmeans(X, k):
    """performs K-means on a dataset"""
    kmeans = sklearn.cluster.KMeans(k).fit(X)
    return kmeans.cluster_centers_, kmeans.labels_
