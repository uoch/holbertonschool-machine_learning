#!/usr/bin/env python3
"""clustering"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """performs agglomerative clustering on a dataset"""
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(linkage, t=dist,
                                            criterion='distance')
    plt.figure()
    scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.show()
    return clss
