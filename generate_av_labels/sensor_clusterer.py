'''
this file consist of clustering algortihms at sensor level
'''

import glob
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from collections import Counter
import os
import pickle
from copy import deepcopy
import shutil
# mmwave for noise reduction
import itertools

# throwing sklearn to the problem
from sklearn.metrics import *
from sklearn.preprocessing import normalize
from sklearn.ensemble import *
import xgboost
from sklearn.svm import SVC
from sklearn.model_selection import *
from sklearn.manifold import TSNE
from sklearn.cluster import *
from sklearn.mixture import GaussianMixture


def merge_dicts(dicts):
    """
    The merge_dicts function takes a list of dictionaries as input and returns a single dictionary that combines all the keys and values from the input dictionaries. For example, if you pass in three dictionaries with two key-value pairs each, then merge_dicts will return a single dictionary with six key-value pairs.

    Args:
        dicts: Pass a list of dictionaries to the function

    Returns:
        A dictionary with the counts of all words in a list of dictionaries

    Doc Author:
        Trelent
    """
    c = Counter()
    for d in dicts:
        c.update(d)
    return dict(c)


def cluster_lidar_data(X_train):
    """
    The cluster_lidar_data function takes in the lidar data and performs a t-SNE dimensionality reduction on it.
    It then uses the OPTICS clustering algorithm to cluster the reduced data. The function returns an array of labels,
    where each label represents which cluster each point belongs to.

    Args:
        X_train: Pass the data to be clustered

    Returns:
        The cluster labels for the lidar data

    Doc Author:
        Trelent
    """
    tsne_lidar = TSNE()
    X_train_tsne = tsne_lidar.fit_transform(X_train)
    optics_clu = OPTICS(min_samples=min(1000, X_train.shape[0]//2),n_jobs=-1)
    lidar_cluster_labels = optics_clu.fit_predict(X_train_tsne)
    return lidar_cluster_labels


def cluster_doppler_data(X_train):
    """
    The cluster_doppler_data function takes in the training data and clusters it using OPTICS.
    The function returns a list of cluster labels for each sample in the dataset.

    Args:
        X_train: Pass the training data to the clustering algorithm

    Returns:
        An array of cluster labels for each row in the data set

    Doc Author:
        Trelent
    """

    optics_clu = OPTICS(min_samples=min(250, X_train.shape[0]//2), metric='cosine',n_jobs=-1)
    doppler_cluster_labels = optics_clu.fit_predict(X_train[:, :164])
    return doppler_cluster_labels


def cluster_thermal_data(X_train):
    """
    The cluster_thermal_data function takes in a dataframe of thermal images and returns the cluster labels for each image.

    Args:
        X_train: Pass in the data to be clustered

    Returns:
        The cluster labels for the thermal data

    Doc Author:
        Trelent
    """
    optics_clu = OPTICS(min_samples=min(500, X_train.shape[0]//2), n_jobs=-1)
    thermal_cluster_labels = optics_clu.fit_predict(X_train)
    return thermal_cluster_labels


def cluster_ENV_data(X_train):
    """
    The cluster_ENV_data function takes in the training data and returns a list of cluster labels for each row.
    The OPTICS algorithm is used to perform clustering on the ENV features.

    Args:
        X_train: Pass the training data to the function

    Returns:
        The cluster labels for each data point in the training set

    Doc Author:
        Trelent
    """
    optics_clu = OPTICS(min_samples=min(100, X_train.shape[0]//2))
    ENV_cluster_labels = optics_clu.fit_predict(X_train)
    return ENV_cluster_labels


def cluster_micarray_data(X_train):
    """
    The cluster_micarray_data function takes in the training data and transforms it into a 2D space using t-SNE.
    Then, it clusters the transformed data using OPTICS clustering. The function returns an array of cluster labels for each
    sample in X_train.

    Args:
        X_train: Pass the training data to be clustered

    Returns:
        The cluster labels for the data

    Doc Author:
        Trelent
    """
    tsne_micarray = TSNE()
    X_train_tsne = tsne_micarray.fit_transform(X_train)
    optics_clu = OPTICS(min_samples=25)
    micarray_cluster_labels = optics_clu.fit_predict(X_train_tsne)
    return micarray_cluster_labels


def cluster_PIR_data(X_train):
    """
    The cluster_PIR_data function takes in the PIR data and clusters it using OPTICS.
    The function returns a list of cluster labels for each point in the dataset.

    Args:
        X_train: Pass the training data to the optics algorithm

    Returns:
        The cluster labels for the pir data

    Doc Author:
        Trelent
    """
    '''
    
    :param X_train: 
    :return: 
    '''
    optics_clu = OPTICS(min_samples=20)
    PIR_cluster_labels = optics_clu.fit_predict(X_train)
    return PIR_cluster_labels
