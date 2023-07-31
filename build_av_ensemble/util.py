"""
Utility functions for optics trainers
"""
from collections import Counter
import numpy as np
import pandas as pd
from HAR.av.AVensembleNNPhase1.config import OpticsConfig
import glob
# from external_mapping import instance_video_map

def merge_dicts(dicts):
    """
    The merge_dicts function takes a list of dictionaries as input and returns a single dictionary.
    The function combines the values for each key in the dictionaries, resulting in one final dictionary.

    Args:
        dicts: Pass a list of dictionaries to the function

    Returns:
        A dictionary

    Doc Author:
        Trelent
    """
    c = Counter()
    for d in dicts:
        c.update(d)
    return dict(c)

def jaccard_score_custom(list1, list2):
    """
    The jaccard_score_custom function takes two lists as input and returns the jaccard score between them.
    The jaccard score is defined as the intersection of two sets divided by their union.
    If either list is empty, then the function will return 0.

    Args:
        list1: Represent the list of words in a document
        list2: Compare the list of predicted labels with

    Returns:
        The jaccard score of two lists

    Doc Author:
        Trelent
    """
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    if union==0:
        return 0.
    return float(intersection) / union

def aggregate_audio_ts_scores(df_audio):
    ptile_10 = int(df_audio.shape[1] / 10)
    if ptile_10 > 0:
        df_audio = df_audio.iloc[:, ptile_10:-ptile_10]
    else:
        df_audio = df_audio.copy()
    df_audio_out = pd.DataFrame(np.percentile(df_audio.values,50,axis=1), index=df_audio.index, columns=[1]).sort_values([1],ascending=False)
    df_audio_out[1] = df_audio_out[1]/df_audio_out[1].sum()
    return df_audio_out

def fetch_model_data(model_list, train_users, test_users):
    """
    The fetch_model_data function takes in a list of models and the train_users and test_users lists.
    It then iterates through each model, reading the output files for that model into dataframes.
    The function then iterates through each activity name (Chopping, Grating, etc.) and instance id
    (e.g., user001_activity-chopping-instance-0001). For each activity/instance pair it checks if there is an output file for that pair; if so it reads the file into a dataframe using pandas read_csv(). It then filters out any labels not present in OpticsConfig.label

    Args:
        model_list: Specify the models to be used for generating the data
        train_users: Filter out the test users from the training data
        test_users: Filter out the test users from the training data

    Returns:
        A dictionary of dictionaries

    Doc Author:
        Trelent
    """
    train_raw_data = dict()
    test_raw_data = dict()
    for model in model_list:
        print(f"Getting output for {model}")
        model_output_files = glob.glob(OpticsConfig.raw_models[model])
        for output_file in model_output_files:
            activity_name = output_file.split("/")[-1].split("_")[0]
            instance_id = "_".join(output_file.split("/")[-1].split("_")[-2:])
            if ('BAD' in output_file.split("/")[-1]):
                continue
            if ('.pb' in instance_id):
                continue
            if (activity_name in OpticsConfig.activity_filters) | (instance_id in OpticsConfig.instance_filters):
                continue
            if model in OpticsConfig.audio_models:
                df_out = pd.read_csv(output_file, header=0, index_col=0)
                if model == 'samosa':
                    # remove labels based on context map
                    df_out = df_out[df_out.index.isin(OpticsConfig.samosa_context_labels_map[OpticsConfig.activity_context_map[activity_name]])]
                df_out = aggregate_audio_ts_scores(df_out)
            elif 'stgcn' in model:
                df_out = pd.read_csv(output_file, delimiter=';', header=None, index_col=0)
                df_out = df_out[df_out[1] > 0.].sort_values(by=[1], ascending=False)
                df_out[1] = np.exp(df_out[1]) / np.sum(np.exp(df_out[1]))
            #             df_out[1] = (1 + df_out[1] + 0.5*np.power(df_out[1],2))/np.sum(1 + df_out[1] + 0.5*np.power(df_out[1],2))
            else:
                df_out = pd.read_csv(output_file, delimiter=';', header=None, index_col=0)

            df_out = df_out[~df_out.index.isin(OpticsConfig.label_filters[model])]
            user_name = output_file.split("/")[-1].split("_")[-2]
            if activity_name in ['Chopping', 'Grating']:
                activity_name = 'Chopping+Grating'
            if activity_name in ['Eating','Drinking']:
                activity_name = 'Drinking/Eating'
            if user_name in train_users:
                if activity_name not in train_raw_data.keys():
                    train_raw_data[activity_name] = {}
                if model not in train_raw_data[activity_name]:
                    train_raw_data[activity_name][model] = []
                train_raw_data[activity_name][model].append(df_out)
            elif user_name in test_users:
                if instance_id not in test_raw_data:
                    test_raw_data[instance_id] = {'activity_gt':activity_name}
                test_raw_data[instance_id][model] = df_out
            else:
                continue
    return train_raw_data, test_raw_data


def get_model_performance(predictions_all, confidence_threshold=0.):
    """
    The get_model_performance function takes in a list of tuples, where each tuple contains the instance ID, ground truth label, predicted label and confidence score.
    It then returns the number of correct predictions (correct_instances), total number of instances (total_instances), accuracy (accuracy) and missing predictions (missing_predictions).


    Args:
        predictions_all: Store the predictions of all models
        confidence_threshold=0.: Filter out the predictions with a confidence score lower than the threshold

    Returns:
        The number of correct instances, total instances, accuracy and the number of missing predictions

    Doc Author:
        Trelent
    """
    correct_instances = 0
    total_instances = 0
    missing_instances = 0
    for (instance_id, gt, pred, pred_score) in predictions_all:
        if (pred == 'Undetected') | (pred_score < confidence_threshold):
            missing_instances += 1
            continue
        else:
            if gt == pred:
                correct_instances += 1
            else:
                print(instance_id, gt, pred, pred_score)
            total_instances += 1
    return correct_instances, total_instances, (correct_instances / total_instances), missing_instances