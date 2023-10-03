'''
This file contains generic utilities used across all sensing platforms
Developer: Prasoon Patidar
Created at: 5th March 2022
'''

import numpy as np
import os
import logging
from logging.handlers import WatchedFileHandler
import subprocess
from collections import Counter
import numpy as np
import pandas as pd

def time_diff(t_start, t_end):
    """
    Get time diff in secs

    Parameters:
        t_start(datetime)               : Start time
        t_end(datetime)                 : End time

    Returns:
        t_diff(int)                     : time difference in secs
    """

    return (t_end - t_start).seconds + np.round((t_end - t_start).microseconds / 1000000, 3)


def get_logger(logname, logdir='cache/logs',console_log=True):
    # Initialize the logger

    logger_master = logging.getLogger(logname)
    logger_master.setLevel(logging.DEBUG)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    ## Add core logger handler

    core_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(filename)s:L%(lineno)d | %(thread)d:%(threadName)s | %(levelname)s | %(message)s')
    core_logging_handler = WatchedFileHandler(logdir + '/' + logname + '.log')
    core_logging_handler.setFormatter(core_formatter)
    logger_master.addHandler(core_logging_handler)

    ## Add stdout logger handler
    if console_log:
        console_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(filename)s:L%(lineno)d | %(thread)d:%(threadName)s | %(levelname)s | %(message)s')
        console_log = logging.StreamHandler()
        console_log.setLevel(logging.DEBUG)
        console_log.setFormatter(console_formatter)
        logger_master.addHandler(console_log)

    # initialize main logger
    logger = logging.LoggerAdapter(logger_master, {})

    return logger

def get_screen_size():
    """
    Get current screen size for selected monitor
    Returns: width and height of screen
    """
    cmd = ['xrandr']
    cmd2 = ['grep', '*']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(cmd2, stdin=p.stdout, stdout=subprocess.PIPE)
    p.stdout.close()
    resolution_string, junk = p2.communicate()
    resolution = resolution_string.split()[0].decode()
    width, height = resolution.split('x')
    return int(width), int(height)


from collections import Counter
import numpy as np
import pandas as pd

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
        The jaccard score between two lists

    Doc Author:
        Trelent
    """

    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    if union==0:
        return 0.
    return float(intersection) / union

def aggregate_ts_scores(df_otc_out):
    """
    The aggregate_ts_scores function takes in a dataframe of audio features and returns a dataframe with the
    audio features aggregated by percentile. The function also sorts the values in descending order, so that
    the most important feature is listed first.

    Args:
        df_otc_out: Store the audio features for each video

    Returns:
        A dataframe with the mean of each row, sorted by descending values

    Doc Author:
        Trelent
    """
    ptile_10 = int(df_otc_out.shape[1] / 10)
    if ptile_10 > 0:
        df_otc_out = df_otc_out.iloc[:, ptile_10:-ptile_10]
    else:
        df_otc_out = df_otc_out.copy()
    df_out = pd.DataFrame(np.percentile(df_otc_out.values,50,axis=1), index=df_otc_out.index, columns=[1]).sort_values([1],ascending=False)
    df_out[1] = df_out[1]/df_out[1].sum()
    return df_out