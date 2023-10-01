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