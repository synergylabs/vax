"""
Author: Prasoon Patidar
Created: 28th Sept 2022
Contains classes defining constants needed for data collection and machine learning
"""

from enum import Enum

class Constants:

    """Constants for different paths to be used"""

    #Pipeline Version
    VAX_VERSION = '1.0.0'

    # Path where we store the logs
    LOG_DIR = 'cache/logs/'
    LOG_FILE = 'run_data_collection.log'

    # Seconds in 1 minute
    SECS_IN_MIN = 60

    # Seconds in 1 hour
    SECS_IN_HR = 3600

    # Seconds in 1 day
    SECS_IN_DAY = 86400

    # milli, micro and nano seconds
    MILLISECS_IN_SEC = 1e3
    MICROSECS_IN_SEC = 1e6
    NANOSECS_IN_SEC = 1e9

    # days in 1 year
    DAYS_IN_YEAR = 365

    # No. of seconds for block level analysis
    BLOCK_SIZE = 120

    # Default output dir for file posting
    DEFAULT_OUTPUT_DIR = 'cache/output'

# exit status from all across pipeline
class exitStatus(Enum):
    SUCCESS = 1
    FAILURE = 2
    PARTIAL_SUCCESS=3
    RESULTS_CACHED = 4