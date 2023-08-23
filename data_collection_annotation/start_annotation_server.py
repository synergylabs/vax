"""
This is main application to run data collection across multiple sensors
"""

import streamlit as st
import sys
import os
import logging
from datetime import datetime
import time
from logging.handlers import WatchedFileHandler
import traceback
import pandas as pd
import csv
import argparse
from queue import Queue
import numpy as np
import pandas as pd
import streamlit as st
import os
import json
from collections import OrderedDict

# custom library
from config.constants import Constants

# Config to run data collection
config_file = 'config/dc_config.json'
run_config = json.load(open(config_file, 'r'))

# Initialize Logger and streamlit page setup
if True:
    # Initialize the logger

    logger_master = logging.getLogger('vax_data_app')
    logger_master.setLevel(logging.DEBUG)

    if not os.path.exists(Constants.LOG_DIR):
        os.makedirs(Constants.LOG_DIR)

    ## Add core logger handler

    core_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(filename)s:L%(lineno)d | %(thread)d:%(threadName)s | %(levelname)s | %(message)s')
    core_logging_handler = WatchedFileHandler(Constants.LOG_DIR + '/' + Constants.LOG_FILE)
    core_logging_handler.setFormatter(core_formatter)
    logger_master.addHandler(core_logging_handler)

    ## Add stdout logger handler
    console_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(filename)s:L%(lineno)d | %(thread)d:%(threadName)s | %(levelname)s | %(message)s')
    console_log = logging.StreamHandler()
    console_log.setLevel(logging.DEBUG)
    console_log.setFormatter(console_formatter)
    logger_master.addHandler(console_log)

    # initialize main logger
    logger = logging.LoggerAdapter(logger_master, {})

    # make sure out data folder is available
    if not os.path.exists(run_config['out_data_dir']):
        logger.error("Out Data Folder not available, Creating one...")
        os.makedirs(run_config['out_data_dir'])

    # Set Page config
    st.set_page_config(
        page_title="Real-Time Data Collection Dashboard",
        page_icon="random",
        layout="wide",
    )

    # Create experiment dir
    if 'exp_dir' not in st.session_state.keys():
        t_data_collection_start = datetime.now()
        experiment_dir = f"{run_config['out_data_dir']}/{run_config['name']}"
        if not os.path.exists(experiment_dir):
            start_time = time.time()
            os.makedirs(experiment_dir)
        st.session_state['exp_dir'] = experiment_dir
        run_config['experiment_dir'] = st.session_state['exp_dir']
    else:
        run_config['experiment_dir'] = st.session_state['exp_dir']
        experiment_dir = st.session_state['exp_dir']

    if 'recording_in_progress' not in st.session_state.keys():
        st.session_state['recording_in_progress'] = False
    if 'label_count' not in st.session_state.keys():
        st.session_state['label_count'] = {}
    # Get title
    st.title("VAX Data Collection App")
    st.markdown(f"## User: {run_config['name']}")

location_activity_mapping = {
    'kitchen': ['Blender', 'Chopping', 'Grating', 'Microwave', 'WashingDishes', 'DishwasherLoading',
                'DishwasherRunning', 'Baking', 'CookingOnStove', 'FridgeOpen', 'FridgeClose'],
    'livingroom': ['Vacuum', 'Doorbell', 'Coughing', 'Eating', 'Drinking', 'WatchingTV', 'Sitting', 'Walking',
                   'Exercising', 'Knocking', 'Talking'],
    'bathroom': ['Shaver In Use', 'ToilerFlushing', 'HairDryer', 'HairBrush', 'Toothbrush', 'HandWash', 'Shower'],
    'bedroom': ['sleeping', 'sleepstages']
}

left_column, right_column = st.columns(2)
with left_column:
    current_location = st.selectbox("Select Location", location_activity_mapping.keys())
with right_column:
    current_activities = location_activity_mapping[current_location]
    current_activity = st.selectbox("Select Activity", current_activities)

recording_status = st.empty()

st.markdown("""---""")
# labelling the plot
# check is label dict exists
if 'gt_labels' not in st.session_state.keys():
    st.session_state['gt_labels'] = []

if 'labels_loaded' not in st.session_state.keys():
    prev_label_file = f"{experiment_dir}/{run_config['name']}_labels.txt"
    if os.path.exists(prev_label_file):
        f = open(prev_label_file, 'r')
        lines = f.readlines()
        for line in lines[1:]:
            S = line[:-1].split(",")
            st.session_state['gt_labels'].append(S)
    st.session_state['labels_loaded'] = True


# # Add labels in sidebar
# st.sidebar.title("Valid Labels Collected")
# for label_key in sorted(list(st.session_state['labels_count'].keys())):
#     st.sidebar.write(f"{label_key}: {st.session_state['labels_loaded'][label_key]}")

def add_activity_label():
    if label_valid:
        validity = True
    else:
        validity = False
    st.session_state['gt_labels'].append(
        (st.session_state['ts_start_recording'].strftime('%Y%m%d_%H%M%S.%f'),
         st.session_state['ts_stop_recording'].strftime('%Y%m%d_%H%M%S.%f'),
         current_location, current_activity, validity))
    if validity:
        label_key = f'{current_location}_{current_activity}'
        if label_key not in st.session_state['label_count'].keys():
            st.session_state['label_count'][label_key] = 0.
        st.session_state['label_count'][label_key] += 1
    # submit_labels()


def start_recoding():
    st.session_state['ts_start_recording'] = datetime.now()
    st.session_state['recording_in_progress'] = True


def stop_recording():
    st.session_state['ts_stop_recording'] = datetime.now()
    st.session_state['recording_in_progress'] = False


def remove_context_label(label_to_remove_index):
    del st.session_state['gt_labels'][label_to_remove_index]


left_col_a, left_col_b, center_col, right_col = st.columns(4)

with left_col_a:
    is_disabled = False
    if st.session_state['recording_in_progress']:
        is_disabled = True
    label_start = st.button("Start recording", on_click=start_recoding, disabled=is_disabled)
with left_col_b:
    is_disabled = True
    if st.session_state['recording_in_progress']:
        is_disabled = False
    label_end = st.button("Stop recording", on_click=stop_recording, disabled=is_disabled)
with center_col:
    is_disabled = False
    if st.session_state['recording_in_progress']:
        is_disabled = True
    label_valid = st.checkbox("Valid Recording", value=True, disabled=is_disabled)
with right_col:
    is_disabled = False
    if st.session_state['recording_in_progress']:
        is_disabled = True
    add_label = st.button("store_label", on_click=add_activity_label, disabled=is_disabled)

if st.session_state['recording_in_progress']:
    with recording_status.container():
        st.info("Recording in progress")

if label_end:
    with recording_status.container():
        st.warning(f"Recording stopped for f{current_activity}. please store label before proceeding")

if add_label:
    with recording_status.container():
        st.success(f"Recording saved successfully for {current_activity}...")


def submit_labels():
    label_list = st.session_state['gt_labels']
    if len(label_list) == 0:
        st.write("No labels to submit")
        return None
    curr_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    labels_dir = f"{experiment_dir}/"
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    with open(f"{labels_dir}/{run_config['name']}_labels.txt", "w") as f:
        f.write("start_time,end_time,location, activity, is_valid\n")
        f.writelines(
            [f"{label[0]},{label[1]},{label[2]}, {label[3]}, {label[4]}\n" for label in label_list])
    with open(f"{labels_dir}/{run_config['name']}_labels_backup_{curr_timestamp}.txt", "w") as f:
        f.write("start_time,end_time,location, activity, is_valid\n")
        f.writelines(
            [f"{label[0]},{label[1]},{label[2]}, {label[3]}, {label[4]}\n" for label in label_list])
    st.write("Labels submitted successfully...")
    st.balloons()
    return None

st.markdown("""---""")
st.markdown("## Collected labels")
label_list = st.session_state['gt_labels']
label_counts = OrderedDict()
for loc_key in location_activity_mapping.keys():
    label_counts[loc_key] = {}
    for activity_key in location_activity_mapping[loc_key]:
        label_counts[loc_key][activity_key] = 0.

for label in label_list:
    if str(label[4]).strip()=='True':
        label_counts[label[2]][label[3].strip()]+=1

col1, col2, col3 = st.columns(3)
with col1:
    for key, value in label_counts['livingroom'].items():
        st.write(f"{key} : {int(value)}")
with col2:
    for key, value in label_counts['kitchen'].items():
        st.write(f"{key} : {int(value)}")
with col3:
    for key, value in label_counts['bathroom'].items():
        st.write(f"{key} : {int(value)}")



st.markdown("""---""")
st.write("Submit labels on the disk")
submit_button = st.button("Submit Labels")
if submit_button:
    submit_labels()

st.markdown("""---""")
label_list = st.session_state['gt_labels']
left_column, right_column = st.columns([5, 1])
for i, labels in enumerate(label_list):
    # with left_column:
        if str(labels[4]).strip()=="True":
            st.success(f"{i}:{labels[2].upper()}-{labels[3]}")
            st.write(f"Start: {labels[0]} End: {labels[1]}")
        elif str(labels[4]).strip()=="False":
            st.error(f"{i}:{labels[2].upper()}-{labels[3]}")
            st.write(f"Start: {labels[0]} End: {labels[1]}")
        # st.markdown("""---""")
    # with right_column:
        remove_label = st.button("remove label", on_click=remove_context_label,
                                 args=(i,), key=f"remove_{i}")
        st.markdown("""---""")
