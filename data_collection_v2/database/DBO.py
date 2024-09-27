'''
Main class for database interface with influxdb
'''

import base64
import pickle
import requests
import time
import pandas as pd
import numpy as np
import logging
import json
import subprocess
from datetime import datetime

class DBO:
    def __init__(self, base_url='http://localhost', port=6182, write_protocol='http', remote_user='admin-autonomous', remote_pass='otto-birthday-jake-paucity',remote_url='https://autonomous.prasoonpatidar.com/send_watch_data', experiment='autonomous_2'):
        self.base_url = base_url
        self.port = port
        self.write_protocol = write_protocol
        self.query_api = f'{self.base_url}:{self.port}/api/query'
        if self.write_protocol=='http':
            self.write_api = f'{self.base_url}:{self.port}/api/put'
            self.write_headers = {"Content-Type": "application/text"}
        else:
            self.write_api=None
            self.write_headers = None
            
        if self.write_protocol=='tcp':
            self.write_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
            self.write_socket.connect((HOST, PORT))
        else:
            self.write_socket=None
            
        if self.write_protocol=='remote':
            self.remote_user = remote_user
            self.remote_pass = remote_pass
            self.remote_url = remote_url
            self.experiment = experiment
            self.remote_token=None
            

            
    def is_http_success(self, logger): 
        params = {
            'cmd': 'ping',
        }
        response = requests.post(f'{self.base_url}:{self.port}/api/admin', params=params)
        output_str = response.text
        if output_str=='pong':
            logger.info(f"TSDB is working properly, {output_str}")
            return True
        logger.error(f"TSDB is not working properly, {output_str}")
        return False
    
    def is_remote_live(self, logger):
        ping_request = f"{self.remote_url}/ping"
        try:
            response = requests.get(ping_request)
            response_dict = json.loads(response.text)
            if response_dict['api_status']=='active':
                return True
        except Exception as e:
            logger.error(f"Remote end point is not available, {str(e)}")
            return False
            
    def get_remote_token(self, logger):
        token_request = f"{self.remote_url}/create_access_token?username={self.remote_user}&password={self.remote_pass}"
        try:
            response = requests.get(token_request)
            response_dict = json.loads(response.text)
            token = response_dict["token"]
            return token
        except Exception as e:
            logger.error(f"Unable to retrieve access token, {str(e)}")
            return None
        
    def write_features_http(self, location, sensor, featurizer, ts_values, features, logger=None):
        assert ts_values.shape[0]==features.shape[0]
        metric_name = f'{location}.{featurizer}'
        #st_time = time.time()
        write_data = '\n'.join([f'put {metric_name} {ts_values[ts_idx]} {features[ts_idx][f_idx]} sensor={sensor},fidx={f_idx}'
              for ts_idx in range(ts_values.shape[0]) for f_idx in range(0,features.shape[1])])
        write_data+='\n'
        #st_time = time.time()
        response = requests.post(self.write_api, headers=self.write_headers, data=write_data)
        #print("data_send:", time.time()-st_time)
        return response

    def write_features_tcp(self, location, sensor, featurizer, ts_values, features, logger=None):
        assert ts_values.shape[0]==features.shape[0]
        metric_name = f'{location}.{featurizer}'
        #st_time = time.time()
        write_data = '\n'.join([f'put {metric_name} {ts_values[ts_idx]} {features[ts_idx][f_idx]} sensor={sensor},fidx={f_idx}'
              for ts_idx in range(ts_values.shape[0]) for f_idx in range(0,features.shape[1])])
        write_data+='\n'
        #print("request create:",time.time()-st_time)
        #st_time = time.time()
        self.write_socket.sendall(write_data.encode('utf-8'));
        #print("data_send:", time.time()-st_time)
        return None
        
    def write_features_remote(self, location, sensor, featurizer, ts_values, features, logger=None):
        assert ts_values.shape[0]==features.shape[0]
        if logger is None:
            logger = logging.getLogger(__name__)
        if self.remote_token is None:
            self.remote_token = self.get_remote_token(logger)
            if self.remote_token is None:
                logger.error("Unable to write features to remote db...")
                return None
            else:
                logger.info("Successfully retrieved token...")
        # create payload
        payload = {
            'data_list': {int(ts):features[i].tolist() for i, ts in enumerate(ts_values)},
            'user':location, # Getting the pi location as mac address
            'experiment':self.experiment,
            'sensor':sensor,
            'location':"Kitchen",
            'session':datetime.now().strftime("%d.%m.%Y"),
            'featurizer':featurizer
            }
        try:
            post_request = f'{self.remote_url}/post_features?token={self.remote_token}'
            response = requests.post(post_request, json=payload)
            logger.info(f"Posted features successfuly, {response.text}")
            return None
        except Exception as e:
            logger.error(f"Unable to post data for {sensor}, {location},{featurizer}, {str(e)}")
            return None
            
        
    def write_features(self, location, sensor, featurizer, ts_values, features, logger=None):
        if self.write_protocol=='http':
            return self.write_features_http(location, sensor, featurizer, ts_values, features, logger)
        elif self.write_protocol=='tcp':
            return self.write_features_tcp(location, sensor, featurizer, ts_values, features, logger)
        elif self.write_protocol=='remote':
            return self.write_features_remote(location, sensor, featurizer, ts_values, features, logger)
        else:
            raise NotImplementedError("Write protocol not implemented")
    
    def read_features_live(self, location, sensor, featurizer='f1', ts_diff='10s-ago', return_df = True):
        metric_name = f'{location}.{featurizer}'
        read_url = f'{self.query_api}?start={ts_diff}&m=avg:{metric_name}{{sensor={sensor},fidx=*}}'
        read_response = requests.get(read_url)
        if return_df:
            response_arr = json.loads(read_response.text)
            response_df_dict = {}

            for metrics_ in response_arr:
                response_df_dict[int(metrics_["tags"]["fidx"])] = metrics_["dps"]

            df_response = pd.DataFrame.from_dict(response_df_dict)
            df_response = df_response.reindex(sorted(df_response.columns),axis=1)
        else:
            return read_response.text
    def read_features(self, location, sensor, ts_start, ts_end, featurizer='f1', return_df = True):
        metric_name = f'{location}.{featurizer}'
        read_url = f'{self.query_api}?start={ts_start}&end={ts_end}&m=avg:{metric_name}{{sensor={sensor},fidx=*}}'
        read_response = requests.get(read_url)
        if return_df:
            response_arr = json.loads(read_response.text)
            response_df_dict = {}

            for metrics_ in response_arr:
                response_df_dict[int(metrics_["tags"]["fidx"])] = metrics_["dps"]

            df_response = pd.DataFrame.from_dict(response_df_dict)
            df_response = df_response.reindex(sorted(df_response.columns),axis=1)
            return df_response
        else:
            return read_response.text

    def close(self):
        if self.write_protocol=='tcp':
            self.write_socket.close()
