'''
Main class for database interface with influxdb
'''

import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import base64
import pickle
import pandas as pd

class InfluxDBO:
    def __init__(self, influx_token, influx_base_url, influx_org='cmu', bucket='sensormlserve'):
        self.influx_token = influx_token
        self.influx_org = influx_org
        self.influx_base_url = influx_base_url
        self.bucket = bucket
        self.client = influxdb_client.InfluxDBClient(url=influx_base_url, token=influx_token, org=influx_org, timeout=100000, retries=3)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()

    def encode_data(self, data):
        '''
        This function encodes the data to base64 format
        '''
        return base64.encodebytes(pickle.dumps(data)).decode()

    def decode_data(self, data):
        '''
        This function decodes the data from base64 format
        '''
        return pickle.loads(base64.decodebytes(data.encode()))

    def write(self, experiment, user, session, location, sensor, featurizer, data_list, encode=False, ts_appended=False):
        '''
        This function writes the data to the influxdb. Assumed that the data is in the form of a list of tuples,
        with first element of tuple as timestamp and second element as data in list format
        '''
        feature_list = []
        for timestamp, data in data_list:
            if ts_appended:
                feature_data_list = data[2:]
                ts_start, ts_end = int(data[0]), int(data[1])
                feature_list.append(
                    (timestamp, ",".join([f"start={ts_start}", f"end={ts_end}"]+[f"f{i}={round(float(feature_data_list[i]),6)}" for i in range(len(feature_data_list))])))
            else:
                feature_list.append(
                    (timestamp, ",".join([f"f{i}={round(float(data[i]),6)}" for i in range(len(data))])))
        write_data = [
            f'{experiment},user={user},session={session},location={location},sensor={sensor},type={featurizer} {features} {timestamp}' for timestamp, features in feature_list
        ]
        response = self.write_api.write(bucket=self.bucket, org=self.influx_org, write_precision='ns', record=write_data)
        return response

    def read(self, experiment, user, location, sensor, featurizer, time_start, time_end, decode=False, return_df = True):
        '''
        thi function reads data into data list from influxdb from start_time to end_time
        :param experiment:
        :param user:
        :param location:
        :param sensor:
        :param featurizer:
        :param time_start:
        :param time_end:
        :return:
        '''
        # print(f"Reading data from influxdb for {experiment} {user} {location} {sensor} {type}")
        df_tables = self.query_api.query_data_frame(f'''
            from(bucket: "{self.bucket}")
            |> range(start: {time_start}, stop: {time_end})
            |> filter(fn: (r) => r._measurement == "{experiment}")
            |> filter(fn: (r) => r.user == "{user}")
            |> filter(fn: (r) => r.location == "{location}")
            |> filter(fn: (r) => r.sensor == "{sensor}")
            |> filter(fn: (r) => r.type == "{featurizer}")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        ''')
        if return_df:
            return df_tables
        else:
            data_list = []
            feature_columns = [col for col in df_tables.columns if col[0]=='f']
            for i in range(len(df_tables)):
                data_list.append((df_tables.iloc[i]['_time'].value, df_tables.iloc[i][feature_columns].values.tolist()))
            return data_list

    def read_live(self, experiment, user, location, sensor, featurizer, time_start, decode=False, return_df = True):
        '''
        thi function reads data into data list from influxdb from start_time to end_time
        :param experiment:
        :param user:
        :param location:
        :param sensor:
        :param featurizer:
        :param time_start:
        :return:
        '''
        # print(f"Reading data from influxdb for {experiment} {user} {location} {sensor} {type}")
        df_tables = self.query_api.query_data_frame(f'''
            from(bucket: "{self.bucket}")
            |> range(start: {time_start})
            |> filter(fn: (r) => r._measurement == "{experiment}")
            |> filter(fn: (r) => r.user == "{user}")
            |> filter(fn: (r) => r.location == "{location}")
            |> filter(fn: (r) => r.sensor == "{sensor}")
            |> filter(fn: (r) => r.type == "{featurizer}")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        ''')
        if return_df:
            return df_tables
        else:
            data_list = []
            feature_columns = [col for col in df_tables.columns if col[0]=='f']
            for i in range(len(df_tables)):
                data_list.append((df_tables.iloc[i]['_time'].value, df_tables.iloc[i][feature_columns].values.tolist()))
            return data_list

    def close(self):
        self.client.close()

    def data_list_to_df(self, data_list):
        '''
        This function converts the data list to a dataframe
        '''
        data_ts = [xr[0] for xr in data_list]
        data_matrix = [xr[1] for xr in data_list]
        df = pd.DataFrame(data_matrix, index=data_ts)
        df = df.reset_index().rename(columns={'index': 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        return df