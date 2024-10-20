import requests
import json
from datetime import datetime
import time
import numpy as np
import socket
import sys

HOST = 'localhost'
PORT = 6182
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
s.connect((HOST, PORT))
print('connect to', HOST, PORT)

ts_range_in_s = 10
ts_diff_in_s = 1
location = 'home1.room4'
featurizer = 'f1'
sensor = 'test'
num_sensor_features = 40

metric_name = f'{location}.{featurizer}'
ts_max = int(time.time())
ts_min = ts_max - ts_range_in_s
ts_values = np.arange(ts_min, ts_max, ts_diff_in_s)
features= np.random.randint(0,1000, (ts_values.shape[0],num_sensor_features))

base_url = 'http://localhost:6182'
headers = {
    "Content-Type": "application/text"
}
st_time = time.time()
write_data = '\n'.join([f'put {metric_name} {ts_values[ts_idx]} {features[ts_idx][f_idx]} sensor={sensor},fidx={f_idx}'
              for ts_idx in range(ts_values.shape[0]) for f_idx in range(0,features.shape[1])])
write_data+='\n'
print("request create:",time.time()-st_time)
#st_time = time.time()

#print(write_data)
#print("request_print:",time.time()-st_time)
st_time = time.time()

#s.sendall(write_data.encode('utf-8'));
response = requests.post(f'{base_url}/api/put', headers=headers, data=write_data)
print("data_send:", time.time()-st_time)
# We need to sleep a few seconds before close the socket in this example.
# Otherwise TickTock server might not be able to read data as the socket is closed too early.
#time.sleep(50)
#print("Done sending two put reqeuests:\n"+req);
s.close();

print(response)
print(write_data)
print("response print:",time.time()-st_time)




