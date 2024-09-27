import requests
import json
from datetime import datetime
import time
import numpy as np
import time
import pandas as pd


location = 'test_user_feb5'
featurizer = 'f1'
sensor = 'rplidar'

metric_name = f'{location}.{featurizer}'

base_url = 'http://localhost:6182'

read_url = f'{base_url}/api/query?start=10s-ago&m=avg:{metric_name}{{sensor={sensor},fidx=*}}'
print(read_url)
st_time  = time.time()
read_response = requests.get(read_url)
print("Read time:", time.time()-st_time)
# print(read_response)

response_arr = json.loads(read_response.text)
response_df_dict = {}

for metrics_ in response_arr:
    response_df_dict[int(metrics_["tags"]["fidx"])] = metrics_["dps"]

df_response = pd.DataFrame.from_dict(response_df_dict)
df_response = df_response.reindex(sorted(df_response.columns),axis=1)

#df_response.info()
print(df_response)
