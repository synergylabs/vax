import subprocess
import json
import time
from datetime import datetime
import json
import base64


DEVICE_NAME = 'micarray'
WRITE_PATH = '/home/vax/cache/micarray_data'
ODAS_EXECUTABLE_PATH = '/home/vax/sensors/micarray/odas/build/bin/odaslive'
ODAS_CONFIG_PATH = '/home/vax/sensors/micarray/odas/build/bin/terminalv2.cfg'

micarray_proc = subprocess.Popen([ODAS_EXECUTABLE_PATH, "-c", ODAS_CONFIG_PATH],
                                 stdout=subprocess.PIPE)
# micarray_out, _ = micarray_proc.communicate()
store1, store2 = "", ""
sst_ssl_ctr = 1
writefile = f'{WRITE_PATH}/{DEVICE_NAME}_{datetime.now().strftime("%H%M%S")}'
f  = open(writefile, 'w')
print("Process started")
curr_timestamp = time.time()
while True:
    next_line= micarray_proc.stdout.readline().decode()
    print(next_line)
    if len(next_line)==0:
        break
    if sst_ssl_ctr == 1:
        store1 += next_line
        if next_line == '}':
            sst_ssl_ctr = 2
    elif sst_ssl_ctr == 2:
        store2 += next_line
        if next_line == '}':
            store1_arr = json.loads(store1)['src']
            store2_arr = json.loads(store2)['src']
            if ("E" in store1_arr[0].keys()) & ("id" in store2_arr[0].keys()):
                detObj = {"SSL": store1_arr, "SST": store2_arr}
            elif ("id" in store1_arr[0].keys()) & ("E" in store2_arr[0].keys()):
                detObj = {"SSL": store2_arr, "SST": store1_arr}
            else:
                continue
            time_val_ = time.time_ns()
            encoded_data = base64.encodebytes(pickle.dumps(data_dict)).decode()
            f.write(f"{time_val_} | {encoded_data} ||")
            if 
            print((time_val_, detObj))
            store1, store2, sst_ssl_ctr = "", "", 1
