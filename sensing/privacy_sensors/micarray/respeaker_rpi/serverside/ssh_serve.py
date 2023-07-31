import paramiko
import json
import time
from datetime import datetime

ssh = paramiko.SSHClient()
# ssh.load_host_keys()
ssh.load_system_host_keys()
ssh.connect('192.168.33.120', 22, username='pi', password='vax@123')
ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
    "/home/pi/odas/build/bin/odaslive -c /home/pi/odas/build/bin/terminal.cfg")


def line_buffered(f):
    line_buf = ""
    while (not f.channel.closed) or (f.channel.recv_ready()) or (f.channel.recv_stderr_ready()):
        line_buf += f.read(1).decode()
        if line_buf.endswith('\n'):
            yield line_buf[:-1]
            line_buf = ''


store1, store2 = "", ""
ctr = 1
# curr_time =
for l in line_buffered(ssh_stdout):
    if ctr == 1:
        store1 += l
        if l == '}':
            ctr = 2
    elif ctr == 2:
        store2 += l
        if l == '}':
            # ts = int(json.loads(store)['timeStamp'])
            # print(store1,store2)
            ts1, ts2 = json.loads(store1)['timeStamp'], json.loads(store2)['timeStamp']
            print(datetime.now().strftime("%H:%M:%S.%f"), ts1, ts2)
            store1, store2, ctr = "", "", 1
ssh.close()
