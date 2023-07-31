import paramiko
import time
RESPEAKER_DEVICE_NAME = 'micarray'
RPI_ADDRESS = "192.168.33.120"
RPI_USERNAME = 'pi'
RPI_PASSWORD = 'vax@123'
ODAS_EXECUTABLE_PATH = '/home/pi/odas/build/bin/odaslive'
ODAS_CONFIG_PATH = '/home/pi/odas/build/bin/terminal.cfg'
MICARRAY_SERVERSIDE_SCRIPT_PATH = '/home/pi/vax/sensing/privacy_sensors/micarray/respeaker_rpi/serverside/serverside_client.py'

ssh = paramiko.SSHClient()
# ssh.load_host_keys()
ssh.load_system_host_keys()
ssh.connect(RPI_ADDRESS, 22, username=RPI_USERNAME, password=RPI_PASSWORD)
transport = ssh.get_transport()
transport.set_keepalive(1)
ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(
    f"python3 {MICARRAY_SERVERSIDE_SCRIPT_PATH}")

# while(True):
#     print(ssh_stdout.readline())
def line_buffered(f):
    line_buf = ""
    while (not f.channel.closed) or (f.channel.recv_ready()) or (f.channel.recv_stderr_ready()):
        line_buf += f.read(1).decode()
        if line_buf.endswith('\n'):
            yield line_buf[:-1]
            line_buf = ''
try:
    idx = 0
    for l in line_buffered(ssh_stdout):
        print(idx, l)
        idx+=1
except:
    _,pid_out,_ = ssh.exec_command("ps aux | grep bin/odaslive | awk {'print$2'}")
    time.sleep(1)
    odas_procs = pid_out.readlines()
    odas_kill_cmd = "kill -9 " + " ".join([xr[:-1] for xr in odas_procs])
    _ = ssh.exec_command(odas_kill_cmd)
    ssh.close()