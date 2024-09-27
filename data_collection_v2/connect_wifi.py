import subprocess
import os
from utils import get_logger
import time
import traceback
from pathlib import Path

WIFI_ACCESSPOINT_SCRIPT_FILE = f'{Path(__file__).parent}/wifi_options/wifi-connect'
ckpt_file = '/tmp/wifi.ckpt'
tailscale_ckpt_file = '/tmp/tailscale.ckpt'
tailscale_auth_token_file = '/home/vax/.ssh/tailscale_authkey'
tailscale_device_remove_script = f'{Path(__file__).parent}/wifi_options/tailscale_remove_device.sh'
tailscale_auth_token = open(tailscale_auth_token_file).read().strip()
ACCESS_POINT_GATEWAY = '192.168.42.5'
ACCESS_POINT_PORT = '80'
ACCESS_POINT_SSID = 'VAX RPI-Connect'
ACCESS_POINT_UI_DIR = f'{Path(__file__).parent}/wifi_options/ui'

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def execute_cmd(msg, cmd, logger, show_command=True):
	logger.info(msg)
	if show_command:
		print("Executing: " + bcolors.OKBLUE + f"{' '.join(cmd)}" + bcolors.ENDC)
	ps = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	stdout, stderr = ps.communicate()
	ps.wait()
	print("Output: " + bcolors.OKGREEN + stdout.decode() + bcolors.ENDC)
	if len(stderr.decode().strip()) > 0:
		print("Error: " + bcolors.FAIL + stderr.decode() + bcolors.ENDC)
	return stdout.decode(), stderr.decode()

if __name__ == '__main__':
	logger = get_logger("wifi_connect", logdir = f'{Path(__file__).parent}/../../cache/logs', console_log=False)
	
	# get device mac address based experiment name    	
	eth0_mac_cmd = "ifconfig eth0 | grep ether | awk 'END { print $2;}'"
	mac_address = subprocess.check_output(eth0_mac_cmd,shell=True).decode('utf-8')
	tailscale_rpi_hostname=f"rpi{mac_address.replace(':','')}".replace('\n','').replace('$','')
    
	# over the loop check if wifi access point exists, and is active, if not, initialize accesspoint script
	while True:
	#check current access point
		ps = subprocess.Popen(['iwgetid'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
		try:
			output = subprocess.check_output(('grep', 'ESSID'), stdin=ps.stdout).decode()
			network_name = output.replace("\n","").split("ESSID:")[-1]
			with open(ckpt_file,'w') as ckpt_f:
				ckpt_f.write(f'connected|{network_name}')
			logger.info(f"got access point {str(output)}:{network_name}")
			# check if tailscale is up and running
			try:
				tailscale_host_cmd = f"tailscale ip {tailscale_rpi_hostname}" 
				# ~ host_output = subprocess.check_output(tailscale_host_cmd,shell=True).decode('utf-8') 
				host_output,host_err = execute_cmd("Get host info", tailscale_host_cmd.split(" "),logger)
				tailscale_ip_cmd = f"tailscale ip -4" 
				# ~ ip_output = subprocess.check_output(tailscale_ip_cmd,shell=True).decode('utf-8').replace('\n','').replace('$','')
				ip_output,ip_err = execute_cmd("Get IP Info",tailscale_ip_cmd.split(" "),logger)
				# ~ logger.info(f"Host Output: {host_output}, IP Output: {ip_output}")
				# ~ logger.info(f"Host Error: {host_err}, IP Error: {ip_err}")
				if (not (host_err=='')) or (not (ip_err=='')) or (ip_output not in host_output):
					logger.info("Tailscale is not set up")
					logger.info("Attempting to register tailscale device")
					# make sure that the device is not pre-registered and then setting up again due to sd card switch.
					output,err=execute_cmd(f"Deregister {tailscale_rpi_hostname}, if exists", [tailscale_device_remove_script, tailscale_rpi_hostname], logger)
					logger.info(f"Output: {output}")
					logger.info(f"Error: {err}")
					output,err=execute_cmd("Turning down tailscale service", ["systemctl","stop","tailscaled.service"], logger)
					logger.info(f"Output: {output}")
					logger.info(f"Error: {err}")
					if not err=='':
						continue
					output,err = execute_cmd("Remove Existing State", ["rm","/var/lib/tailscale/tailscaled.state"], logger)
					logger.info(f"Output: {output}")
					logger.info(f"Error: {err}")
					output,err = execute_cmd("Turn up tailscale service", ["systemctl","start","tailscaled.service"], logger)
					logger.info(f"Output: {output}")
					logger.info(f"Error: {err}")
					if not err=='':
						continue
					# register tailscale device
					output,err = execute_cmd("Register new tailscale device", ["tailscale","up","--hostname",tailscale_rpi_hostname, "--authkey",tailscale_auth_token], logger, show_command=True)
					logger.info(f"Output: {output}")
					logger.info(f"Error: {err}")
				else:
					logger.info("Tailscale is set up correctly.")
					with open(tailscale_ckpt_file,'w') as ckpt_f:
						ckpt_f.write(f'connected|{tailscale_rpi_hostname}|{ip_output.strip()}')				
			except:
				logger.info(f"\n\nTailscale setup error: {traceback.format_exc()}")
				with open(tailscale_ckpt_file,'w') as ckpt_f:
					ckpt_f.write(f'tailscale_not_connected')				
		except subprocess.CalledProcessError:
			# grep did not match any lines
			logger.info("No wireless networks connected, need to initialize accesspoint...")
			with open(ckpt_file,'w') as ckpt_f:
				ckpt_f.write(f'not_connected|{ACCESS_POINT_SSID}|{ACCESS_POINT_GATEWAY}|{ACCESS_POINT_PORT}')
			proc_out = subprocess.run([WIFI_ACCESSPOINT_SCRIPT_FILE,
			'-g', ACCESS_POINT_GATEWAY,
			'-o', ACCESS_POINT_PORT,
			'-s',ACCESS_POINT_SSID,
			'-u',ACCESS_POINT_UI_DIR],capture_output=True, text=True)
			logger.info(f"Connected to access point with logs {str(proc_out.stdout)}")
		
		time.sleep(10)
		
