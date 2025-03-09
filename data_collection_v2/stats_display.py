import time
import subprocess
import base64
import pickle
import sys
import signal
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib import cm
from rich.jupyter import display

from utils import get_logger, time_diff, get_screen_size
import traceback
from pathlib import Path
import os
from database.DBO import DBO
import cv2
from datetime import datetime, timedelta
import pyedid
import docker


def hdmi_display_connected():
    # Path where EDID information is typically stored on Linux systems
    edid_path = "/sys/class/drm/"

    for card in os.listdir(edid_path):
        card_path = os.path.join(edid_path, card)
        if os.path.isdir(card_path):
            edid_file = os.path.join(card_path, "edid")
            if os.path.exists(edid_file):
                try:
                    with open(edid_file, "rb") as f:
                        edid_data = f.read()

                    edid = pyedid.parse_edid(edid_data)
                    print(f"Display detected on {card}:")
                    print(f"Manufacturer: {edid.manufacturer}")
                    print(f"Model: {edid.name}")
                    print(f"Serial Number: {edid.serial}")
                    return True
                except Exception as e:
                    print(f"Error reading EDID from {card}: {str(e)}")
    print("No HDMI display detected")
    return False

def kill_process(stats_process_name, logger):
    running_processes = subprocess.check_output(['ps', 'aux']).decode().split("\n")
    stats_processes = [xr for xr in running_processes if ((stats_process_name in xr) & ('exec bash' not in xr))]
    for stats_process in stats_processes:
        stats_process_info = stats_process.split(" ")[1:]
        logger.info(stats_process_info)
        stats_process_pid = None
        for process_info_str in stats_process_info:
            if not process_info_str == '':
                stats_process_pid = int(process_info_str)
                break
        logger.info(f"Sensor {stats_process_name} process running, PID: {stats_process_pid}")
        if stats_process_pid is None:
            logger.info(f"Stats Subprocess does not exist...")
        else:
            try:
                os.kill(stats_process_pid, signal.SIGKILL)
            except ProcessLookupError:
                logger.info(f"Stats Subprocess does not exist...")
                continue

        time.sleep(2)
        logger.info(f"Stats Subprocess killed: {stats_process_pid}")
    logger.info(f"Successfully Terminated Stats Process...")
    return

def init_cv2_display():

    # check if there is any display connected to the device
    while not hdmi_display_connected():
        print("No HDMI display detected, waiting for display to be connected...")
        time.sleep(5)
    # ch
    print("HDMI display detected, starting stats display...")
    time.sleep(5)
    display_window_name = "Data Collection Stats"
    # Initialize the display
    cv2.namedWindow(display_window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(display_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(display_window_name, 0, 0)
    width, height = get_screen_size()

    # get the top and bottom padding
    top = int(0.05 * height)
    bottom = int(0.05 * height)
    padding = int(0.05 * width)

    return display_window_name, width, height, top, bottom, padding

if __name__=='__main__':
    
    logger = get_logger("stats_display",logdir = f'{Path(__file__).parent}/../../cache/logs',console_log=True)
    disp_window_name,width,height,top, bottom, padding = init_cv2_display()
    x=0

    # get device mac address based experiment name    	
    eth0_mac_cmd = "ifconfig enp86s0 | grep ether | awk 'END { print $2;}'"
    mac_address = subprocess.check_output(eth0_mac_cmd,shell=True).decode('utf-8')
    exp_name=f"vax_{mac_address.replace(':','')}".replace('\n','').replace('$','')
    logger.info(f"Expname: {exp_name}")

    # maintain time for zero frames
    last_system_health_checkpoint = time.time()
    SYSTEM_CHECK_FREQ = 150
    CHECKPOINT_FREQ = 20
    sensors_config = {
        'flir':{'file_ckpt':'/tmp/thermal.ckpt', 'low_fps':3.,'sensor_process_name':'record_flir.py'},
        'rplidar':{'file_ckpt':'/tmp/rplidar.ckpt', 'low_fps':1,'sensor_process_name':'record_rplidar.py'},
        'micarray':{'file_ckpt':'/tmp/micarray.ckpt', 'low_fps':25,'sensor_process_name':'record_respeakerv2.py'},
        'doppler':{'file_ckpt':'/tmp/doppler.ckpt','low_fps':5,'sensor_process_name':'record_doppler.py'},
        'pose':{'file_ckpt':'/tmp/oakdlite_pose.ckpt','low_fps':3,'sensor_process_name':'record_oakdlite.py'},
        'depth':{'file_ckpt':'/tmp/oakdlite_depth.ckpt','low_fps':10,'sensor_process_name':'record_oakdlite.py'},
        'rgb':{'file_ckpt':'/tmp/oakdlite_rgb.ckpt','low_fps':5,'sensor_process_name':'record_oakdlite.py'},
    }
    sensor_live_status = {        
        'flir':{'time_since_ckpt':0.},
        'rplidar':{'time_since_ckpt':0.},
        'micarray':{'time_since_ckpt':0.},
        'doppler':{'time_since_ckpt':0.},
        'pose':{'time_since_ckpt':0.},
        'depth':{'time_since_ckpt':0.},
        'rgb':{'time_since_ckpt':0.}
    }

    while True:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 25)
        image = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(image)
        # Draw a black filled box to clear the image.
        draw.rectangle((0, 0, width, height), outline=0, fill=(0, 0, 0))
        # https://unix.stackexchange.com/questions/119126/command-to-display-memory-usage-disk-usage-and-cpu-load
        curr_time = datetime.now().strftime("Time: %m/%d %H:%M:%S")
        cmd = "hostname -I | cut -d' ' -f1"
        IP = "IP: " + subprocess.check_output(cmd, shell=True).decode("utf-8")
        # cmd = "top -bn1 | grep load | awk '{printf \"CPU Load: %.2f\", $(NF-2)}'"
        # CPU = subprocess.check_output(cmd, shell=True).decode("utf-8")
        # cmd = "free -m | awk 'NR==2{printf \"Mem:%s/%sMB  %.0f%%\", $3,$2,$3*100/$2 }'"
        # MemUsage = subprocess.check_output(cmd, shell=True).decode("utf-8")
        # cmd = 'df -h | awk \'$NF=="/"{printf "Disk: %d/%d GB  %s", $3,$2,$5}\''
        # Disk = subprocess.check_output(cmd, shell=True).decode("utf-8")
        #
        # new compressed load stats
        cmd = "top -bn1 | grep load | awk '{printf \"%.2f\", $(NF-2)}'"
        CPU = subprocess.check_output(cmd, shell=True).decode("utf-8")
        cmd = "free -m | awk 'NR==2{printf \"%.0f%%\", $3*100/$2 }'"
        MemUsage = subprocess.check_output(cmd, shell=True).decode("utf-8")
        cmd = 'df -h | awk \'$NF=="/"{printf "%s", $5}\''
        Disk = subprocess.check_output(cmd, shell=True).decode("utf-8")
        device_load_stats = f"C|M|D:{CPU}|{MemUsage}|{Disk}"
        # logger.info(device_load_stats)
        seperator = '-------------'
        # Write four lines of text.
        y = top
        draw.text((x, y), curr_time, font=font, fill="#FFFFFF")
        y += font.getbbox(curr_time)[3]
        draw.text((x, y), IP, font=font, fill="#FFFFFF")
        y += font.getbbox(IP)[3]
        # ~ draw.text((x, y), CPU, font=font, fill="#00FFFF")
        # ~ y += font.getbbox(CPU)[3]
        # ~ draw.text((x, y), MemUsage, font=font, fill="#00FFFF")
        # ~ y += font.getbbox(MemUsage)[3]
        draw.text((x, y), device_load_stats, font=font, fill="#0000FF")
        y += font.getbbox(Disk)[3]
        draw.text((x, y), seperator, font=font, fill="#FFFFFF")
        y += font.getbbox(seperator)[3]
        
        # check wifi connection status and VPN connection status
        
        wifi_config_file = '/tmp/wifi.ckpt'
        tailscale_config_file = '/tmp/tailscale.ckpt'
        if not os.path.exists(wifi_config_file):
            wifi_stat_str = f"NO WiFi info!!"
            wifi_stat_color = "#FF00FF"
            draw.text((x, y), wifi_stat_str, font=font, fill=wifi_stat_color)
            y += font.getbbox(wifi_stat_str)[3]
            # logger.info(wifi_stat_str)
            # Display image.
            # convert image to cv2 format
            image = np.array(image)
            # convert RGB to BGR
            image = image[:, :, ::-1].copy()
            cv2.imshow(disp_window_name, image)
            if cv2.waitKey(1) == 27:
                logger.info(f"Closing {disp_window_name}")
                break
            if cv2.getWindowProperty(disp_window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            time.sleep(0.5)
            continue
            
        wifi_ckpt_arr = open(wifi_config_file).read().split("|")
        if wifi_ckpt_arr[0]=='connected':
            wifi_stat_str = f'WiFi:{wifi_ckpt_arr[1]}'
            wifi_stat_color = "#00FF00"
            draw.text((x, y), wifi_stat_str, font=font, fill=wifi_stat_color)
            y += font.getbbox(wifi_stat_str)[3]
            if not os.path.exists(tailscale_config_file):
                tailscale_ckpt_arr =["no-config"]
            else:
                tailscale_ckpt_arr = open(tailscale_config_file).read().split("|")
            if tailscale_ckpt_arr[0]=='connected':
                tailscale_host = f'dev: {tailscale_ckpt_arr[1]}'
                tailscale_stat_color = "#00FF00"
                draw.text((x, y), tailscale_host, font=font, fill=tailscale_stat_color)
                y += font.getbbox(tailscale_host)[3]
                tailscale_ip = f'dev-ip:{tailscale_ckpt_arr[2]}'
                draw.text((x, y), tailscale_ip, font=font, fill=tailscale_stat_color)
                y += font.getbbox(tailscale_ip)[3]
            else:
                tailscale_stat_str = f'{tailscale_ckpt_arr[0]}!!!'
                tailscale_stat_color = "#FFFF00"
                draw.text((x, y), tailscale_stat_str, font=font, fill=tailscale_stat_color)
                y += font.getbbox(tailscale_stat_str)[3]
        else:
            wifi_stats_str1="conn. ur phone wifi to:"
            wifi_stats_str2 = f"{wifi_ckpt_arr[1]}."
            wifi_stats_str3 = f"addr: {wifi_ckpt_arr[2]}:{wifi_ckpt_arr[3]}"
            wifi_stat_color = "#FFFFFF"
            draw.text((x, y), wifi_stats_str1, font=font, fill=wifi_stat_color)
            y += font.getbbox(wifi_stats_str1)[3]
            draw.text((x, y), wifi_stats_str2, font=font, fill=wifi_stat_color)
            y += font.getbbox(wifi_stats_str2)[3]
            draw.text((x, y), wifi_stats_str3, font=font, fill=wifi_stat_color)
            y += font.getbbox(wifi_stats_str3)[3]
            
        
        
        #font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        for sensor in sensors_config:
            no_fps_color ='#FF0000'
            low_fps_color='#FFFF00'
            ok_color = '#00FF00'
            pause_color = '#00B9FF'
            diskfull_color = '#FF6C00'
            try:
                ckpt_time, ckpt_fps = open(sensors_config[sensor]['file_ckpt'],'r').read().split(",")
                curr_time = datetime.now()
                time_since_ckpt = int(time_diff(datetime.strptime(ckpt_time,"%Y-%m-%d %H:%M:%S.%f"), curr_time))
                time_since_ckpt = time_since_ckpt%999
                sensor_live_status[sensor]['time_since_ckpt'] = time_since_ckpt
                if (int(float(ckpt_fps)) ==-1):
                    sensor_stat_str = f"{sensor}: >{time_since_ckpt}s, Paused."
                elif (int(float(ckpt_fps)) ==-10):
                    sensor_stat_str = f"{sensor}: >{time_since_ckpt}s, Disk Full."
                elif (int(float(ckpt_fps)) <1) & (float(ckpt_fps) > 0):
                    sensor_stat_str = f"{sensor}: >{time_since_ckpt}s,{round(float(ckpt_fps),1)} fps."
                else:               
                    sensor_stat_str = f"{sensor}: >{time_since_ckpt}s,{int(float(ckpt_fps))} fps."
                sensor_color = ok_color
                if float(ckpt_fps)==0.:
                    sensor_color = no_fps_color
                elif int(float(ckpt_fps))==-1:
                    sensor_color = pause_color
                elif int(float(ckpt_fps))==-10:
                    sensor_color = diskfull_color
                elif float(ckpt_fps) <= sensors_config[sensor]['low_fps']:
                    sensor_color = low_fps_color
                
                if time_since_ckpt > 1.5 * CHECKPOINT_FREQ:
                    sensor_color = no_fps_color
            except:
                # logger.info(traceback.format_exc())
                sensor_stat_str = f"{sensor}: NO INFO !!!"
                sensor_color = "#FFFFFF"
            draw.text((x, y), sensor_stat_str, font=font, fill=sensor_color)
            y += font.getbbox(sensor_stat_str)[3]
            # logger.info(sensor_stat_str)
        
        if time.time() > last_system_health_checkpoint + SYSTEM_CHECK_FREQ:
            # check if database is working, else kill db
            # database_dbo = DBO()
            # if not database_dbo.is_http_success(logger):
            #     database_dbo.close()
            #     logger.error("Database not responding, restarting db")
            #     kill_process('/opt/ticktock/bin/ticktock', logger)
            #     # restart database
            #     # _ = os.popen("sudo systemctl start ticktockdb.service")
            logger.info(device_load_stats)
            for sensor in sensor_live_status:
                if sensor_live_status[sensor]['time_since_ckpt'] > SYSTEM_CHECK_FREQ:
                    logger.info(f"No data collected from sensor {sensor}, restarting recording")
                    process_name = sensors_config[sensor]['sensor_process_name']
                    kill_process(process_name, logger)
                    # restart sensor
                    # _ = os.popen("sudo systemctl start record_%s.service" % (sensor))
                    sensor_live_status[sensor]['time_since_ckpt'] = 0.
                else:
                    logger.info(f"Data collected from sensor {sensor} in last {sensor_live_status[sensor]['time_since_ckpt']}s")
            last_system_health_checkpoint = time.time()
                
            
        # Display image.
        # convert image to cv2 format
        image = np.array(image)
        #convert RGB to BGR
        image = image[:, :, ::-1].copy()
        cv2.imshow(disp_window_name, image)
        if cv2.waitKey(1) == 27:
            logger.info(f"Closing {disp_window_name}")
            break
        if cv2.getWindowProperty(disp_window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        time.sleep(0.5)



