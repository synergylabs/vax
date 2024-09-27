import time
import subprocess
import board
import digitalio
from adafruit_rgb_display import st7789  # pylint: disable=unused-import
import base64
import pickle
import sys
import signal
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib import cm
from utils import get_logger, time_diff
import traceback
from pathlib import Path
import os
from database.DBO import DBO
from datetime import datetime, timedelta

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



def init_tft_display():
    #initialize minitft display
    # Configuration for CS and DC pins (these are PiTFT defaults):
    
    # cs_pin = digitalio.DigitalInOut(board.CE0) (REMOVED FOR RPI-5)
    cs_pin=None
    dc_pin = digitalio.DigitalInOut(board.D25)
    reset_pin = digitalio.DigitalInOut(board.D24)

    # Config for display baudrate (default max is 24mhz):
    BAUDRATE = 24000000
    # Setup SPI bus using hardware SPI:
    spi = board.SPI()
    disp = st7789.ST7789(spi, height=240, width=240, x_offset=0, y_offset=80, rotation=0,  # 1.3", 1.54" ST7789
        cs=cs_pin,
        dc=dc_pin,
        rst=reset_pin,
        baudrate=BAUDRATE,
    )

    if disp.rotation % 180 == 90:
        height = disp.width  # we swap height/width to rotate it to landscape!
        width = disp.height
    else:
        width = disp.width  # we swap height/width to rotate it to landscape!
        height = disp.height

    # fixed constants for stats display
    padding = -2
    top = padding
    bottom = height - padding
    # Move left to right keeping track of the current x position for drawing shapes.
    x = 0
    image = Image.new("RGB", (width, height))
    # Get drawing object to draw on image.
    draw = ImageDraw.Draw(image)
    # Draw a black filled box to clear the image.
    draw.rectangle((0, 0, width, height), outline=0, fill=(0, 0, 0))
    y = top
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)

    draw.text((x, y), "Starting \n Data \n Collection...", font=font, fill="#FFFFFF")
    disp.image(image, disp.rotation)
    # Turn on the backlight
    backlight = digitalio.DigitalInOut(board.D22)
    backlight.switch_to_output()
    backlight.value = True
    return disp, width, height, top, bottom, padding

def displayWrite(display_text, y_offset):
    ...
    
if __name__=='__main__':
    
    logger = get_logger("stats_display",logdir = f'{Path(__file__).parent}/../../cache/logs',console_log=True)
    disp,width,height,top, bottom, padding = init_tft_display()
    x=0

    # get device mac address based experiment name    	
    eth0_mac_cmd = "ifconfig eth0 | grep ether | awk 'END { print $2;}'"
    mac_address = subprocess.check_output(eth0_mac_cmd,shell=True).decode('utf-8')
    exp_name=f"rpi5_{mac_address.replace(':','')}".replace('\n','').replace('$','')
    logger.info(f"Expname: {exp_name}")

    # maintain time for zero frames
    last_system_health_checkpoint = time.time()
    SYSTEM_CHECK_FREQ = 150
    CHECKPOINT_FREQ = 20
    sensors_config = {
        'flir':{'file_ckpt':'/tmp/thermal.ckpt', 'tsdb_ckpt':'/tmp/thermal_tsdb.ckpt', 'low_fps':3.},
        'rplidar':{'file_ckpt':'/tmp/rplidar.ckpt', 'tsdb_ckpt':'/tmp/rplidar_tsdb.ckpt','low_fps':1},
        #'micarray':{'file_ckpt':'/tmp/micarray.ckpt', 'tsdb_ckpt':'/tmp/micarray_tsdb.ckpt', 'low_fps':25},
        'yamnet':{'file_ckpt':'/tmp/yamnet.ckpt', 'tsdb_ckpt':'/tmp/yamnet_tsdb.ckpt', 'low_fps':5},
        'tofimager':{'file_ckpt':'/tmp/tofimager.ckpt', 'tsdb_ckpt':'/tmp/tofimager_tsdb.ckpt', 'low_fps':5}
    }
    sensor_live_status = {        
        'flir':{'time_since_ckpt':0.},
        'rplidar':{'time_since_ckpt':0.},
        #'micarray':{'time_since_ckpt':0.},
        'yamnet':{'time_since_ckpt':0.},
        'tofimager':{'time_since_ckpt':0.}}

    while True:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        image = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(image)
        # Draw a black filled box to clear the image.
        draw.rectangle((0, 0, width, height), outline=0, fill=(0, 0, 0))
        # https://unix.stackexchange.com/questions/119126/command-to-display-memory-usage-disk-usage-and-cpu-load
        curr_time = datetime.now().strftime("Time: %m/%d %H:%M:%S")
        cmd = "hostname -I | cut -d' ' -f1"
        IP = "IP: " + subprocess.check_output(cmd, shell=True).decode("utf-8")
        # ~ cmd = "top -bn1 | grep load | awk '{printf \"CPU Load: %.2f\", $(NF-2)}'"
        # ~ CPU = subprocess.check_output(cmd, shell=True).decode("utf-8")
        # ~ cmd = "free -m | awk 'NR==2{printf \"Mem:%s/%sMB  %.0f%%\", $3,$2,$3*100/$2 }'"
        # ~ MemUsage = subprocess.check_output(cmd, shell=True).decode("utf-8")
        # ~ cmd = 'df -h | awk \'$NF=="/"{printf "Disk: %d/%d GB  %s", $3,$2,$5}\''
        # ~ Disk = subprocess.check_output(cmd, shell=True).decode("utf-8")
        
        # new compressed load stats
        cmd = "top -bn1 | grep load | awk '{printf \"%.2f\", $(NF-2)}'"
        CPU = subprocess.check_output(cmd, shell=True).decode("utf-8")
        cmd = "free -m | awk 'NR==2{printf \"%.0f%%\", $3*100/$2 }'"
        MemUsage = subprocess.check_output(cmd, shell=True).decode("utf-8")
        cmd = 'df -h | awk \'$NF=="/"{printf "%s", $5}\''
        Disk = subprocess.check_output(cmd, shell=True).decode("utf-8")
        device_load_stats = f"C|M|D:{CPU}|{MemUsage}|{Disk}"
        
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
        
        if time.time() > last_system_health_checkpoint + SYSTEM_CHECK_FREQ:
            # check if database is working, else kill db
            database_dbo = DBO()
            if not database_dbo.is_http_success(logger):
                database_dbo.close()
                logger.error("Database not responding, restarting db")
                kill_process('/home/vax/vax-rpi/sensing/database/ticktock.0.11.1/bin/tt', logger)
                # restart database
                #_ = os.popen("sudo systemctl start ticktockdb.service")
            
            for sensor in sensor_live_status:
                if sensor_live_status[sensor]['time_since_ckpt'] > SYSTEM_CHECK_FREQ:
                    logger.info(f"No data collected from sensor {sensor}, restarting recording")
                    kill_process(f'record_{sensor}.py', logger)
                    # restart sensor
                    # _ = os.popen("sudo systemctl start record_%s.service" % (sensor))
                    sensor_live_status[sensor]['time_since_ckpt'] = 0.
            last_system_health_checkpoint = time.time()
                
            
        # Display image.
        disp.image(image, disp.rotation)
        time.sleep(0.5)



