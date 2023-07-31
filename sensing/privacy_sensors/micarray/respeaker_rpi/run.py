import time
import subprocess
import os
import re
import sys
import argparse
import time
from datetime import date, datetime, timedelta
from dateutil import parser
import json
import base64
import socket
import pickle
import threading
import nmap

MICARRAY_PORT = 1243


class RPiReader:
    def __init__(self, ip, port):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip = ip
        self.port = port
        self.s.connect((self.ip, self.port))

    def read(self):
        while True:
            # self.l.acquire()
            try:
                self.data = []
                while True:
                    self.msg = self.s.recv(2048)
                    #                    self.l.acquire()
                    self.data.append(self.msg)
                    # print(self.msg)
                    #                    self.l.release()
                    # print(len(self.msg))
                    # print(len(self.data))
                    if len(b"".join(self.data[len(self.data) - 2:])) == 1644 or len(
                            self.msg) == 1644: break  # Try breaking on >1651 to make it more robust
                self.ans = pickle.loads(b"".join(self.data))
                return self.ans
            except OSError:
                print("Connection closed. Something may be wrong with the Pi")
                break

    def fetch(self):
        # self.l.acquire()
        try:
            print("Data=", len(self.data))
            ans = pickle.loads(b"".join(self.data))
        finally:
            # self.l.release()
            return ans


    def close(self):
        print("Closing Socket")
        self.s.shutdown(socket.SHUT_RDWR)
        self.s.close()


def startMicarray():
    # get broadcast address for our pc, it should be something like 10.x.x.255
    broadcast_addr = os.popen("ifconfig enp0s25 | grep 'broadcast' | awk '{print $NF}'").read().split("\n")[0]
    print(f"Broadcast Address: {broadcast_addr}")

    # get host address based on broadcast address
    host_addr = '.'.join(broadcast_addr.split('.')[:-1] + ['1'])
    print(f"Host Address: {host_addr}")

    # scan and get RPI address in eth local
    nm = nmap.PortScanner()
    scan_range = nm.scan(hosts=f'{broadcast_addr}/24', arguments="-n -sP")
    rpi_addr = ''
    for key in scan_range['scan'].keys():
        if not (key == host_addr):
            rpi_addr = key
    print(f"Raspberry Pi Address: {rpi_addr}")

    if rpi_addr == '':
        print("Raspberry Pi not detected. Exiting..")
        sys.exit(1)
    rpi_proc = RPiReader(rpi_addr, MICARRAY_PORT)
    return rpi_proc

def stopMicarray(rpi_proc):
    rpi_proc.close()
    return None

# -------------------------    MAIN: Code sample to interface with sensor   -----------------------------------------
if __name__ == '__main__':
    # get broadcast address for our pc, it should be something like 10.x.x.255
    # broadcast_addr = os.popen("ifconfig enp0s25 | grep 'broadcast' | awk '{print $NF}'").read().split("\n")[0]
    # print(f"Broadcast Address: {broadcast_addr}")
    #
    # # get host address based on broadcast address
    # host_addr = '.'.join(broadcast_addr.split('.')[:-1] + ['1'])
    # print(f"Host Address: {host_addr}")
    #
    # # scan and get RPI address in eth local
    # nm = nmap.PortScanner()
    # scan_range = nm.scan(hosts=f'{broadcast_addr}/24', arguments="-n -sP")
    # rpi_addr = ''
    # for key in scan_range['scan'].keys():
    #     if not (key == host_addr):
    #         rpi_addr = key
    rpi_addr = '192.168.33.120'
    print(f"Raspberry Pi Address: {rpi_addr}")

    if rpi_addr == '':
        print("Raspberry Pi not detected. Exiting..")
        sys.exit(1)
    else:
        rpi_proc = RPiReader(rpi_addr, MICARRAY_PORT)
        frameData = []
        numFrames = 0
        st_time = time.time()
        try:
            while True:
                rpi_dic = rpi_proc.read()
                time_str = parser.parse(rpi_dic['Timestamp']).strftime("%H:%M:%S.%f")

                #frameData.append(rpi_dic)
                src_str =''
                for sound_source in rpi_dic['SST'].keys():
                    if rpi_dic['SST'][sound_source][0]==0.:
                        ...
                    else:
                        src_str+=f'{sound_source}: {str(rpi_dic["SST"][sound_source])}'
                print(time_str, src_str)
                numFrames += 1
                if time.time() - st_time > 10.0:
                    print(numFrames)
                    numFrames = 0
                    frameData = []
                    st_time = time.time()
        except KeyboardInterrupt:
            rpi_proc.close()
