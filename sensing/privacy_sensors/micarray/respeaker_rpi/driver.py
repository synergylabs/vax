"""
Main driver class for doppler data collection from awr1642boost-ods
Author: Prasoon Patidar
Created at: 28th Sept 2022
"""

# basic libraries
import threading
import time
import queue
import logging
import traceback
import sys
import numpy as np
import serial
import os
import socket
import pickle
from pathlib import Path
import tempfile
from serial.tools import list_ports
import nmap
from dateutil import parser

# custom libraries
from sensing.deviceInterface import DeviceInterface

# config parameters (not to be changed)
RESPEAKER_DEVICE_NAME = 'Respeaker4mic'
RESPEAKER_SOCKET_PORT = 1243
BROADCAST_CMD = "ifconfig enp0s25 | grep 'broadcast' | awk '{print $NF}'"


class Respeaker4MicReaderThread(threading.Thread):
    """
    Reader thread for doppler sensing from awr1642boost
    """

    def __init__(self, out_queue, rpi_address, logger):
        threading.Thread.__init__(self)
        self.out_queue = out_queue
        self.logger = logger
        self.rpi_address = rpi_address
        self.running = False

        self.socket_rpi = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.port = RESPEAKER_SOCKET_PORT
        self.socket_rpi.connect((self.rpi_address, self.port))

    def readAndParsePacket(self):
        # Initialize variables
        while True:
            # self.l.acquire()
            try:
                packet_data = []
                while True:
                    socket_msg = self.socket_rpi.recv(2048)
                    packet_data.append(socket_msg)
                    if len(b"".join(packet_data[len(packet_data) - 2:])) == 1644 or len(
                            socket_msg) == 1644: break  # Try breaking on >1651 to make it more robust
                detObj = pickle.loads(b"".join(packet_data))
                return detObj
            except OSError:
                print("Connection closed. Something may be wrong with the Pi")
                break

        return detObj

    def start(self):

        # mark this is running
        self.running = True

        # start
        super(Respeaker4MicReaderThread, self).start()

    def stop(self):
        # set thread running to False
        self.running = False

    def run(self):
        # connect with device for reading data
        try:
            while self.running:
                detObj = self.readAndParsePacket()
                self.out_queue.put(detObj)
            self.logger.debug("Closing Socket")
            self.socket_rpi.shutdown(socket.SHUT_RDWR)
            self.socket_rpi.close()
        except Exception as e:
            self.running = False
            self.logger.debug("Closing Socket")
            self.socket_rpi.shutdown(socket.SHUT_RDWR)
            self.socket_rpi.close()
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)
        finally:
            self.stop()


class Respeaker4MicWriterThread(threading.Thread):
    """
    Writer thread for doppler sensing from awr device
    """

    def __init__(self, in_queue, write_src, logger):
        threading.Thread.__init__(self)

        self.in_queue = in_queue
        self.logger = logger
        self.write_src = write_src
        self.is_running = False

    def start(self):
        # connect with device for reading data
        ...

        # mark this is running
        self.running = True
        self.logger.info(f"Starting writing data from {RESPEAKER_DEVICE_NAME} sensor...")
        # start
        super(Respeaker4MicWriterThread, self).start()

    def stop(self):
        # destroy device relevant object
        # set thread running to False
        self.running = False

    def run(self):
        is_header_set = False
        try:
            with open(self.write_src, 'w') as f:
                while self.running:
                    if not is_header_set:
                        f.write("ts | data\n")
                    data_dict = self.in_queue.get()
                    time_val = parser.parse(data_dict['Timestamp'])
                    ts = int(float(time_val.strftime('%s.%f')) * 1e9)
                    f.write(f"{ts} | {data_dict}\n")
        except Exception as e:
            self.running = False
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)


class Respeaker4MicDevice(DeviceInterface):
    """
    Device implementation for Respeaker4MicDevice
    """

    def __init__(self, run_config, sensor_queue, logger):
        """
        Initialized required containers for sensing device
        Args:
            run_config: basic config for pipeline run
            logger: Logging object
        """
        super().__init__(run_config, sensor_queue, logger)

        # initialize Respeaker4MicDevice
        self.name = RESPEAKER_DEVICE_NAME

        self.rpi_address = ''
        self.reader = None
        self.writer = None
        return

    def is_available(self):
        """
        Check if this particular device is available to collect data from
        Returns: True is device is available else False
        """

        # get broadcast address for our pc, it should be something like 10.x.x.255
        broadcast_addr = os.popen("ifconfig enp0s25 | grep 'broadcast' | awk '{print $NF}'").read().split("\n")[0]
        self.logger.debug(f"Broadcast Address: {broadcast_addr}")

        # get host address based on broadcast address
        host_addr = '.'.join(broadcast_addr.split('.')[:-1] + ['1'])
        self.logger.debug(f"Host Address: {host_addr}")

        # scan and get RPI address in eth local
        nm = nmap.PortScanner()
        scan_range = nm.scan(hosts=f'{broadcast_addr}/24', arguments="-n -sP")
        rpi_addr = ''
        for key in scan_range['scan'].keys():
            if not (key == host_addr):
                rpi_addr = key
        self.logger.debug(f"Raspberry Pi Address: {rpi_addr}")

        if rpi_addr == '':
            self.logger.error("Unable to find respeaker rpi device in local network.")
            return False
        self.rpi_address = rpi_addr
        return True

    def startReader(self):
        """
        Start reader thread, should not be called before initialization
        Returns: True if initialization successful, else false
        """
        try:
            self.reader = Respeaker4MicReaderThread(self.sensor_queue,
                                                 self.rpi_address,
                                                 self.logger)
            self.reader.start()

            return True
        except:
            self.logger.error(f"Failed to start reader thread: {traceback.format_exc()}")
            return False

    def stopReader(self):
        """
        Gracefully stop reader thread, and destroy device relevant objects
        Returns: True if thread destroyed successfully, else false
        """
        try:
            self.reader.stop()
            self.reader.join()
            return True
        except:
            self.logger.error(f"Failed to stop reader thread, {traceback.format_exc()}")
            return False

    def startWriter(self):
        """
        Start writer thread, should not be called before initialization
        Returns: True if initialization successful, else false
        """
        try:
            self.write_method = 'csv'
            if self.write_method == 'csv':
                self.write_src = f"{self.run_config['experiment_dir']}/respeaker.csv"
                print(f"Write Source: {self.write_src}")
                self.writer = Respeaker4MicWriterThread(self.sensor_queue,
                                                     self.write_src,
                                                     self.logger)
                self.writer.start()
                return True
        except:
            self.logger.error(f"Failed to start writer thread, {traceback.format_exc()}")
            return False

    def stopWriter(self):
        """
        Gracefully stop writer thread, and destroy device relevant objects
        Returns: True if thread destroyed successfully, else false
        """
        try:
            self.writer.stop()
            self.writer.join()
            return True
        except:
            self.logger.error(f"Failed to stop writer thread, {traceback.format_exc()}")
            return False
