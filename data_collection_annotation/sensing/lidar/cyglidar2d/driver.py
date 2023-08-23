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
from pathlib import Path
import tempfile
from serial.tools import list_ports

# custom libraries
from sensing.deviceInterface import DeviceInterface

# config parameters (not to be changed)
CYGLIDAR_DEVICE_NAME = 'CYGLIDAR_2D'
CYGLIDAR_PRODUCT_NAME = 'USB-Serial Controller D'
# Constants
RESPONSE_DEVICE_INFO = np.frombuffer(b"\x10", dtype='uint8')[0]
RESPONSE_2D_MODE = np.frombuffer(b"\x01", dtype='uint8')[0]
RESPONSE_3D_MODE = np.frombuffer(b"\x08", dtype='uint8')[0]
maxBufferSize = 2 ** 15
headerLengthInBytes = 1
magicWord = [90, 119, 255]
word_3byte = [2 ** 16, 2 ** 8, 1]
SERIAL_BAUDRATE = 3000000  # set Baud rate to 9600
SERIAL_BYTESIZE = 8  # Number of data bits = 8
SERIAL_PARITY = 'N'  # No parity
SERIAL_STOPBITS = 1  # Number of Stop bits = 1
DATA_2D_REQUEST = b"\x5A\x77\xFF\x02\x00\x01\x00\x03"
DATA_STOP_REQUEST = b'\x5A\x77\xFF\x02\x00\x02\x00\x00'


class Cyglidar2DReaderThread(threading.Thread):
    """
    Reader thread for doppler sensing from awr1642boost
    """

    def __init__(self, out_queue, port_address, logger):
        threading.Thread.__init__(self)
        self.out_queue = out_queue
        self.logger = logger
        self.port_address = port_address
        self.running = False

        self.byteBuffer = np.zeros(2 ** 15, dtype='uint8')
        self.byteBufferLength = 0
        self.config_parameters = {}
        self.serialport = None

    def configure_serial(self):
        serialObj = serial.Serial(self.port_address)
        serialObj.baudrate = SERIAL_BAUDRATE  # set Baud rate to 9600
        serialObj.bytesize = SERIAL_BYTESIZE  # Number of data bits = 8
        serialObj.parity = SERIAL_PARITY  # No parity
        serialObj.stopbits = SERIAL_STOPBITS  # Number of Stop bits = 1
        data_2d_request = DATA_2D_REQUEST
        serialObj.write(data_2d_request)
        return serialObj

    def readAndParsePacket(self, dataport):
        # Initialize variables
        magicOK = 0  # Checks if magic number has been read
        dataOK = 0  # Checks if the data has been read correctly
        frameNumber = 0
        detObj = {}
        startIdx = None

        readBuffer = dataport.read(dataport.in_waiting)
        byteVec = np.frombuffer(readBuffer, dtype='uint8')
        byteCount = len(byteVec)
        # print("bytes read:",byteCount)

        # Check that the buffer is not full, and then add the data to the buffer
        if (self.byteBufferLength + byteCount) < maxBufferSize:
            self.byteBuffer[self.byteBufferLength:self.byteBufferLength + byteCount] = byteVec[:byteCount]
            self.byteBufferLength = self.byteBufferLength + byteCount

        # Check that the buffer has some data
        if self.byteBufferLength > 3:

            # Check for all possible locations of the magic word
            possibleLocs = np.where(self.byteBuffer == magicWord[0])[0]

            # Confirm that is the beginning of the magic word and store the index in startIdx
            startIdx = []
            for loc in possibleLocs:
                check = self.byteBuffer[loc:loc + 3]
                if np.all(check == magicWord):
                    startIdx.append(loc)

        # Check that startIdx is not empty
        if startIdx:
            # print("start found at location",startIdx[0])
            # Remove the data before the first start index
            if 0 < startIdx[0] < self.byteBufferLength:
                self.byteBuffer[:self.byteBufferLength - startIdx[0]] = self.byteBuffer[
                                                                        startIdx[0]:self.byteBufferLength]
                self.byteBuffer[self.byteBufferLength - startIdx[0]:] = np.zeros(
                    len(self.byteBuffer[self.byteBufferLength - startIdx[0]:]),
                    dtype='uint8')
                self.byteBufferLength = self.byteBufferLength - startIdx[0]

            # Check that there have no errors with the byte buffer length
            if self.byteBufferLength < 0:
                self.byteBufferLength = 0

            # word array to convert 2 bytes to a 16 bit number
            word = [1, 2 ** 8]

            # Read the total packet length
            totalPacketLen = np.matmul(self.byteBuffer[3:3 + 2], word)

            # Check that all the packet has been read
            if (self.byteBufferLength >= totalPacketLen + 6) and (self.byteBufferLength != 0):
                magicOK = 1

        # If magicOK is equal to 1 then process the message
        if magicOK:
            # word array to convert 2 bytes to a 16 bit number
            word = [1, 2 ** 8]

            # Initialize the pointer index
            idX = 0

            # Read the header
            magicNumber = self.byteBuffer[idX:idX + 3]
            idX += 3
            totalPacketLen = np.matmul(self.byteBuffer[idX:idX + 2], word)
            idX += 2
            payload_hdr = self.byteBuffer[idX]
            idX += 1

            # Version info response
            if payload_hdr == RESPONSE_DEVICE_INFO:
                firm_version_hsb = self.byteBuffer[idX]
                idX += 1
                firm_version_msb = self.byteBuffer[idX]
                idX += 1
                firm_version_lsb = self.byteBuffer[idX]
                idX += 1
                hardware_version_hsb = self.byteBuffer[idX]
                idX += 1
                hardware_version_msb = self.byteBuffer[idX]
                idX += 1
                hardware_version_lsb = self.byteBuffer[idX]
                idX += 1
                detObj['firmware_version'] = f'{firm_version_hsb}.{firm_version_msb}.{firm_version_lsb}'
                detObj['hardware_version'] = f'{hardware_version_hsb}.{hardware_version_msb}.{hardware_version_lsb}'

                checksum = self.byteBuffer[idX]
                idX += 1
                dataOK = 1

            if payload_hdr == RESPONSE_2D_MODE:
                info_2d_hdr = np.arange(-60, +60.5, 0.75)
                info_2d_size = info_2d_hdr.shape[0]
                info_2d = np.full(info_2d_size, fill_value=-1, dtype='uint16')
                for i in range(0, info_2d_size):
                    info_2d[i] = np.matmul(self.byteBuffer[idX:idX + 2], word)
                    idX += 2
                checksum = self.byteBuffer[idX]
                idX += 1
                dataOK = 1
                detObj.update({'hdr_2d': info_2d_hdr, 'arr_2d': info_2d})

            if payload_hdr == RESPONSE_3D_MODE:
                info_3d_col_hdr = np.linspace(-60, 60, 160)
                info_3d_row_hdr = np.linspace(-65 / 2, 65 / 2, 60)
                info_3d_row_size, info_3d_col_size = info_3d_row_hdr.shape[0], info_3d_col_hdr.shape[0]
                info_3d = np.full((info_3d_row_size, info_3d_col_size), fill_value=-1, dtype='uint16')
                for row_idx in range(info_3d_row_size):
                    for col_idx in range(0, info_3d_col_size, 2):
                        bit_24_val = np.matmul(self.byteBuffer[idX:idX + 3], word_3byte)
                        idX += 3
                        info_3d[row_idx][col_idx] = bit_24_val // 2 ** 12
                        info_3d[row_idx][col_idx + 1] = bit_24_val % 2 ** 12
                checksum = self.byteBuffer[idX]
                idX += 1
                dataOK = 1
                detObj.update({'mat_3d': info_3d, 'row_hdr': info_3d_row_hdr, 'col_hdr': info_3d_col_hdr})

            # Remove already processed data
            if idX > 0 and self.byteBufferLength >= idX:
                shiftSize = totalPacketLen + 6

                self.byteBuffer[:self.byteBufferLength - shiftSize] = self.byteBuffer[shiftSize:self.byteBufferLength]
                self.byteBuffer[self.byteBufferLength - shiftSize:] = np.zeros(
                    len(self.byteBuffer[self.byteBufferLength - shiftSize:]),
                    dtype='uint8')
                self.byteBufferLength = self.byteBufferLength - shiftSize

                # Check that there are no errors with the buffer length
                if self.byteBufferLength < 0:
                    self.byteBufferLength = 0

        return dataOK, detObj

    def start(self):

        # mark this is running
        self.running = True

        # start
        super(Cyglidar2DReaderThread, self).start()

    def stop(self):
        # set thread running to False
        self.running = False

    def run(self):
        # connect with device for reading data
        dataport = self.configure_serial()
        try:
            while self.running:
                dataOk, detObj = self.readAndParsePacket(dataport)
                if dataOk:
                    # Store the current frame into frameData
                    timestamp_ns = time.time_ns()
                    self.out_queue.put((timestamp_ns, detObj))
            dataport.write(DATA_STOP_REQUEST)
            dataport.close()
        except Exception as e:
            self.running = False
            dataport.close()
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)
        finally:
            self.stop()


class Cyglidar2DWriterThread(threading.Thread):
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
        self.logger.info(f"Starting writing data from {CYGLIDAR_DEVICE_NAME} sensor...")
        # start
        super(Cyglidar2DWriterThread, self).start()

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
                    ts, data_dict = self.in_queue.get()
                    f.write(f"{ts} | {data_dict}\n")
        except Exception as e:
            self.running = False
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)


class Cyglidar2DDevice(DeviceInterface):
    """
    Device implementation for Cyglidar2DDevice
    """

    def __init__(self, run_config, sensor_queue, logger):
        """
        Initialized required containers for sensing device
        Args:
            run_config: basic config for pipeline run
            logger: Logging object
        """
        super().__init__(run_config, sensor_queue, logger)

        # initialize Cyglidar2DDevice
        self.name = CYGLIDAR_DEVICE_NAME
        self.product_name = CYGLIDAR_PRODUCT_NAME

        self.port_address = None
        self.reader = None
        self.writer = None
        return

    def is_available(self):
        """
        Check if this particular device is available to collect data from
        Returns: True is device is available else False
        """

        ports = serial.tools.list_ports.comports()
        port_address = ''
        for port in ports:
            if port.product == self.product_name:
                port_address = port.device
        if port_address == '':
            self.logger.error("Unable to find device on usb ports.")
            return False
        self.port_address = port_address
        return True

    def startReader(self):
        """
        Start reader thread, should not be called before initialization
        Returns: True if initialization successful, else false
        """
        try:
            self.reader = Cyglidar2DReaderThread(self.sensor_queue,
                                                 self.port_address,
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
                self.write_src = f"{self.run_config['experiment_dir']}/cyglidar3d.csv"
                print(f"Write Source: {self.write_src}")
                self.writer = Cyglidar2DWriterThread(self.sensor_queue,
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
