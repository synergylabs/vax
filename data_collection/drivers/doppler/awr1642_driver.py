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

# custom libraries
from sensing.deviceInterface import DeviceInterface

# config parameters (not to be changed)
AWR_DEVICE_NAME = 'AWR1642BOOST-ODS'
BASE_CONFIG = f'{Path(__file__).parent}/RangeDopplerHeatmap.cfg'
CLIPORT_ADDR = '/dev/ttyACM0'
CLIPORT_BAUDRATE = 115200
DATAPORT_ADDR = '/dev/ttyACM1'
DATAPORT_BAUDRATE = 921600

# Constants
OBJ_STRUCT_SIZE_BYTES = 12
BYTE_VEC_ACC_MAX_SIZE = 2 ** 15
MMW_OUTPUT_MSG_DETECTED_POINTS = 1
MMW_OUTPUT_MSG_RANGE_PROFILE = 2
MMW_OUTPUT_MSG_NOISE_PROFILE = 3
MMW_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP = 4
MMW_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP = 5
MMW_OUTPUT_MSG_STATS = 6
maxBufferSize = 2 ** 15
tlvHeaderLengthInBytes = 8
pointLengthInBytes = 16
magicWord = [2, 1, 4, 3, 6, 5, 8, 7]


class Awr1642BoostReaderThread(threading.Thread):
    """
    Reader thread for doppler sensing from awr1642boost
    """

    def __init__(self, out_queue, logger):
        threading.Thread.__init__(self)
        self.out_queue = out_queue
        self.logger = logger
        self.running = False

        #
        self.byteBuffer = np.zeros(2 ** 15, dtype='uint8')
        self.byteBufferLength = 0
        self.config_parameters = {}

    def configure_serial(self, configFileName):
        cliport = serial.Serial(CLIPORT_ADDR, CLIPORT_BAUDRATE)

        # Read the configuration file and send it to the board
        config = [line.rstrip('\r\n') for line in open(configFileName)]
        for i in config:
            cliport.write((i + '\n').encode())
            print(i)
            time.sleep(0.01)
        return cliport

    def parse_config_file(self, configFileName):
        configParameters = {}  # empty dict to store config parameters
        # Read the configuration file and send it to the board
        config = [line.rstrip('\r\n') for line in open(configFileName)]
        for i in config:

            # Split the line
            splitWords = i.split(" ")

            # Hard code the number of antennas, change if other configuration is used
            numRxAnt = 4
            numTxAnt = 2
            numVirAnt = 8  # Used for azimuthal angle bins

            # Get the information about the profile configuration
            if "profileCfg" in splitWords[0]:
                startFreq = int(float(splitWords[2]))
                idleTime = int(splitWords[3])
                rampEndTime = float(splitWords[5])
                freqSlopeConst = float(splitWords[8])
                numAdcSamples = int(splitWords[10])
                numAdcSamplesRoundTo2 = 1

                while numAdcSamples > numAdcSamplesRoundTo2:
                    numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2

                digOutSampleRate = int(splitWords[11])

            # Get the information about the frame configuration
            elif "frameCfg" in splitWords[0]:

                chirpStartIdx = int(splitWords[1])
                chirpEndIdx = int(splitWords[2])
                numLoops = int(splitWords[3])
                numFrames = int(splitWords[4])
                framePeriodicity = float(splitWords[5])

        # Combine the read data to obtain the configuration parameters
        numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
        configParameters["numDopplerBins"] = numChirpsPerFrame // numTxAnt
        configParameters["numRangeBins"] = numAdcSamplesRoundTo2
        configParameters["numVirtualAntennas"] = numVirAnt
        configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (
                2 * freqSlopeConst * 1e12 * numAdcSamples)
        configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (
                2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
        configParameters["dopplerResolutionMps"] = 3e8 / (
                2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
        configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (2 * freqSlopeConst * 1e3)
        configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)

        return configParameters

    def readAndParsePacket(self, cliport, dataport):

        # Initialize variables
        magicOK = 0  # Checks if magic number has been read
        dataOK = 0  # Checks if the data has been read correctly
        frameNumber = 0
        detObj = {}

        readBuffer = dataport.read(dataport.in_waiting)
        byteVec = np.frombuffer(readBuffer, dtype='uint8')
        byteCount = len(byteVec)

        # Check that the buffer is not full, and then add the data to the buffer
        if (self.byteBufferLength + byteCount) < maxBufferSize:
            self.byteBuffer[self.byteBufferLength:self.byteBufferLength + byteCount] = byteVec[:byteCount]
            self.byteBufferLength = self.byteBufferLength + byteCount

        # Check that the buffer has some data
        if self.byteBufferLength > 16:

            # Check for all possible locations of the magic word
            possibleLocs = np.where(self.byteBuffer == magicWord[0])[0]

            # Confirm that is the beginning of the magic word and store the index in startIdx
            startIdx = []
            for loc in possibleLocs:
                check = self.byteBuffer[loc:loc + 8]
                if np.all(check == magicWord):
                    startIdx.append(loc)

            # Check that startIdx is not empty
            if startIdx:

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

                # word array to convert 4 bytes to a 32 bit number
                word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

                # Read the total packet length
                totalPacketLen = np.matmul(self.byteBuffer[12:12 + 4], word)

                # Check that all the packet has been read
                if (self.byteBufferLength >= totalPacketLen) and (self.byteBufferLength != 0):
                    magicOK = 1

        # If magicOK is equal to 1 then process the message
        if magicOK:
            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

            # Initialize the pointer index
            idX = 0

            # Read the header
            magicNumber = self.byteBuffer[idX:idX + 8]
            idX += 8
            version = format(np.matmul(self.byteBuffer[idX:idX + 4], word), 'x')
            idX += 4
            totalPacketLen = np.matmul(self.byteBuffer[idX:idX + 4], word)
            idX += 4
            platform = format(np.matmul(self.byteBuffer[idX:idX + 4], word), 'x')
            idX += 4
            frameNumber = np.matmul(self.byteBuffer[idX:idX + 4], word)
            idX += 4
            timeCpuCycles = np.matmul(self.byteBuffer[idX:idX + 4], word)
            idX += 4
            numDetectedObj = np.matmul(self.byteBuffer[idX:idX + 4], word)
            idX += 4
            numTLVs = np.matmul(self.byteBuffer[idX:idX + 4], word)
            idX += 4
            subFrameNumber = np.matmul(self.byteBuffer[idX:idX + 4], word)
            idX += 4

            # Read the TLV messages
            for tlvIdx in range(numTLVs):

                # word array to convert 4 bytes to a 32 bit number
                word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

                # Check the header of the TLV message
                try:
                    tlv_type = np.matmul(self.byteBuffer[idX:idX + 4], word)
                    idX += 4
                    tlv_length = np.matmul(self.byteBuffer[idX:idX + 4], word)
                    idX += 4
                except:
                    continue

                # Read the data depending on the TLV message
                if tlv_type == MMW_OUTPUT_MSG_DETECTED_POINTS:

                    # Initialize the arrays
                    x = np.zeros(numDetectedObj, dtype=np.float32)
                    y = np.zeros(numDetectedObj, dtype=np.float32)
                    z = np.zeros(numDetectedObj, dtype=np.float32)
                    velocity = np.zeros(numDetectedObj, dtype=np.float32)

                    for objectNum in range(numDetectedObj):
                        # Read the data for each object
                        x[objectNum] = self.byteBuffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4
                        y[objectNum] = self.byteBuffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4
                        z[objectNum] = self.byteBuffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4
                        velocity[objectNum] = self.byteBuffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4

                    # Store the data in the detObj dictionary
                    detObj.update({"numObj": numDetectedObj, "x": x, "y": y, "z": z, "velocity": velocity})
                    dataOK = 1
                elif tlv_type == MMW_OUTPUT_MSG_RANGE_PROFILE:
                    # Get the number of bytes to read
                    numBytes = 2 * self.config_parameters["numRangeBins"]

                    # Convert the raw data to int16 array
                    payload = self.byteBuffer[idX:idX + numBytes]
                    idX += numBytes
                    rangeProfile = payload.view(dtype=np.int16)

                    # Some frames have strange values, skip those frames
                    # TO DO: Find why those strange frames happen
                    if np.max(rangeProfile) > 10000:
                        continue
                    # Generate the range array idx for the plot
                    rangeArray = np.array(range(self.config_parameters["numRangeBins"])) * self.config_parameters[
                        "rangeIdxToMeters"]
                    detObj.update(
                        {'rangeArrayIdx': rangeArray, 'rangeProfile': rangeProfile})
                    dataOK = 1
                elif tlv_type == MMW_OUTPUT_MSG_NOISE_PROFILE:
                    # Get the number of bytes to read
                    numBytes = 2 * self.config_parameters["numRangeBins"]

                    # Convert the raw data to int16 array
                    payload = self.byteBuffer[idX:idX + numBytes]
                    idX += numBytes
                    noiseProfile = payload.view(dtype=np.int16)

                    # Some frames have strange values, skip those frames
                    # TO DO: Find why those strange frames happen
                    if np.max(noiseProfile) > 10000:
                        continue

                    # Generate the range array idx for the plot
                    rangeArray = np.array(range(self.config_parameters["numRangeBins"])) * self.config_parameters[
                        "rangeIdxToMeters"]
                    detObj.update(
                        {'rangeArrayIdx': rangeArray, 'noiseProfile': noiseProfile})
                    dataOK = 1
                elif tlv_type == MMW_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP:
                    # Get the number of bytes to read
                    numBytes = 4 * self.config_parameters["numRangeBins"] * self.config_parameters["numVirtualAntennas"]

                    # Convert the raw data to int16 array
                    payload = self.byteBuffer[idX:idX + numBytes]
                    idX += numBytes
                    azimuthRange = payload.view(dtype=np.int16)
                    azimuthRangeImaginaryPart = azimuthRange[::2]
                    azimuthRangeRealPart = azimuthRange[1::2]

                    # Some frames have strange values, skip those frames
                    # TO DO: Find why those strange frames happen
                    if np.max(azimuthRange) > 10000:
                        continue

                    # Convert the azimuth range arrays to a matrix
                    azimuthRangeImaginaryPart = np.reshape(azimuthRangeImaginaryPart,
                                                           (self.config_parameters["numVirtualAntennas"],
                                                            self.config_parameters["numRangeBins"]),
                                                           'F')  # Fortran-like reshape
                    azimuthRangeRealPart = np.reshape(azimuthRangeRealPart,
                                                      (self.config_parameters["numVirtualAntennas"],
                                                       self.config_parameters["numRangeBins"]),
                                                      'F')  # Fortran-like reshape
                    # Generate the range and doppler arrays for the plot
                    rangeArray = np.array(range(self.config_parameters["numRangeBins"])) * self.config_parameters[
                        "rangeIdxToMeters"]
                    detObj.update(
                        {'rangeArrayIdx': rangeArray, 'azimuthRangeReal': azimuthRangeRealPart,
                         'azimuthRangeIm': azimuthRangeImaginaryPart})
                    dataOK = 1
                elif tlv_type == MMW_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP:

                    # Get the number of bytes to read
                    numBytes = 2 * self.config_parameters["numRangeBins"] * self.config_parameters["numDopplerBins"]

                    # Convert the raw data to int16 array
                    payload = self.byteBuffer[idX:idX + numBytes]
                    idX += numBytes
                    rangeDoppler = payload.view(dtype=np.int16)

                    # Some frames have strange values, skip those frames
                    # TO DO: Find why those strange frames happen
                    if np.max(rangeDoppler) > 10000:
                        continue

                    # Convert the range doppler array to a matrix
                    rangeDoppler = np.reshape(rangeDoppler,
                                              (self.config_parameters["numDopplerBins"],
                                               self.config_parameters["numRangeBins"]),
                                              'F')  # Fortran-like reshape
                    rangeDoppler = np.append(rangeDoppler[int(len(rangeDoppler) / 2):],
                                             rangeDoppler[:int(len(rangeDoppler) / 2)], axis=0)

                    # Generate the range and doppler arrays for the plot
                    rangeArray = np.array(range(self.config_parameters["numRangeBins"])) * self.config_parameters[
                        "rangeIdxToMeters"]
                    dopplerArray = np.multiply(
                        np.arange(-self.config_parameters["numDopplerBins"] / 2,
                                  self.config_parameters["numDopplerBins"] / 2),
                        self.config_parameters["dopplerResolutionMps"])
                    detObj.update(
                        {'rangeArrayIdx': rangeArray, 'rangeDopplerIdx': dopplerArray,
                         'rangeDopplerMatrix': rangeDoppler})
                    dataOK = 1
                elif tlv_type == MMW_OUTPUT_MSG_STATS:
                    # word array to convert 4 bytes to a 32 bit number
                    word = [1, 2 ** 8, 2 ** 16, 2 ** 24]
                    stats_dict = {}
                    # Check the header of the TLV message
                    stats_dict['interframe_processing_time'] = np.matmul(self.byteBuffer[idX:idX + 4], word)
                    idX += 4
                    stats_dict['output_transmit_time'] = np.matmul(self.byteBuffer[idX:idX + 4], word)
                    idX += 4
                    stats_dict['interframe_processing_margin'] = np.matmul(self.byteBuffer[idX:idX + 4], word)
                    idX += 4
                    stats_dict['interchirp_processing_margin'] = np.matmul(self.byteBuffer[idX:idX + 4], word)
                    idX += 4
                    stats_dict['activeframe_cpu_load'] = np.matmul(self.byteBuffer[idX:idX + 4], word)
                    idX += 4
                    stats_dict['interframe_cpu_load'] = np.matmul(self.byteBuffer[idX:idX + 4], word)
                    idX += 4

                    detObj.update(stats_dict)
                    dataOK = 1
                else:
                    idX = idX + tlv_length  # Unknown TLV: Skip to next tlv

            # Remove already processed data
            if idX > 0 and self.byteBufferLength > idX:
                shiftSize = totalPacketLen

                self.byteBuffer[:self.byteBufferLength - shiftSize] = self.byteBuffer[shiftSize:self.byteBufferLength]
                self.byteBuffer[self.byteBufferLength - shiftSize:] = np.zeros(
                    len(self.byteBuffer[self.byteBufferLength - shiftSize:]),
                    dtype='uint8')
                self.byteBufferLength = self.byteBufferLength - shiftSize

                # Check that there are no errors with the buffer length
                if self.byteBufferLength < 0:
                    self.byteBufferLength = 0

        return dataOK, frameNumber, detObj

    def start(self):

        # parse config parameters
        self.config_parameters = self.parse_config_file(BASE_CONFIG)

        # mark this is running
        self.running = True

        # start
        super(Awr1642BoostReaderThread, self).start()

    def stop(self):
        # set thread running to False
        self.running = False

    def run(self):
        # connect with device for reading data
        cliport = self.configure_serial(BASE_CONFIG)
        dataport = serial.Serial(DATAPORT_ADDR, DATAPORT_BAUDRATE)
        try:
            while self.running:
                dataOk, frameNumber, detObj = self.readAndParsePacket(cliport, dataport)
                if dataOk:
                    # Store the current frame into frameData
                    timestamp_ns = time.time_ns()
                    self.out_queue.put((timestamp_ns, detObj))
            cliport.write(('sensorStop\n').encode())
            cliport.close()
            dataport.close()
        except Exception as e:
            self.running = False
            cliport.close()
            dataport.close()
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)
        finally:
            self.stop()


class Awr1642BoostWriterThread(threading.Thread):
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
        self.logger.info(f"Starting writing data from {AWR_DEVICE_NAME} sensor...")
        # start
        super(Awr1642BoostWriterThread, self).start()

    def stop(self):
        # destroy device relevant object
        # set thread running to False
        self.running = False

    def run(self):
        is_header_set = False
        try:
            with open(self.write_src,'w') as f:
                while self.running:
                    if not is_header_set:
                        f.write("ts | data\n")
                    ts, data_dict = self.in_queue.get()
                    f.write(f"{ts} | {data_dict}\n")
        except Exception as e:
            self.running = False
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)


class Awr1642BoostDevice(DeviceInterface):
    """
    Device implementation for Awr1642BoostDevice
    """

    def __init__(self, run_config, sensor_queue, logger):
        """
        Initialized required containers for sensing device
        Args:
            run_config: basic config for pipeline run
            logger: Logging object
        """
        super().__init__(run_config, sensor_queue, logger)

        # initialize Awr1642BoostDevice
        self.name = AWR_DEVICE_NAME
        self.reader = None
        self.writer = None
        return

    def is_available(self):
        """
        Check if this particular device is available to collect data from
        Returns: True is device is available else False
        """
        try:
            test_cliport = serial.Serial(CLIPORT_ADDR, CLIPORT_BAUDRATE)
            test_dataport = serial.Serial(DATAPORT_ADDR, DATAPORT_BAUDRATE)
            test_cliport.close()
            test_dataport.close()
            return True
        except:
            self.logger.error(f"Unable to reach {AWR_DEVICE_NAME}: {traceback.format_exc()}")
            return False

    def startReader(self):
        """
        Start reader thread, should not be called before initialization
        Returns: True if initialization successful, else false
        """
        try:
            self.reader = Awr1642BoostReaderThread(self.sensor_queue,
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
                self.write_src = f"{self.run_config['experiment_dir']}/doppler.csv"
                print(f"Write Source: {self.write_src}")
                self.writer = Awr1642BoostWriterThread(self.sensor_queue,
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
