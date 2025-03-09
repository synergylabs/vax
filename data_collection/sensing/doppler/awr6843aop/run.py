import serial
import time
import numpy as np
# import matplotlib.pyplot as plt

# Change the configuration file name
configFileName = './RangeDopplerAOP.cfg'
# configFileName = 'configs/30fps_config.cfg'
# configFileName = './RangeDopplerHeatmap.cfg'

CLIport = {}
Dataport = {}
byteBuffer = np.zeros(2 ** 15, dtype='uint8')
byteBufferLength = 0


# ------------------------------------------------------------------

# Function to configure the serial ports and send the data from
# the configuration file to the radar
def serialConfig(configFileName):
    global CLIport
    global Dataport
    # Open the serial ports for the configuration and the data ports

    # Raspberry pi
    # CLIport = serial.Serial('/dev/ttyACM0', 115200)
    # Dataport = serial.Serial('/dev/ttyACM1', 921600)

    # Windows
    CLIport = serial.Serial('/dev/ttyUSB0', 115200)
    Dataport = serial.Serial('/dev/ttyUSB1', 921600)

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIport.write((i + '\n').encode())
        print(i)
        time.sleep(0.01)

    return CLIport, Dataport

def serialConfigCustom(cliport_address, dataport_address, configFileName):
    global CLIport
    global Dataport
    # Open the serial ports for the configuration and the data ports

    # Raspberry pi
    # CLIport = serial.Serial('/dev/ttyACM0', 115200)
    # Dataport = serial.Serial('/dev/ttyACM1', 921600)

    # Windows
    CLIport = serial.Serial(cliport_address, 115200)
    Dataport = serial.Serial(dataport_address, 921600)

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIport.write((i + '\n').encode())
        print(i)
        time.sleep(0.01)

    return CLIport, Dataport


# ------------------------------------------------------------------

# Function to parse the data inside the configuration file
def parseConfigFile(configFileName):
    configParameters = {}  # Initialize an empty dictionary to store the configuration parameters

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:

        # Split the line
        splitWords = i.split(" ")

        # Hard code the number of antennas, change if other configuration is used
        numRxAnt = 4
        numTxAnt = 2
        numVirAnt = 8 # Used for azimuthal angle bins

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


# ------------------------------------------------------------------

# Funtion to read and parse the incoming data
def readAndParseData1642Boost(Dataport, configParameters):
    global byteBuffer, byteBufferLength

    # Constants
    OBJ_STRUCT_SIZE_BYTES = 12
    BYTE_VEC_ACC_MAX_SIZE = 2 ** 15
    MMWDEMO_OUTPUT_MSG_DETECTED_POINTS = 1
    MMWDEMO_OUTPUT_MSG_RANGE_PROFILE = 2
    MMWDEMO_OUTPUT_MSG_NOISE_PROFILE = 3
    MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP = 4
    MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP = 5
    MMWDEMO_OUTPUT_MSG_STATS = 6
    maxBufferSize = 2 ** 15
    tlvHeaderLengthInBytes = 8
    pointLengthInBytes = 16
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]

    # Initialize variables
    magicOK = 0  # Checks if magic number has been read
    dataOK = 0  # Checks if the data has been read correctly
    frameNumber = 0
    detObj = {}

    readBuffer = Dataport.read(Dataport.in_waiting)
    byteVec = np.frombuffer(readBuffer, dtype='uint8')
    byteCount = len(byteVec)

    # Check that the buffer is not full, and then add the data to the buffer
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
        byteBufferLength = byteBufferLength + byteCount

    # Check that the buffer has some data
    if byteBufferLength > 16:

        # Check for all possible locations of the magic word
        possibleLocs = np.where(byteBuffer == magicWord[0])[0]

        # Confirm that is the beginning of the magic word and store the index in startIdx
        startIdx = []
        for loc in possibleLocs:
            check = byteBuffer[loc:loc + 8]
            if np.all(check == magicWord):
                startIdx.append(loc)

        # Check that startIdx is not empty
        if startIdx:

            # Remove the data before the first start index
            if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                byteBuffer[:byteBufferLength - startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                byteBuffer[byteBufferLength - startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength - startIdx[0]:]),
                                                                       dtype='uint8')
                byteBufferLength = byteBufferLength - startIdx[0]

            # Check that there have no errors with the byte buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

            # Read the total packet length
            totalPacketLen = np.matmul(byteBuffer[12:12 + 4], word)

            # Check that all the packet has been read
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1

    # If magicOK is equal to 1 then process the message
    if magicOK:
        # word array to convert 4 bytes to a 32 bit number
        word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

        # Initialize the pointer index
        idX = 0

        # Read the header
        magicNumber = byteBuffer[idX:idX + 8]
        idX += 8
        version = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
        idX += 4
        totalPacketLen = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        platform = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
        idX += 4
        frameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        timeCpuCycles = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        numDetectedObj = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        numTLVs = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        subFrameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4

        # Read the TLV messages
        for tlvIdx in range(numTLVs):

            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

            # Check the header of the TLV message
            try:
                tlv_type = np.matmul(byteBuffer[idX:idX + 4], word)
                idX += 4
                tlv_length = np.matmul(byteBuffer[idX:idX + 4], word)
                idX += 4
            except:
                continue

            # Read the data depending on the TLV message
            if tlv_type == MMWDEMO_OUTPUT_MSG_DETECTED_POINTS:

                # Initialize the arrays
                x = np.zeros(numDetectedObj, dtype=np.float32)
                y = np.zeros(numDetectedObj, dtype=np.float32)
                z = np.zeros(numDetectedObj, dtype=np.float32)
                velocity = np.zeros(numDetectedObj, dtype=np.float32)

                for objectNum in range(numDetectedObj):
                    # Read the data for each object
                    x[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                    idX += 4
                    y[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                    idX += 4
                    z[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                    idX += 4
                    velocity[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                    idX += 4

                # Store the data in the detObj dictionary
                detObj.update({"numObj": numDetectedObj, "x": x, "y": y, "z": z, "velocity": velocity})
                dataOK = 1
            elif tlv_type == MMWDEMO_OUTPUT_MSG_RANGE_PROFILE:
                # Get the number of bytes to read
                numBytes = 2 * configParameters["numRangeBins"]

                # Convert the raw data to int16 array
                payload = byteBuffer[idX:idX + numBytes]
                idX += numBytes
                rangeProfile = payload.view(dtype=np.int16)

                # Some frames have strange values, skip those frames
                # TO DO: Find why those strange frames happen
                if np.max(rangeProfile) > 10000:
                    continue
                # Generate the range array idx for the plot
                rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]
                detObj.update(
                    {'rangeArrayIdx': rangeArray, 'rangeProfile': rangeProfile})
                dataOK = 1
            elif tlv_type == MMWDEMO_OUTPUT_MSG_NOISE_PROFILE:
                # Get the number of bytes to read
                numBytes = 2 * configParameters["numRangeBins"]

                # Convert the raw data to int16 array
                payload = byteBuffer[idX:idX + numBytes]
                idX += numBytes
                noiseProfile = payload.view(dtype=np.int16)

                # Some frames have strange values, skip those frames
                # TO DO: Find why those strange frames happen
                if np.max(noiseProfile) > 10000:
                    continue

                # Generate the range array idx for the plot
                rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]
                detObj.update(
                    {'rangeArrayIdx': rangeArray, 'noiseProfile': noiseProfile})
                dataOK = 1
            elif tlv_type==MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP:
                # Get the number of bytes to read
                numBytes = 4 * configParameters["numRangeBins"] * configParameters["numVirtualAntennas"]

                # Convert the raw data to int16 array
                payload = byteBuffer[idX:idX + numBytes]
                idX += numBytes
                azimuthRange = payload.view(dtype=np.int16)
                azimuthRangeImaginaryPart =  azimuthRange[::2]
                azimuthRangeRealPart = azimuthRange[1::2]


                # Some frames have strange values, skip those frames
                # TO DO: Find why those strange frames happen
                if np.max(azimuthRange) > 10000:
                    continue

                # Convert the azimuth range arrays to a matrix
                azimuthRangeImaginaryPart = np.reshape(azimuthRangeImaginaryPart,
                                          (configParameters["numVirtualAntennas"], configParameters["numRangeBins"]),
                                          'F')  # Fortran-like reshape
                azimuthRangeRealPart = np.reshape(azimuthRangeRealPart,
                                          (configParameters["numVirtualAntennas"], configParameters["numRangeBins"]),
                                          'F')  # Fortran-like reshape
                # Generate the range and doppler arrays for the plot
                rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]
                detObj.update(
                    {'rangeArrayIdx': rangeArray, 'azimuthRangeReal': azimuthRangeRealPart, 'azimuthRangeIm': azimuthRangeImaginaryPart})
                dataOK = 1
            elif tlv_type == MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP:

                # Get the number of bytes to read
                numBytes = 2 * configParameters["numRangeBins"] * configParameters["numDopplerBins"]

                # Convert the raw data to int16 array
                payload = byteBuffer[idX:idX + numBytes]
                idX += numBytes
                rangeDoppler = payload.view(dtype=np.int16)

                # Some frames have strange values, skip those frames
                # TO DO: Find why those strange frames happen
                if np.max(rangeDoppler) > 10000:
                    continue

                # Convert the range doppler array to a matrix
                rangeDoppler = np.reshape(rangeDoppler,
                                          (configParameters["numDopplerBins"], configParameters["numRangeBins"]),
                                          'F')  # Fortran-like reshape
                rangeDoppler = np.append(rangeDoppler[int(len(rangeDoppler) / 2):],
                                         rangeDoppler[:int(len(rangeDoppler) / 2)], axis=0)

                # Generate the range and doppler arrays for the plot
                rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]
                dopplerArray = np.multiply(
                    np.arange(-configParameters["numDopplerBins"] / 2, configParameters["numDopplerBins"] / 2),
                    configParameters["dopplerResolutionMps"])
                detObj.update(
                    {'rangeArrayIdx': rangeArray, 'rangeDopplerIdx': dopplerArray, 'rangeDopplerMatrix': rangeDoppler})
                dataOK = 1
            elif tlv_type==MMWDEMO_OUTPUT_MSG_STATS:
                # word array to convert 4 bytes to a 32 bit number
                word = [1, 2 ** 8, 2 ** 16, 2 ** 24]
                stats_dict = {}
                # Check the header of the TLV message
                stats_dict['interframe_processing_time'] = np.matmul(byteBuffer[idX:idX + 4], word)
                idX += 4
                stats_dict['output_transmit_time'] = np.matmul(byteBuffer[idX:idX + 4], word)
                idX += 4
                stats_dict['interframe_processing_margin'] = np.matmul(byteBuffer[idX:idX + 4], word)
                idX += 4
                stats_dict['interchirp_processing_margin'] = np.matmul(byteBuffer[idX:idX + 4], word)
                idX += 4
                stats_dict['activeframe_cpu_load'] = np.matmul(byteBuffer[idX:idX + 4], word)
                idX += 4
                stats_dict['interframe_cpu_load'] = np.matmul(byteBuffer[idX:idX + 4], word)
                idX += 4

                detObj.update(stats_dict)
                dataOK = 1
            else:
                idX = idX + tlv_length  # Unknown TLV: Skip to next tlv

        # Remove already processed data
        if idX > 0 and byteBufferLength > idX:
            shiftSize = totalPacketLen

            byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
            byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]),
                                                                 dtype='uint8')
            byteBufferLength = byteBufferLength - shiftSize

            # Check that there are no errors with the buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

    return dataOK, frameNumber, detObj

def startDoppler(cliport_address, dataport_address, cfg_file):
    # Configurate the serial port
    CLIport, Dataport = serialConfigCustom(cliport_address, dataport_address, cfg_file)

    # Get the configuration parameters from the configuration file
    configParameters = parseConfigFile(cfg_file)

    return CLIport, Dataport, configParameters

def stopDoppler(CLIport, Dataport):
    CLIport.write(('sensorStop\n').encode())
    CLIport.close()
    Dataport.close()
    return None

# -------------------------    MAIN: Code sample to interface with sensor   -----------------------------------------
if __name__=='__main__':
    # Configurate the serial port
    CLIport, Dataport = serialConfig(configFileName)

    # Get the configuration parameters from the configuration file
    configParameters = parseConfigFile(configFileName)

    # Main loop
    detObj = {}
    frameData = {}
    numFrames=0
    st_time = time.time()
    while True:
        try:
            dataOk, frameNumber, detObj = readAndParseData1642Boost(Dataport, configParameters)

            if dataOk:
                # Store the current frame into frameData
                timestamp_ns = time.time()
                frameData[timestamp_ns] = detObj
                numFrames+=1
                if time.time()-st_time > 10.0:
                    print(numFrames)
                    numFrames=0
                    frameData = {}
                    st_time=time.time()
        # Stop the program and close everything if Ctrl + c is pressed
        except KeyboardInterrupt:
            CLIport.write(('sensorStop\n').encode())
            CLIport.close()
            Dataport.close()
            break
