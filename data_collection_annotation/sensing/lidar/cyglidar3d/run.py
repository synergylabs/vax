import sys

import serial
import time
import numpy as np
from serial.tools import list_ports



def readAndParseDataCygbot(Dataport):
    global byteBuffer, byteBufferLength

    # Constants
    RESPONSE_DEVICE_INFO = np.frombuffer(b"\x10", dtype='uint8')[0]
    RESPONSE_2D_MODE = np.frombuffer(b"\x01", dtype='uint8')[0]
    RESPONSE_3D_MODE = np.frombuffer(b"\x08", dtype='uint8')[0]
    maxBufferSize = 2 ** 15
    headerLengthInBytes = 1
    magicWord = [90, 119, 255]
    word_3byte = [2 ** 16, 2 ** 8, 1]

    # Initialize variables
    magicOK = 0  # Checks if magic number has been read
    dataOK = 0  # Checks if the data has been read correctly
    frameNumber = 0
    detObj = {}
    startIdx = None

    readBuffer = Dataport.read(Dataport.in_waiting)
    byteVec = np.frombuffer(readBuffer, dtype='uint8')
    byteCount = len(byteVec)
    # print("bytes read:",byteCount)

    # Check that the buffer is not full, and then add the data to the buffer
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
        byteBufferLength = byteBufferLength + byteCount

    # Check that the buffer has some data
    if byteBufferLength > 3:

        # Check for all possible locations of the magic word
        possibleLocs = np.where(byteBuffer == magicWord[0])[0]

        # Confirm that is the beginning of the magic word and store the index in startIdx
        startIdx = []
        for loc in possibleLocs:
            check = byteBuffer[loc:loc + 3]
            if np.all(check == magicWord):
                startIdx.append(loc)

    # Check that startIdx is not empty
    if startIdx:
        # print("start found at location",startIdx[0])
        # Remove the data before the first start index
        if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
            byteBuffer[:byteBufferLength - startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
            byteBuffer[byteBufferLength - startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength - startIdx[0]:]),
                                                                   dtype='uint8')
            byteBufferLength = byteBufferLength - startIdx[0]

        # Check that there have no errors with the byte buffer length
        if byteBufferLength < 0:
            byteBufferLength = 0

        # word array to convert 2 bytes to a 16 bit number
        word = [1, 2 ** 8]

        # Read the total packet length
        totalPacketLen = np.matmul(byteBuffer[3:3 + 2], word)

        # Check that all the packet has been read
        if (byteBufferLength >= totalPacketLen + 6) and (byteBufferLength != 0):
            magicOK = 1

    # If magicOK is equal to 1 then process the message
    if magicOK:
        # word array to convert 2 bytes to a 16 bit number
        word = [1, 2 ** 8]

        # Initialize the pointer index
        idX = 0

        # Read the header
        magicNumber = byteBuffer[idX:idX + 3]
        idX += 3
        totalPacketLen = np.matmul(byteBuffer[idX:idX + 2], word)
        idX += 2
        payload_hdr = byteBuffer[idX]
        idX += 1

        # Version info response
        if payload_hdr == RESPONSE_DEVICE_INFO:
            firm_version_hsb = byteBuffer[idX]
            idX += 1
            firm_version_msb = byteBuffer[idX]
            idX += 1
            firm_version_lsb = byteBuffer[idX]
            idX += 1
            hardware_version_hsb = byteBuffer[idX]
            idX += 1
            hardware_version_msb = byteBuffer[idX]
            idX += 1
            hardware_version_lsb = byteBuffer[idX]
            idX += 1
            detObj['firmware_version'] = f'{firm_version_hsb}.{firm_version_msb}.{firm_version_lsb}'
            detObj['hardware_version'] = f'{hardware_version_hsb}.{hardware_version_msb}.{hardware_version_lsb}'

            checksum = byteBuffer[idX]
            idX += 1
            dataOK = 1

        if payload_hdr == RESPONSE_2D_MODE:
            info_2d_hdr = np.arange(-60, +60.5, 0.75)
            info_2d_size = info_2d_hdr.shape[0]
            info_2d = np.full(info_2d_size, fill_value=-1, dtype='uint16')
            for i in range(0, info_2d_size):
                info_2d[i] = np.matmul(byteBuffer[idX:idX + 2], word)
                idX += 2
            checksum = byteBuffer[idX]
            idX += 1
            dataOK = 1
            detObj.update({'hdr_2d':info_2d_hdr,'arr_2d':info_2d})

        if payload_hdr == RESPONSE_3D_MODE:
            info_3d_col_hdr = np.linspace(-60,60,160)
            info_3d_row_hdr = np.linspace(-65/2,65/2,60)
            info_3d_row_size, info_3d_col_size = info_3d_row_hdr.shape[0], info_3d_col_hdr.shape[0]
            info_3d = np.full((info_3d_row_size, info_3d_col_size), fill_value=-1, dtype='uint16')
            for row_idx in range(info_3d_row_size):
                for col_idx in range(0, info_3d_col_size, 2):
                    bit_24_val = np.matmul(byteBuffer[idX:idX + 3], word_3byte)
                    idX += 3
                    info_3d[row_idx][col_idx] = bit_24_val // 2 ** 12
                    info_3d[row_idx][col_idx + 1] = bit_24_val % 2 ** 12
            checksum = byteBuffer[idX]
            idX += 1
            dataOK = 1
            detObj.update({'mat_3d': info_3d, 'row_hdr': info_3d_row_hdr, 'col_hdr':info_3d_col_hdr})

        # Remove already processed data
        if idX > 0 and byteBufferLength >= idX:
            shiftSize = totalPacketLen + 6

            byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
            byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]),
                                                                 dtype='uint8')
            byteBufferLength = byteBufferLength - shiftSize

            # Check that there are no errors with the buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

    return dataOK, detObj


def start3DLidar():
    # constants for read function
    byteBuffer = np.zeros(2 ** 15, dtype='uint8')
    byteBufferLength = 0

    product_name = 'USB-Serial Controller D'
    ports = serial.tools.list_ports.comports()
    port_address = ''
    for port in ports:
        if port.product==product_name:
            port_address = port.device
    if port_address=='':
        print("Device not found. Exiting.")
        sys.exit(1)
    # Set baud rate to lower value
    serialObj = serial.Serial(port_address)
    serialObj.baudrate = 250000  # set Baud rate to 9600
    serialObj.bytesize = 8  # Number of data bits = 8
    serialObj.parity = 'N'  # No parity
    serialObj.stopbits = 1  # Number of Stop bits = 1
    setbaudrate_req = b"\x5A\x77\xFF\x02\x00\x12\x55\x45"
    # setbaudrate_req = b"\x5A\x77\xFF\x02\x00\x12\x77\x67"
    serialObj.write(setbaudrate_req)
    serialObj.close()

    serialObj = serial.Serial(port_address)
    serialObj.baudrate = 250000  # set Baud rate to 9600
    serialObj.bytesize = 8  # Number of data bits = 8
    serialObj.parity = 'N'  # No parity
    serialObj.stopbits = 1  # Number of Stop bits = 1
    data_3d_request = b"\x5A\x77\xFF\x02\x00\x08\x00\x0A"
    serialObj.write(data_3d_request)
    return serialObj

def stop3DLidar(serialObj):
    serialObj.write(b'\x5A\x77\xFF\x02\x00\x02\x00\x00')
    serialObj.close()
    return None


# constants for read function
byteBuffer = np.zeros(2 ** 15, dtype='uint8')
byteBufferLength = 0

# -------------------------    MAIN: Code sample to interface with sensor   -----------------------------------------
if __name__=='__main__':

    product_name = 'USB-Serial Controller D'
    ports = serial.tools.list_ports.comports()
    port_address = ''
    for port in ports:
        if port.product==product_name:
            port_address = port.device
    if port_address=='':
        print("Device not found. Exiting.")
        sys.exit(1)
    #
    serialObj = serial.Serial(port_address)
    serialObj.baudrate = 3000000  # set Baud rate to 9600
    serialObj.bytesize = 8  # Number of data bits = 8
    serialObj.parity = 'N'  # No parity
    serialObj.stopbits = 1  # Number of Stop bits = 1
    print("Initialized Serial Object")
    detObj = {}
    frameData = {}
    numFrames = 0
    st_time = time.time()
    # send request to read device info
    # get_info_request =b"\x5A\x77\xFF\x02\x00\x10\x00\x12"
    # serialObj.write(get_info_request)

    # data_2d_request = b"\x5A\x77\xFF\x02\x00\x01\x00\x03"
    # serialObj.write(data_2d_request)
    data_3d_request = b"\x5A\x77\xFF\x02\x00\x08\x00\x0A"
    serialObj.write(data_3d_request)
    # data_all_request = b"\x5A\x77\xFF\x02\x00\x07\x00\x05"
    # serialObj.write(data_all_request)

    print("Sent Device Info Request")
    # time.sleep(1)
    while True:
        try:
            dataOk, detObj = readAndParseDataCygbot(serialObj)

            if dataOk:
                # Store the current frame into frameData
                timestamp_ns = time.time()
                frameData[timestamp_ns] = detObj
                numFrames += 1
                if time.time() - st_time > 10.0:
                    print(numFrames)
                    numFrames = 0
                    frameData = {}
                    st_time = time.time()
        # Stop the program and close everything if Ctrl + c is pressed
        except KeyboardInterrupt:
            serialObj.write(b'\x5A\x77\xFF\x02\x00\x02\x00\x00')
            serialObj.close()
            break
