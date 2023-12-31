"""
This file contains device adapter mapping from device name to device class
"""
# import sys
#
# from sensing.video import LogitechDevice
# from sensing.audio import YetiDevice
#
# from sensing.doppler import Awr1642BoostDevice
# from sensing.lidar import Cyglidar3DDevice, Cyglidar2DDevice, RPLidarDevice
# from sensing.thermalcam import FlirDevice
# from sensing.micarray import Respeaker4MicDevice
#
# available_devices = {
#     'flir': FlirDevice,
#     'awr1642': Awr1642BoostDevice,
#     'respeaker': Respeaker4MicDevice,
#     'rplidar_2d': RPLidarDevice,
#     'cyglidar3d': Cyglidar3DDevice,
#     'cyglidar2d': Cyglidar2DDevice,
#     'logitech': LogitechDevice,
#     'yeti': YetiDevice,
# }
#
#
# def get_device_adapter(device_name):
#     """
#     Get device adapter based on config
#     Args:
#         device_name:
#
#     Returns:
#
#     """
#     if device_name in available_devices.keys():
#         return available_devices[device_name]
#     else:
#         print(f"Device adapter for {device_name} is not available. Exiting...")
#         sys.exit(1)
