"""
Template for creating new device driver
Author: Prasoon Patidar
Created at: 28th Sept 2022
"""

# basic libraries
import threading
import queue
import logging
import traceback
import sys

# custom libraries
from sensing.deviceInterface import DeviceInterface

#config parameters at device level (not to be changed)


class TemplateReaderThread(threading.Thread):
    """
    Reader thread for video sensing from logitech camera
    """

    def __init__(self, out_queue, logger):
        threading.Thread.__init__(self)
        self.queue = queue
        self.logger = logger
        self.running = False

    def start(self):
        # connect with device for reading data
        ...

        # mark this is running
        self.is_running = True

        # start
        super(TemplateReaderThread, self).start()

    def stop(self):
        # destroy device relevant object

        # set thread running to False
        self.running = False

    def run(self):
        try:
            while self.running:
                ...
        except Exception as e:
            self.running = False
            self.logger.info("Exception thrown")
            traceback.print_exc(file=sys.stdout)
        finally:
            self.stop()


class TemplateDevice(DeviceInterface):
    """
    Device implementation for logitech camera Video recording
    """

    def __init__(self, run_config, logger):
        """
        Initialized required containers for sensing device
        Args:
            run_config: basic config for pipeline run
            logger: Logging object
        """
        super().__init__(run_config, logger)

        # initialize TemplateDevice
        return

    def is_available(self):
        """
        Check if this particular device is available to collect data from
        Returns: True is device is available else False
        """
        return False

    def initializeReader(self, out_queue):
        """
        Setup reader thread for given device with given output queue
        Returns: None
        """
        return None

    def startReader(self):
        """
        Start reader thread, should not be called before initialization
        Returns: True if initialization successful, else false
        """
        return True

    def stopReader(self):
        """
        Gracefully stop reader thread, and destroy device relevant objects
        Returns: True if thread destroyed successfully, else false
        """
        return True
