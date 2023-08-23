"""
This is base interface class to be implemented for all devices
"""
import threading


class DeviceInterface():
    """Interface design for device object"""

    def __init__(self, run_config, sensor_queue, logger, viz_queue=None):
        """
        Initialized required containers for sensing device
        Args:
            run_config: basic config for pipeline run
            logger: Logging object
        """
        self.run_config = run_config
        self.sensor_queue = sensor_queue
        self.logger = logger
        self.viz_queue = viz_queue
        pass

    def is_available(self):
        """
        Check if this particular device is available to collect data from
        Returns: True is device is available else False
        """
        pass

    def startReader(self):
        """
        Start reader thread, should not be called before initialization
        Returns: True if initialization successful, else false
        """
        pass

    def stopReader(self):
        """
        Gracefully stop reader thread, and destroy device relevant objects
        Returns: True if thread destroyed successfully, else false
        """
        pass

    def startWriter(self):
        """
        Start writer thread, should not be called before initialization
        Returns: True if initialization successful, else false
        """
        pass

    def stopWriter(self):
        """
        Gracefully stop writer thread, and destroy device relevant objects
        Returns: True if thread destroyed successfully, else false
        """
        pass
