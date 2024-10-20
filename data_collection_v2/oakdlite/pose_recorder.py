'''
This file contains thread responsible for storing depth information from video
Developer: Kevin Xie
Created: 6th Nov, 2022
'''

# basic libraries
import os
import threading
import numpy as np
import csv
import base64
import pickle
from datetime import datetime
openPose_to_poseNet = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

class PoseRecorderThread(threading.Thread):
    def __init__(self, pose_queue, logger, write_folder, Config,  file_prefix='pose'):
        threading.Thread.__init__(self)

        self.input_queue = pose_queue
        self.logger = logger
        self.config = Config
        self.write_folder = write_folder
        if not os.path.exists(self.write_folder):
            os.makedirs(self.write_folder)
        self.max_frames_per_video = self.config.max_duration_per_video * self.config.pose_fps * 60

        #initialize output video config
        if Config.downscaleColor: #video downscaled to 720p
            self.video_width = 1280
            self.video_height = 720
        else: # video recorded at 1080p
            self.video_width = 1920
            self.video_height = 1080

        self.file_prefix=file_prefix
        curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_path = f'{self.write_folder}/{file_prefix}_{curr_time}.csv'
        # output containers
        self.out_file = None
        # self.csv_out = None
        self.current_frame_number = 0
        self.running=False

        # for debugging purposes
        # self.window_name = 'depthCam'
        # cv2.namedWindow(self.window_name)


    def start(self):

        # create csv output buffer
        self.out_file = open(self.out_path, 'w', newline='')
        # self.csv_out = csv.writer(self.out_file)
        self.running = True
        super(PoseRecorderThread, self).start()

    def renew_file(self, timestamp):

        # release older csv
        self.out_file.close()

        # create new csv based on timestamp of next frame and reset current frame number
        curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_path = f'{self.write_folder}/{self.file_prefix}_{curr_time}.csv'
        self.out_file = open(self.out_path, 'w')
        # self.csv_out = csv.writer(self.out_file)
        self.current_frame_number = 0


    def stop(self):
        # release current csv
        # self.out_file.close()

        # set thread running to false
        self.running=False


    def run(self):

        #run till thread is running
        while self.running:
            #run till this video exhausts
            while self.current_frame_number<self.max_frames_per_video:
                if self.running:
                    frame_time, frame = self.input_queue.get()
                    if frame_time is None:
                        self.running = False
                        continue
                    encoded_data = base64.encodebytes(pickle.dumps(frame)).decode()
                    self.out_file.write(f"{frame_time} | {encoded_data} ||")
                    # # print(f"{self.out_file} {self.csv_out} {frame}")
                    # self.csv_out.writerow([frame_time, np.array2string(frame, separator=', ', suppress_small=True)])
                    self.current_frame_number += 1
                    self.out_file.flush()
                else:
                    self.out_file.close()
                    break
            if self.running:
                # frame_time, frame = self.input_queue.get()
                self.renew_file()




