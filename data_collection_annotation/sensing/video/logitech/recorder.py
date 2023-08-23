'''
This file contains thread responsible for storing depth information from video
Developer: Prasoon Patidar
Created: 5th March, 2022
'''

# basic libraries
import os
import threading
import cv2

# Custom libraries
from sensing.video.logitech.config import Config


class VideoRecorderThread(threading.Thread):
    def __init__(self, rgb_queue, start_timestamp, logger, video_prefix=''):
        threading.Thread.__init__(self)

        self.input_queue = rgb_queue
        self.max_duration = Config.max_duration_per_video
        self.logger = logger

        if not os.path.exists(Config.data_folder):
            os.makedirs(Config.data_folder)

        # initialize output video config
        self.video_width = Config.video_width
        self.video_height = Config.video_height

        self.video_fps = Config.fps
        self.video_prefix = video_prefix
        self.video_file = f'{Config.data_folder}/{video_prefix}_{start_timestamp}.mkv'
        # self.video_format = cv2.VideoWriter_fourcc(*f'{Config.video_codec}')
        self.video_format = cv2.VideoWriter_fourcc(*'MJPG')

        # output containers
        self.video_out = None
        self.max_frames_per_video = self.max_duration * self.video_fps * 60
        self.current_frame_number = 0
        self.is_running = False

        # for debugging purposes
        # self.window_name = 'rgbCam'
        # cv2.namedWindow(self.window_name)

    def start(self):

        # create video output buffer
        self.video_out = cv2.VideoWriter(self.video_file, self.video_format, float(self.video_fps),
                                         (self.video_width, self.video_height))
        self.is_running = True
        super(VideoRecorderThread, self).start()

    def renew_video(self, timestamp):

        # release older video
        self.video_out.release()

        # create new video based on timestamp of next frame and reset current frame number
        self.video_file = f'{Config.data_folder}/{self.video_prefix}_{timestamp}.avi'
        self.video_out = cv2.VideoWriter(self.video_file, self.video_format, float(self.video_fps),
                                         (self.video_width, self.video_height))
        self.current_frame_number = 0

    def stop(self):
        # release current video
        self.video_out.release()

        # destroy all cv2 windows
        cv2.destroyAllWindows()

        # set thread running to false
        self.is_running = False

    def run(self):

        # run till thread is running
        while self.is_running:
            # run till this video exhausts
            while self.current_frame_number < self.max_frames_per_video:
                frame_time, frame = self.input_queue.get()
                frame = cv2.rectangle(frame, (0,frame.shape[1]), (frame.shape[0],frame.shape[1]//2+135), (0,0,0), -1)
                frame = cv2.putText(frame, f"{frame_time}",
                                    (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255))
                self.video_out.write(frame)
                self.current_frame_number += 1

                # for debugging purposes
                # cv2.imshow(self.window_name, frame)
                # if cv2.waitKey(1) == ord('q'):
                #     break

            frame_time, frame = self.input_queue.get()
            self.renew_video(frame_time)
