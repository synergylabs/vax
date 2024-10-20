'''
This file contains thread responsible for storing depth information from video
Developer: Prasoon Patidar
Created: 5th March, 2022
'''

# basic libraries
import os
import threading
import cv2
from datetime import datetime
class DepthRecorderThread(threading.Thread):
    def __init__(self, depth_queue, logger, write_folder, Config, video_prefix='depth'):
        threading.Thread.__init__(self)

        self.input_queue = depth_queue
        self.logger = logger
        self.config = Config
        self.write_folder = write_folder
        if not os.path.exists(self.write_folder):
            os.makedirs(self.write_folder)
        self.max_frames_per_video = self.config.max_duration_per_video * self.config.depth_fps * 60

        #initialize output video config
        if Config.downscaleColor: #video downscaled to 720p
            self.video_width = 1280
            self.video_height = 720
        else: # video recorded at 1080p
            self.video_width = 1920
            self.video_height = 1080


        self.video_prefix=video_prefix
        curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_file = f'{self.write_folder}/{video_prefix}_{curr_time}.mkv'
        # self.video_format = cv2.VideoWriter_fourcc(*f'{Config.video_codec}')
        self.video_format = cv2.VideoWriter_fourcc(*f'{self.config.video_codec}')

        # output containers
        self.video_out = None
        self.current_frame_number = 0
        self.running=False

        # for debugging purposes
        # self.window_name = 'depthCam'
        # cv2.namedWindow(self.window_name)


    def start(self):

        # create video output buffer
        self.video_out =cv2.VideoWriter(self.video_file, self.video_format, float(self.config.depth_fps),
                        (self.video_width, self.video_height))
        self.running = True
        super(DepthRecorderThread, self).start()

    def renew_video(self):

        # release older video
        self.video_out.release()

        # create new video based on timestamp of next frame and reset current frame number
        curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_file = f'{self.write_folder}/{self.video_prefix}_{curr_time}.mkv'
        self.video_out = cv2.VideoWriter(self.video_file, self.video_format, float(self.config.depth_fps),
                                         (self.video_width, self.video_height))
        self.current_frame_number = 0


    def stop(self):
        # release current video
        # self.video_out.release()

        # destroy all cv2 windows
        # cv2.destroyAllWindows()

        # set thread running to false
        self.running=False


    def run(self):
        #run till thread is running
        while self.running:
            #run till this video exhausts
            while self.current_frame_number<self.max_frames_per_video:
                if self.running:
                    frame_time, frame =  self.input_queue.get()
                    if frame_time is None:
                        self.running = False
                        continue
                    frame = cv2.rectangle(frame, (0, frame.shape[0]), (frame.shape[1]//2-300, frame.shape[0] // 2+325),
                                          (0, 0, 0), -1)
                    frame = cv2.putText(frame, f"{frame_time}",
                                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255))
                    self.video_out.write(frame)
                    self.current_frame_number +=1

                    # for debugging purposes
                    # cv2.imshow(self.window_name, frame)
                    # if cv2.waitKey(1) == ord('q'):
                    #     break
                else:
                    self.video_out.release()
                    break
            if self.running:
                # frame_time, frame = self.input_queue.get()
                self.renew_video()




