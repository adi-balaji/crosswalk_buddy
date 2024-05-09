import sys

""" 
Parameters for Crosswalk Buddy robot
Description: This file contains the parameters for the Crosswalk Buddy robot used in the other scripts
Author: Advaith Balaji
Progress: In development
"""


class Params:
    def __init__(self):
        # in degrees
        # self.camera_fov = 120 # iphone
        self.camera_fov = 120 #mac

        # in mm
        # self.camera_focal_length = 77 #iphone
        self.camera_focal_length = 520 #mac

        # in pixels
        # self.img_res = [1080, 1920] # caffe
        self.img_res = [1080, 1920] # yolo
        # self.img_res = [225, 400] # HOG

        # std dev of camera sensor
        self.camera_sigma = 30

        # std dev of lidar sensor
        self.lidar_sigma = 4
