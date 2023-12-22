""" 
Parameters for Crosswalk Buddy robot
Description: This file contains the parameters for the Crosswalk Buddy robot used in the other scripts
Author: Advaith Balaji
Progress: In development
"""


class Params:
    def __init__(self):
        # in degrees
        self.camera_fov = 120

        # in mm
        self.camera_focal_length = 77

        # in pixels
        # self.img_res = [1080, 1920] # yolo
        self.img_res = [225, 400] # HOG

        # std dev of camera sensor
        self.camera_sigma = 30

        # std dev of lidar sensor
        self.lidar_sigma = 4
