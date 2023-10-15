class Params:

    def __init__(self):

        # in degrees
        self.camera_fov = 120 

        #in mm 
        self.camera_focal_length = 26

        # in pixels
        self.img_res = [225, 400]

        # std dev of camera sensor
        self.camera_sigma = 25

        # std dev of lidar sensor
        self.lidar_sigma = 4
