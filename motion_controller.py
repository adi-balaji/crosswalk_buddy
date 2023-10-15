"""
Motion Controller for Crosswalk Buddy
Description: This file contains the implementation of the motion controller for the Crosswalk Buddy robot.
Author: Advaith Balaji
Progress: In development
""" 

class MotionController:

    def __init__(self, vel, kp):
        self.vel = vel
        self.kp = kp

    def drive(pedestrian_theta):
        if pedestrian_theta == None:
            drive_command = self.vel
            print("Drive " + str(drive_command))
        else:
            drive_command = self.vel + (pedestrian_theta * self.kp)
            drive_command = drive_command.__round__(2)
            print("Drive " + str(drive_command))

        return drive_command
    
    def stop():
        print("Stop")
        return 0
