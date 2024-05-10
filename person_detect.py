"""
Pedestrian Tracking Scripts for Crosswalk Buddy
Description: This file contains the implementation of the pedestrian tracking algorithm for the Crosswalk Buddy robot utilizing YOLO. It tracks the centroid of the pedestrian and uses state_estimator.py to estimate the angle between the robot and the pedestrian.
Author: Advaith Balaji
Progress: In development
"""

import cv2
import numpy as np
import math

from state_estimator import KalmanFilter
from params import Params

import serial
import time
import struct


# horizontal angle between pedestrian and robot
def calc_x_theta(x_centroid):
    img_width = params.img_res[1]  # in pixels
    f = params.camera_focal_length  # in mm
    x_theta_rad = math.atan2((img_width / 2) - x_centroid, f)
    x_theta_deg = -math.degrees(x_theta_rad)

    return x_theta_deg

def send_angle_to_serial(ser, angle):
    arduino_angle = int(-horizontal_angle + 90)
    angle_bytes = struct.pack('<H', arduino_angle)  # Convert angle to little-endian unsigned short (2 bytes)
    ser.write(angle_bytes)

# Load YOLO
net = cv2.dnn.readNet(
    "extra_files/yolov3.weights",
    "extra_files/yolov3.cfg",
)  # Replace with the paths to your YOLO model and configuration files
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# cap = cv2.VideoCapture("extra_files/pedestrian2.mp4")
cap = cv2.VideoCapture(1)

params = Params()
num_theta_states = params.camera_fov
measurements = []
# k = KalmanFilter(62.5, 19, 0.475, 12000, 19)
k = KalmanFilter(initial_estimate=0, initial_estimate_variance=1, action_model_signal=0.0, process_noise=1, measurement_noise=45)
# ser = serial.Serial('/dev/tty.usbmodem101', 9600)

mean_centroid = [1122, 673]
while cap.isOpened():
    ret, image = cap.read()
    if ret:
        height, width, channels = image.shape
        # Detecting objects using YOLO
        blob = cv2.dnn.blobFromImage(
            image, 0.00392, (416, 416), (0, 0, 0), True, crop=False
        )
        net.setInput(blob)
        outs = net.forward(output_layers)[0:1]

        centroids = []

        for i, out in enumerate(outs):
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if (
                    confidence > 0.99 and class_id == 0
                ):  # Class 0 corresponds to 'person' in YOLO
                    # Object detected as a person

                    centroid_x = int(detection[0] * width)
                    centroid_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    centroids.append([centroid_x, centroid_y])

            if centroids:
                mean_centroid = centroids[-1]
                
            cv2.circle(
                image,
                (int(mean_centroid[0]), int(mean_centroid[1])),
                10,
                (0, 255, 0),  # Red color
                -1,
            )

            horizontal_angle = calc_x_theta(mean_centroid[0])
            measurements.append(horizontal_angle)
            k.estimate(horizontal_angle)
            k.show_graph(measurements)

            # send_angle_to_serial(ser, horizontal_angle)

            centroids.clear()


        cv2.imshow("YOLO Person Detection", image)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    else:
        break

# ser.close()
cap.release()
cv2.destroyAllWindows()
