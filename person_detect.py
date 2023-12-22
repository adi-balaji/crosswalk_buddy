"""
Pedestrian Tracking Scripts for Crosswalk Buddy
Description: This file contains the implementation of the pedestrian tracking algorithm for the Crosswalk Buddy robot utilizing HOG and YOLO. It tracks the centroid of the pedestrian and uses bayesian_estimation.py to estimate the angle between the robot and the pedestrian.
Author: Advaith Balaji
Progress: In development
"""

import cv2
import imutils
import math
import numpy as np
import matplotlib.pyplot as plt

from state_estimator import KalmanFilter
from params import Params

Params = Params()


# horizontal angle between pedestrian and robot
def calc_x_theta(x_centroid):
    img_width = Params.img_res[1]  # in pixels
    f = Params.camera_focal_length  # in mm
    x_theta_rad = math.atan2((img_width / 2) - x_centroid, f)
    x_theta_deg = -math.degrees(x_theta_rad)

    return x_theta_deg.__round__(4)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture("extra_files/pedestrian2.mp4")
# cap = cv2.VideoCapture(0)

bounding_boxes = []
centroids = []
measurements = [60]

k = KalmanFilter(60, 13, 0.675, 19.75, 13)

while cap.isOpened():
    # Reading the video stream
    ret, image = cap.read()
    if ret:
        image = imutils.resize(image, width=min(400, image.shape[1]))

        # Detecting all the region in the Image that has a pedestrian inside it
        (regions, _) = hog.detectMultiScale(
            image, winStride=(4, 4), padding=(20, 20), scale=1.01
        )

        for x, y, w, h in regions:
            for i in range(1, 100):
                centroid = [x + w // 2, y + h // 2]
                bounding_box = [x, y, x + w, y + h]
                bounding_boxes.append(bounding_box)
                centroids.append(centroid)

            bounding_boxes_arr = np.array(bounding_boxes)
            centroids_arr = np.array(centroids)
            mean_bbox = np.mean(bounding_boxes_arr, axis=0)
            mean_centroid = np.mean(centroids_arr, axis=0)

            bounding_boxes.clear()
            centroids.clear()

            bounding_box = cv2.rectangle(
                image,
                (int(mean_bbox[0]), int(mean_bbox[1])),
                (int(mean_bbox[2]), int(mean_bbox[3])),
                (0, 0, 255),
                2,
            )

            cv2.circle(
                image,
                (int(mean_centroid[0]), int(mean_centroid[1])),
                5,
                (0, 255, 0),
                -1,
            )

            theta_to_object = calc_x_theta(mean_centroid[0])

            if(abs(theta_to_object - measurements[-1]) > 45):
                measurements.append(measurements[-1])
            
            measurements.append(theta_to_object)
            print(theta_to_object)
            k.estimate(measurement=theta_to_object)
            k.show_graph(measurements=measurements)

        # Showing the output Image
        cv2.flip(image, 1)
        cv2.imshow("Person Detection", image)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import math

# from state_estimator import KalmanFilter
# from params import Params


# # horizontal angle between pedestrian and robot
# def calc_x_theta(x_centroid):
#     img_width = params.img_res[1]  # in pixels
#     f = params.camera_focal_length  # in mm
#     x_theta_rad = math.atan2((img_width / 2) - x_centroid, f)
#     x_theta_deg = -math.degrees(x_theta_rad)

#     return x_theta_deg


# # Load YOLO
# net = cv2.dnn.readNet(
#     "extra_files/yolov3.weights",
#     "extra_files/yolov3.cfg",
# )  # Replace with the paths to your YOLO model and configuration files
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# cap = cv2.VideoCapture("extra_files/pedestrian2.mp4")

# params = Params()
# num_theta_states = params.camera_fov
# measurements = []
# k = KalmanFilter(62.5, 19, 0.475, 12000, 19)

# while cap.isOpened():
#     ret, image = cap.read()
#     if ret:
#         height, width, channels = image.shape
#         # Detecting objects using YOLO
#         blob = cv2.dnn.blobFromImage(
#             image, 0.00392, (416, 416), (0, 0, 0), True, crop=False
#         )
#         net.setInput(blob)
#         outs = net.forward(output_layers)

#         class_ids = []
#         confidences = []
#         boxes = []
#         centroids = []

#         for out in outs:
#             for detection in out:
#                 scores = detection[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]
#                 if (
#                     confidence > 0.5 and class_id == 0
#                 ):  # Class 0 corresponds to 'person' in YOLO
#                     # Object detected as a person

#                     centroid_x = int(detection[0] * width)
#                     centroid_y = int(detection[1] * height)
#                     w = int(detection[2] * width)
#                     h = int(detection[3] * height)

#                     centroids.append([centroid_x, centroid_y])

#             if centroids:
#                 mean_centroid = np.mean(np.array(centroids), axis=0)

#                 # Draw the average centroid
#                 cv2.circle(
#                     image,
#                     (int(mean_centroid[0]), int(mean_centroid[1])),
#                     10,
#                     (0, 255, 0),  # Red color
#                     -1,
#                 )

#                 horizontal_angle = calc_x_theta(mean_centroid[0])
#                 measurements.append(horizontal_angle)
#                 k.estimate(horizontal_angle)
#                 k.show_graph(measurements)

#                 # Clear the list of detected centroids for the next frame
#                 centroids.clear()

#         # cv2.line(image, (1280 // 2, 0), (1280 // 2, 960), (0, 0, 255), thickness=1)
#         cv2.imshow("YOLO Person Detection", image)
#         if cv2.waitKey(25) & 0xFF == ord("q"):
#             break
#     else:
#         break

# cap.release()
# cv2.destroyAllWindows()
