import cv2
import imutils
import math
import numpy as np

from bayesian_estimation import BayesianEstimator
from bayesian_estimation import SensorProbabilityDistribution
from params import Params
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

detector_params = Params()

def calc_x_theta(x_centroid):
    img_width = detector_params.img_res[1]  # in pixels
    f = detector_params.camera_focal_length # in mm
    x_theta_rad = math.atan2((img_width / 2) - x_centroid, f)
    x_theta_deg = -math.degrees(x_theta_rad)

    return x_theta_deg.__round__(2)

def sim_drive(theta):
    vel = 0.5
    kp = 0.001
    drive_command = 0
    if math.fabs(theta) <= 15:
        drive_command = vel
        print("Drive " + str(drive_command))
    else:
        drive_command = vel + (theta * kp)
        drive_command = drive_command.__round__(2)
        print("Drive " + str(drive_command))


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture('extra_files/pedestrian.mp4')
# cap = cv2.VideoCapture(0)

bounding_boxes = []
centroids = []

num_theta_states = detector_params.camera_fov
b = BayesianEstimator(num_theta_states)
camera_sensor = SensorProbabilityDistribution(12, -19, num_theta_states)

while cap.isOpened():
    # Reading the video stream
    ret, image = cap.read()
    if ret:
        image = imutils.resize(image, width=min(400, image.shape[1]))

        # Detecting all the region in the Image that has a pedestrian inside it
        (regions, _) = hog.detectMultiScale(
            image, winStride=(4, 4), padding=(20, 20), scale=1.01
        )

        # Drawing the regions in the
        # Image
        # for x, y, w, h in regions:
        #     centroid_x = x + w // 2
        #     centroid_y = y + h // 2
        #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #     theta_to_object = calc_x_theta(centroid_x)
        #     cv2.circle(image, (centroid_x, centroid_y), 5, (0, 255, 0), -1)

        for x, y, w, h in regions:
            for i in range(1, 500):
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

            camera_sensor.update_sensor_reading(theta_to_object)
            b.sensor_fusion(camera_sensor.particle_weights)
            b.show_belief_distribution()
        
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

# # Load YOLO
# net = cv2.dnn.readNet(
#     "yolov3.weights",
#     "yolov3.cfg",
# )  # Replace with the paths to your YOLO model and configuration files
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# cap = cv2.VideoCapture('pedestrian.mp4')

# cam_fov = 60  # in degrees
# num_theta_states = cam_fov
# b = BayesianEstimator(num_theta_states)
# camera_sensor = SensorProbabilityDistribution(12, 1, num_theta_states)

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

#                 # camera_sensor.update_sensor_reading(calc_x_theta(mean_centroid[0]))
#                 # b.sensor_fusion(camera_sensor.particle_weights)

#                 print(calc_x_theta(mean_centroid[0]))

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
# b.show_belief_distribution()

