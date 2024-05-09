from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import cv2
import math
import struct
from state_estimator import KalmanFilter
from params import Params

params = Params()

# horizontal angle between pedestrian and robot
def calc_x_theta(x_centroid):
    img_width = params.img_res[1]  # in pixels
    f = params.camera_focal_length  # in mm
    x_theta_rad = math.atan2((img_width / 2) - x_centroid, f)
    x_theta_deg = -math.degrees(x_theta_rad)

    return x_theta_deg


model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
cap = cv2.VideoCapture(1)
measurements = []
k = KalmanFilter(0, 1, 0.0, 1, 35)
centroid = (0,0)        


while cap.isOpened():
    ret, img = cap.read()
    if ret:
        image = Image.fromarray(img.astype('uint8'), 'RGB')
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        logits = outputs.logits
        bboxes = outputs.pred_boxes

        # print results
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
        # for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        #     # box = [round(i, 2) for i in box.tolist()]
        #     # print(
        #     #     f"Detected {model.config.id2label[label.item()]} with confidence "
        #     #     f"{round(score.item(), 3)} at location {box}"
        #     # )
        #     centroid = int(box[0] + box[2] / 2), int(box[1] + box[3] / 2)
        #     cv2.circle(img, centroid, 10, (255, 0, 0), -1)

        box = results['boxes']
        
        if(box.size() != (0, 4)):
            box = box[0]
            centroid = int(box[0] + box[2] / 2), int(box[1] + box[3] / 2)
        
        horizontal_angle = calc_x_theta(centroid[0])
        measurements.append(horizontal_angle)
        k.estimate(horizontal_angle)
        k.show_graph(measurements)

        # arduino_angle = int(-horizontal_angle + 90)
        # angle_bytes = struct.pack('<H', arduino_angle)  # Convert angle to little-endian unsigned short (2 bytes)
        # ser.write(angle_bytes)
        # time.sleep(0.01)
            
        # cv2.circle(img, (centroid[0], centroid[1]), 5, (255, 0, 0), -1)
        # cv2.imshow("YOLO Person Detection", img)
        # if cv2.waitKey(25) & 0xFF == ord("q"):
        #     break