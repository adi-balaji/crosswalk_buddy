# Crosswalk Buddy

Crosswalk Buddy is an independent research project under UM Robotics with the goal of developing a robot that will increase safety in pedestrian spaces. The aim of this robot is to increase driver visibility of pedestrians in low visibility scenarios. This repository contains the software stack for the robot that allows it to autonomously follow, lead, or travel alongside a pedestrian with a red light to ensure drivers can know the location of a pedestrian within a crosswalk at all times. Primary research will focus on human-robot interaction in the pedestrian space - judging human-robot trust, likability, comfortability, usability etc. between the pedestrian and robot in crosswalks.

<br>

<p align="center">
  <img src="https://github.com/adi-balaji/crosswalk_buddy/initial_cad.png" alt="Image">
  <br>
  <sub><em>Crosswalk Buddy (initial CAD by Annalise Richmond)</em></sub>
</p>


## Overview

- **params.py:** Contains global parameters for the bayesian state estimator, person tracking algorithm, and robot motion controller.

- **bayesian_estimator:** Provides a Bayesian state estimation class that allows modeling of the pedestrian state (in the angle between the robot and person centroid) probabilistically, enabling the use of multiple sensor inputs.

- **person_detect.py:** Offers a pedestrian tracking algorithm utilizing yolov3 to track the centroid of the detected person, as well as the angle between the robot and pedestrian.

- **motion_controller.py:** Currently in progress.

## Researchers

- Advaith Balaji
- Tamaer Alharastani
- Derrick Yeo

## Sponsors

- UM Robotics
- MCity

## Download weights and cfg for YOLOv3

Please download `yolov3.weights` and `yolov3.cfg` from [YOLOv3-320](https://pjreddie.com/darknet/yolo/) and place them in the appropriate location.
