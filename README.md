# Crosswalk Buddy

Crosswalk Buddy is an independent research project under UM Robotics with the goal of developing a robot that will increase safety in pedestrian spaces. The aim of this robot is to increase driver visibility of pedestrians in low visibility scenarios. This repository contains the software stack for the robot that allows it to autonomously follow, lead, or travel alongside a pedestrian with a red light to ensure drivers can know the location of a pedestrian within a crosswalk at all times. Primary research will focus on human-robot interaction in the pedestrian space - judging human-robot trust, likability, comfortability, usability etc. between the pedestrian and robot in crosswalks.

<br>

This research project will be done in 2 phases. The first phase includes development of a VR robot simulation platform on Unity in order to test the nature of human-robot interaction in the pedestrian space. We will conduct VR crosswalk trials with the simulated robot on humans and gather data that helps us determine the effect of different robot configurations on making a pleasant crosswalk experience. We aim to analyze the effect of various robot positions, proximities, alert modalities, motion planning methods, etc on humans' perceived sense of safety, trust in the robot, and comfortability. The second phase includes prototyping and iterating on robot design according to the findings of phase 1 and testing various designs in real world scenarios. We aim to develop a full autonomous navigation software stack to support pedestrian tracking and state estimation, motion planning and driver alert systems.

<br>

This repository contains code **in development** to be applied on a prototype that will be tested in real world crosswalk scenarios at MCity.

<br>

<p align="center">
  <img src="https://github.com/adi-balaji/crosswalk_buddy/blob/main/github_assets/initial_cad.png" alt="Image">
  <br>
  <sub><em>Crosswalk Buddy (initial CAD by Annalise Richmond)</em></sub>
</p>


## Overview

- **params.py:** Contains global parameters for the bayesian state estimator, person tracking algorithm, and robot motion controller.
  
- **person_detect.py:** Offers a pedestrian tracking algorithm utilizing yolov3 or HOG Descriptor to track the centroid of the detected person, as well as the angle between the robot and pedestrian.

- **bayesian_estimator.py:** *THIS MODULE IS DEPRECATED!!* Please refer to `state_estimator.py` for the latest, better implementation of pedestrian state estimation using 1D Kalman Filter. Currently estimates pedestrian state as horizontal angle between robot and pedestrian probabalistically in the robot coordinate frame.

<br>

<p align="center">
  <img src="https://github.com/adi-balaji/crosswalk_buddy/blob/main/github_assets/working_bayesian.gif" alt="Image">
  <br>
  <sub><em>Bayesian State Estimator with state reading from person_detect.py </em></sub>
</p>

<br>

- **state_estimator.py:** Contains an implementation of a One-Dimensional Kalman Filter to model pedestrian state probabilistically, building upon the deprecated `bayesian_estimation.py`. Allows the use of multiple sensor inputs to generate reasonably accurate estimates. **Currently in progress**.
  
  <p align="center">
    <img src="https://github.com/adi-balaji/crosswalk_buddy/blob/main/github_assets/kalman_dynamix.gif" alt="Dynamic Kalman Filter" width="45%"/>
    <img src="https://github.com/adi-balaji/crosswalk_buddy/blob/main/github_assets/kalman_statix.gif" alt="Static Kalman Filter" width="45%"/>
  </p>
  <p align="center">
    <em>One-Dimensional Kalman Filter working on test data</em>
  </p>
  <br>

- **motion_controller.py:** Motion controller to maintain a certain constant position relative to moving pedestrian. **Currently in progress**.

## Researchers

- Advaith Balaji
- Tamaer Alharastani
- Derrick Yeo

## Sponsors

- UM Robotics
- MCity

## Download weights and cfg for YOLOv3

Please download `yolov3.weights` and `yolov3.cfg` from [YOLOv3-320](https://pjreddie.com/darknet/yolo/) and place them in the appropriate location.
