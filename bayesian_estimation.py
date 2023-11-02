"""
Bayesian State Estimation for Crosswalk Buddy
Description: This file contains the implementation of the Bayesian Estimation algorithm for the estimation theta between robot and pedestrian.
Author: Advaith Balaji
Progress: In development
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def plot(data):
    num_states = len(data)
    plt.plot(range(-num_states // 2,num_states // 2), data, label="Motion Model")
    plt.xlabel("State")
    plt.ylabel("Likelihood")
    plt.title("Motion Model")
    plt.grid(True)
    plt.show()

class BayesianEstimator:

    def normalize_belief(self):
        self.beliefs = self.beliefs / np.sum(self.beliefs)

    def sensor_fusion(self, weights):
        beliefs_as_arr = np.array(self.beliefs)
        weights_arr = np.array(weights)

        new_belief = beliefs_as_arr * abs(weights_arr)

        self.beliefs = new_belief.tolist()
        self.normalize_belief()

    def motion_fusion(self, weights):
        beliefs_as_arr = np.array(self.beliefs)
        weights_arr = np.array(weights)

        new_belief = beliefs_as_arr * abs(weights_arr)

        self.beliefs = new_belief.tolist()
        self.normalize_belief()

    
    def show_belief_distribution(self, realtime = False):
        if(realtime):
            plt.clf()
        plt.plot(range(-self.num_states // 2,self.num_states // 2), self.beliefs, label="Belief Distribution")
        plt.xlabel("State")
        plt.ylabel("Likelihood")
        plt.title("Belief distribution")
        plt.grid(True)
        if(realtime):
            plt.ion()
        plt.show()

    def add_noise(self, noise_sigma):
        noise = np.random.normal(0, noise_sigma, self.num_states)
        self.beliefs = self.beliefs + noise
        self.normalize_belief()
    
    def __init__(self, n_states, beliefs = None):
        self.num_states = n_states
        self.beliefs = [1/self.num_states] * self.num_states
        self.normalize_belief()


class SensorModel:

    def compute_gaussian_distribution(self):
        self.particle_weights = [
            (1 / (self.stddev * np.sqrt(2 * np.pi))) * (np.e ** (-0.5 * (((x - self.mean) / self.stddev) ** 2))) for x in self.particle_weights
        ]

    def update_sensor_reading(self, sensor_reading):
        self.reset_weights()
        self.mean = sensor_reading
        self.compute_gaussian_distribution()

    def show_particle_distribution(self):
        plt.plot(range(-len(self.particle_weights)//2,len(self.particle_weights)//2), self.particle_weights, label="Sensor pdf")
        plt.grid(True)
        plt.show()
    
    def reset_weights(self):
        self.particle_weights = list(range(-len(self.particle_weights)//2,len(self.particle_weights)//2))

    def __init__(self, stddev, mean, num_states, particle_weights = None):
        self.stddev = stddev    #tune according to sensor

        self.mean = mean    #plug in sensor reading
        self.particle_weights = list(range(-num_states // 2, num_states // 2))
        self.compute_gaussian_distribution()

class PedestrianMotionModel:

    motion_model_sigma = 25 #uncertainty of motion model
    noise_sigma = 0.0005 #noise constant

    def compute_gaussian_distribution(self):
        self.state_curve = [
            (1 / (self.motion_model_sigma * np.sqrt(2 * np.pi))) * (np.e ** (-0.5 * (((x - self.current_state) / self.motion_model_sigma) ** 2))) for x in self.state_curve
        ]

    def normalize_state_curve(self):
        self.state_curve = self.state_curve / np.sum(self.state_curve)

    def __init__(self, current_state, dtheta, num_states):
        self.current_state = current_state
        self.dtheta = dtheta
        self.num_states = num_states
        self.state_curve = list(range(-num_states // 2, num_states // 2))
        self.compute_gaussian_distribution()
        self.normalize_state_curve()

    def add_noise(self):
        noise = np.random.normal(0, self.noise_sigma, self.num_states)
        self.state_curve = self.state_curve + noise
        self.normalize_state_curve()

    def update_motion_model(self):
        self.state_curve = list(range(-len(self.state_curve)//2,len(self.state_curve)//2)) # reset
        self.current_state += self.dtheta
        self.compute_gaussian_distribution()
        # self.add_noise()
        # self.normalize_state_curve()

    def show_state_curve(self, realtime=False):
        if(realtime):
            plt.clf()
        plt.plot(range(-self.num_states // 2,self.num_states // 2), self.state_curve, label="Motion Model")
        plt.xlabel("State")
        plt.ylabel("Likelihood")
        plt.title("Motion Model")
        plt.grid(True)
        if(realtime):
            plt.ion()
        plt.show()


#---------------------------------------------------------------- END HELPER CLASSES ------------------------------------------------------------------------------------------

#For testing purposes

# n_states = 120
# motion = PedestrianMotionModel(current_state=-11, dtheta=30, num_states=n_states)
# belief = BayesianEstimator(n_states=n_states)
# sensor = SensorModel(stddev=12, mean=-11, num_states=n_states)
# sensy = SensorModel(stddev=30, mean=-11, num_states=n_states)

# for i in range(1,100):
#     sensor.update_sensor_reading(-11)
#     belief.sensor_fusion(sensor.particle_weights)
#     sensy.update_sensor_reading(-11 + i)
#     belief.sensor_fusion(sensy.particle_weights)
#     belief.show_belief_distribution(realtime=False)










    
    













              
