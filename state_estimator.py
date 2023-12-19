"""
State Estimator for Crosswalk Buddy
Description: This file contains an implementation of a One-Dimensional Kalman Filter used for the estimation of horizontal angle between robot and pedestrian.
Author: Advaith Balaji
Progress: In Progress
"""

import numpy as np
import matplotlib.pyplot as plt
import time

class KalmanFilter:

    def __init__(self, initial_estimate, initial_estimate_variance, action_model_signal, process_noise, measurement_noise):
        self.estimates = [initial_estimate]
        self.estimate_variances = [initial_estimate_variance]
        self.action_model_signal = action_model_signal
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.truths = None

    def estimate(self, measurement):
        prior_prediction = self.estimates[-1]
        prior_prediction_variance = self.estimate_variances[-1]

        curr_estimate_from_prior = prior_prediction + self.action_model_signal
        curr_estimate_from_prior_variance = prior_prediction_variance + self.process_noise

        kalman_gain = curr_estimate_from_prior_variance / (curr_estimate_from_prior_variance + self.measurement_noise)

        innovation = measurement - curr_estimate_from_prior

        posterior = curr_estimate_from_prior + (kalman_gain * innovation)
        posterior_variance = (1 - kalman_gain) * curr_estimate_from_prior_variance

        self.estimates.append(posterior)
        self.estimate_variances.append(posterior_variance)

        return posterior
    
    def show_graph(self, measurements):
        plt.clf()
        plt.ion()
        plt.plot(self.estimates, label="Estimate", marker=".")
        plt.plot(measurements, label="Measurements", linestyle="-")
        if self.truths is not None:
            plt.plot(self.truths, label="Truths", color="g", linestyle="--")
        plt.title("State Estimation with Kalman Filter for Static System")
        plt.legend()
        plt.show()
        plt.pause(0.065)

    def show_belief_distribution(self):
        beliefs = list(range(-120 // 2, 120 // 2))
        beliefs = [(1 / (np.sqrt(self.estimate_variances[-1]) * np.sqrt(2 * np.pi))) * (np.e ** (-0.5 * (((x - self.estimates[-1]) / np.sqrt(self.estimate_variances[-1])) ** 2))) for x in beliefs]
        plt.clf()
        plt.ion()
        plt.plot(range(-120 // 2, 120 // 2), beliefs,label="Belief Distribution")
        plt.title("Belief distibution")
        plt.show()
        plt.pause(0.1)

# ---------------------------------------------------------------- END CLASSES ------------------------------------------------------------------------------------------

# For testing with dummy data
# xvals = np.linspace(0,100, 70)
# yvals = 0.87 * xvals
# dummy_truths = yvals

# dummy_measurements = dummy_truths + np.random.normal(0, 9.1, 70)
# time.sleep(2)

# test_measurement_noise_sigma = 199
# test_process_noise_sigma = 7.9
# test_action_model_signal = 1.103
# k = KalmanFilter(dummy_measurements[0], test_measurement_noise_sigma, test_action_model_signal, test_process_noise_sigma, test_measurement_noise_sigma)
# k.truths = dummy_truths

# for measurement in dummy_measurements[1:]:
#     k.estimate(measurement)
#     # k.show_graph(dummy_measurements)
#     k.show_belief_distribution()
