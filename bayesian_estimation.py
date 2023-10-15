import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class BayesianEstimator:

    def normalize_belief(self):
        self.beliefs = self.beliefs / np.sum(self.beliefs)

    def sensor_fusion(self, particle_weights):
        beliefs_as_arr = np.array(self.beliefs)
        particle_weights_arr = np.array(particle_weights)

        new_belief = beliefs_as_arr * abs(particle_weights_arr)

        self.beliefs = new_belief.tolist()
        self.normalize_belief()

    def __str__(self):
        return f"beliefs_max={max(self.beliefs)}, beliefs_size={len(self.beliefs)}, num_states={self.num_states})"
    
    def show_belief_distribution(self):
        plt.plot(range(-self.num_states // 2,self.num_states // 2), self.beliefs, label="Belief Distribution")
        plt.xlabel("State")
        plt.ylabel("Likelihood")
        plt.title("Belief distribution")
        plt.grid(True)
        plt.show()
    
    def __init__(self, n_states, beliefs = None):
        self.num_states = n_states
        self.beliefs = [1/self.num_states] * self.num_states
        self.normalize_belief()


    
class SensorProbabilityDistribution:

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

#---------------------------------------------------------------- END HELPER CLASSES ------------------------------------------------------------------------------------------

#For testing purposes

# n_states = 120
# b = BayesianEstimator(n_states)
# camera = SensorProbabilityDistribution(12, -11, n_states)
# b.sensor_fusion(camera.particle_weights)
# camera.update_sensor_reading(-8)
# b.show_belief_distribution()














              
