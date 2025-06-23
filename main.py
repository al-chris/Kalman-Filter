import numpy as np
import matplotlib.pyplot as plt

# Kalman Filter Implementation
class KalmanFilter:
    def __init__(self, A, B, H, Q, R, x_0, P_0):
        # Initialize variables
        self.A = A  # State transition matrix
        self.B = B  # Control matrix
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x = x_0  # Initial state estimate
        self.P = P_0  # Initial covariance estimate

    def predict(self, u):
        # Predict the next state
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        # Kalman Gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(np.dot(np.dot(self.H, self.P), self.H.T) + self.R))
        # Update state estimate
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        # Update covariance estimate
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

# Define system parameters
A = np.array([[1, 1], [0, 1]])  # State transition matrix (1 step prediction)
B = np.array([[0.5], [1]])  # Control matrix (for velocity)
H = np.array([[1, 0]])  # Measurement matrix (only position is measured)
Q = np.array([[1, 0], [0, 1]])  # Process noise covariance
R = np.array([[10]])  # Measurement noise covariance
x_0 = np.array([0, 1])  # Initial state estimate (position 0, velocity 1)
P_0 = np.eye(2)  # Initial covariance estimate

# Instantiate Kalman Filter
kf = KalmanFilter(A, B, H, Q, R, x_0, P_0)

# Simulate real-world data with noise
n_steps = 100
true_positions = np.linspace(0, 50, n_steps)
noisy_measurements = true_positions + np.random.normal(0, 2, n_steps)

# Track the Kalman filter estimate
estimates = []

# Simulate the Kalman filter tracking over time
for i in range(n_steps):
    u = np.array([1])  # Control input (constant velocity)
    kf.predict(u)
    kf.update(np.array([noisy_measurements[i]]))  # New measurement
    estimates.append(kf.x[0])  # Store the position estimate

# Plot the results
plt.plot(true_positions, label='True Position')
plt.plot(noisy_measurements, label='Noisy Measurements', linestyle='dashed')
plt.plot(estimates, label='Kalman Filter Estimate', linestyle='dotted')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.title('Kalman Filter Tracking')
plt.show()
