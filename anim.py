import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Time settings
dt = 1.0
num_steps = 50

# True motion settings
true_velocity = 1.0
true_initial_position = 0.0

# Kalman filter setup
A = np.array([[1, dt],
              [0, 1]])  # Transition matrix
H = np.array([[1, 0]])  # Measurement model
Q = np.array([[1e-5, 0],
              [0, 1e-5]])  # Process noise
R = np.array([[0.5]])    # Measurement noise
x = np.array([[0], [0]])  # Initial estimate (position, velocity)
P = np.eye(2)

# Simulated data
true_positions = [true_initial_position + i * true_velocity for i in range(num_steps)]
measurements = [pos + np.random.normal(0, np.sqrt(R[0,0])) for pos in true_positions]

# For storing results
kalman_positions = []
predicted_positions = []

for z in measurements:
    # Prediction
    x = A @ x
    P = A @ P @ A.T + Q
    predicted_positions.append(x[0, 0])
    
    # Update
    z = np.array([[z]])
    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = (np.eye(2) - K @ H) @ P
    kalman_positions.append(x[0, 0])

# Animation
fig, ax = plt.subplots()
ax.set_xlim(0, num_steps)
ax.set_ylim(min(measurements) - 1, max(measurements) + 1)
line_true, = ax.plot([], [], 'g-', label='True Position')
line_meas, = ax.plot([], [], 'rx', label='Measurements')
line_kalman, = ax.plot([], [], 'b-', label='Kalman Estimate')
ax.legend()

def init():
    line_true.set_data([], [])
    line_meas.set_data([], [])
    line_kalman.set_data([], [])
    return line_true, line_meas, line_kalman

def update(frame):
    x_vals = list(range(frame + 1))
    line_true.set_data(x_vals, true_positions[:frame + 1])
    line_meas.set_data(x_vals, measurements[:frame + 1])
    line_kalman.set_data(x_vals, kalman_positions[:frame + 1])
    return line_true, line_meas, line_kalman

ani = animation.FuncAnimation(fig, update, frames=num_steps,
                              init_func=init, blit=True)

plt.show()
