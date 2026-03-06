import sys
import os
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gruanpy as gp
import pandas as pd

folder = r'gdp\products_RS41-GDP-1_LIN_2017' # open folder with chosen GDP files
file_paths = [
    os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.nc')
]
for file_path in file_paths[:1]:
    gdp = gp.read(file_path) # read GDP file
    data = gdp.data.head(50).sort_values('time')
    
    # Extract altitude and its uncertainty data
    start = data['time'].values[0]
    time = data['time'].values
    seconds = (time - start) / np.timedelta64(1, 's')
    seconds = seconds.astype(float)
    altitude = data['alt'].values
    altitude_unc = data['alt_uc'].values
    altitude_var = (altitude_unc * 0.5)**2  # variance of altitude measurements
    finite_diff_velocity = np.diff(altitude) / np.diff(seconds)
    finite_diff_velocity_unc = np.sqrt(altitude_var[1:50] + altitude_var[:-1])
    # Plot observation and uncertainties
    plt.figure(figsize=(10, 6))
    plt.errorbar(seconds, altitude, yerr=altitude_unc, fmt='.', alpha=0.5, label='Observations', color='#1f77b4')
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.title('Altitude Observations with Uncertainties')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Kalman filter - local level model
    # State vector: [altitude, velocity]
    # State transition: x_t = A * x_{t-1} + w_t
    # Measurement: y_t = H * x_t + v_t
    dt = 1.0  # time step (1 second)
    A = np.array([[1, dt], [0, 1]])  # state transition matrix
    H = np.array([[1, 0]])  # measurement matrix
    Q = np.array([[1, 0], [0, 1]]) * 0.1  # process noise covariance
    R = np.array([[1]]) * 0.5  # measurement noise covariance
    R = np.mean(altitude_var* 0.5)  # measurement noise covariance (based on altitude uncertainty)
    R = [np.array([[var]])* 0.5 for var in altitude_var]  # measurement noise covariance (time-varying)
    x_est = np.array([[altitude[0]], [finite_diff_velocity[0]]])  # initial state estimate
    P_est = np.eye(2)  # initial estimate covariance
    x_estimates = []
    P_estimates = []
    for i in range(len(altitude)):
        # Prediction step
        x_pred = A @ x_est
        P_pred = A @ P_est @ A.T + Q
        
        # Measurement update
        y_meas = np.array([[altitude[i]]])
        y_pred = H @ x_pred
        y_residual = y_meas - y_pred
        S = H @ P_pred @ H.T + R[i]
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        x_est = x_pred + K @ y_residual
        P_est = (np.eye(2) - K @ H) @ P_pred
        
        x_estimates.append(x_est.flatten())
        P_estimates.append(P_est)
    x_estimates = np.array(x_estimates)
    P_estimates = np.array(P_estimates)
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.errorbar(seconds, altitude, yerr=altitude_unc, fmt='.', alpha=0.5, label='Observations', color='#1f77b4')
    plt.errorbar(seconds, x_estimates[:, 0], yerr=np.sqrt(P_estimates[:, 0, 0]), label='Kalman Filter Estimate', color='#ff7f0e')
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.title('Kalman Filter Altitude Estimation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Plot velocity estimates
    plt.figure(figsize=(10, 6))
    plt.errorbar(seconds[1:], finite_diff_velocity, yerr=finite_diff_velocity_unc, label='Finite Difference Velocity', color='#1f77b4')
    plt.errorbar(seconds, x_estimates[:, 1], yerr=np.sqrt(P_estimates[:, 1, 1]), label='Kalman Filter Velocity Estimate', color='#ff7f0e')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Kalman Filter Velocity Estimation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Kalman smoother 
    x_smooth = np.copy(x_estimates)
    P_smooth = np.copy(P_estimates)
    for i in range(len(altitude) - 2, -1, -1):
        A_inv = np.linalg.inv(A)
        P_pred = A @ P_estimates[i] @ A.T + Q
        C = P_estimates[i] @ A.T @ np.linalg.inv(P_pred)
        x_smooth[i] = x_estimates[i] + C @ (x_smooth[i + 1] - A @ x_estimates[i])
        P_smooth[i] = P_estimates[i] + C @ (P_smooth[i + 1] - P_pred) @ C.T
    # Plot smoothed estimates
    plt.figure(figsize=(10, 6))
    plt.errorbar(seconds, altitude, yerr=altitude_unc, fmt='.', alpha=0.5, label='Observations', color='#1f77b4')
    plt.errorbar(seconds, x_smooth[:, 0], yerr=np.sqrt(P_smooth[:, 0, 0]), label='Kalman Smoother Estimate', color='#2ca02c')
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.title('Kalman Smoother Altitude Estimation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.errorbar(seconds[1:], finite_diff_velocity, yerr=finite_diff_velocity_unc, label='Finite Difference Velocity', color='#1f77b4')
    plt.errorbar(seconds, x_smooth[:, 1], yerr=np.sqrt(P_smooth[:, 1, 1]), label='Kalman Smoother Velocity Estimate', color='#2ca02c')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Kalman Smoother Velocity Estimation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # compare estimates
    plt.figure(figsize=(10, 6))
    plt.errorbar(seconds, altitude, yerr=altitude_unc, fmt='.', alpha=0.5, label='Observations', color='#1f77b4')
    plt.errorbar(seconds, x_estimates[:, 0], yerr=np.sqrt(P_estimates[:, 0, 0]), label='Kalman Filter Estimate', color='#ff7f0e')
    plt.errorbar(seconds, x_smooth[:, 0], yerr=np.sqrt(P_smooth[:, 0, 0]), label='Kalman Smoother Estimate', color='#2ca02c')
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.title('Kalman Filter vs Smoother Altitude Estimation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.errorbar(seconds[1:], finite_diff_velocity, yerr=finite_diff_velocity_unc, label='Finite Difference Velocity', color='#1f77b4')
    plt.errorbar(seconds, x_estimates[:, 1], yerr=np.sqrt(P_estimates[:, 1, 1]), label='Kalman Filter Velocity Estimate', color='#ff7f0e')
    plt.errorbar(seconds, x_smooth[:, 1], yerr=np.sqrt(P_smooth[:, 1, 1]), label='Kalman Smoother Velocity Estimate', color='#2ca02c')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Kalman Filter vs Smoother Velocity Estimation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()