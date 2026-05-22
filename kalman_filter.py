import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, x_0, P_0):
        self.A = A  
        self.B = B  
        self.H = H  
        self.Q = Q  
        self.R = R 
        
        self.x = x_0
        self.P = P_0 

    def predict(self, u=None):
        """
        PREDICT NEXT STATE AND NEXT ERROR COVARIANCE MATRIX
        """
        # Predict next state: x_k = A * x_{k-1} + B * u
        if self.B is not None and u is not None:
            self.x = self.A @ self.x + self.B @ u
        else:
            self.x = self.A @ self.x
            
        # Predict next error covariance matrix: P_k = A * P_{k-1} * A^T + Q
        self.P = self.A @ self.P @ self.A.T + self.Q
        
        return self.x, self.P

    def update(self, z):
        """
        UPDATE NEW STATE AND NEW ERROR COVARIANCE MATRIX
        """
        # 1. Calculate Innovation/Measurement Residual: y = z - H * x
        y = z - self.H @ self.x
        
        # Calculate Innovation Covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # 3. Calculate Kalman Gain: K = P * H^T * S^-1
        # Kalman gain will determine whether the system should base on the measurement or prediction more.
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 4. Update state estimation: x_k = x_{predict} + K * y
        self.x = self.x + K @ y
        
        # 5. Update Error Covariance Matrix: P_k = (I - K * H) * P_{predict}
        I = np.eye(self.P.shape[0]) # Ma trận đơn vị I
        self.P = (I - K @ self.H) @ self.P
        
        return self.x, self.P

def run_kalman_filter(kf, measurements, control_inputs=None):
    """
    Run the z_k through Kalman filter
    """
    estimated_states = []
    error_covariances = []
    
    for i, z in enumerate(measurements):
        u = control_inputs[i] if control_inputs is not None else None
        
        # Predict -> Update
        kf.predict(u)
        x_k, P_k = kf.update(z)
        
        # Store the results in arrays
        estimated_states.append(x_k.copy())
        error_covariances.append(P_k.copy())
        
    return estimated_states, error_covariances

def simulate_position_velocity_1d(
    N=50,
    dt=1.0,
    x0_true=None,
    Q=None,
    R=None,
    seed=42
):
    """
    Simulate a 1D position-velocity system and generate:
    - true state
    - measurement noise
    - measured data

    State model:
        x_k = A x_{k-1} + w_k

    Measurement model:
        z_k = H x_k + v_k
    """

    np.random.seed(seed)

    # Default initial true state
    if x0_true is None:
        x0_true = np.array([[0.0],
                            [1.0]])  

    # State transition matrix
    A = np.array([[1, dt],
                  [0, 1]])

    # No control input
    B = None

    # Observation matrix: only position is measured
    H = np.array([[1, 0]])

    # Default process noise covariance
    if Q is None:
        Q = np.array([[0.1, 0.0],
                      [0.0, 0.1]])

    # Default measurement noise covariance
    if R is None:
        R = np.array([[5.0]])

    # Time vector
    t = np.arange(N)

    # Containers
    true_states = np.zeros((2, N))
    measurement_noise = np.zeros((1, N))
    measured_data = np.zeros((1, N))

    # Initial true state
    true_states[:, [0]] = x0_true

    # Generate true states
    for k in range(1, N):
        w_k = np.random.multivariate_normal(
            mean=[0, 0],
            cov=Q
        ).reshape(2, 1)

        true_states[:, [k]] = A @ true_states[:, [k-1]] + w_k

    # Generate measurements
    for k in range(N):
        v_k = np.random.normal(
            loc=0.0,
            scale=np.sqrt(R[0, 0])
        )

        measurement_noise[:, k] = v_k
        measured_data[:, [k]] = H @ true_states[:, [k]]
        + np.array([[v_k]])

    # Initial estimate and covariance for Kalman Filter
    x_0 = np.array([[0.0],
                    [0.0]])   # initial estimated state

    P_0 = np.array([[1.0, 0.0],
                    [0.0, 1.0]])

    # Convert measured_data into the format expected by kalman_filter.py
    measurements = [measured_data[:, [k]] for k in range(N)]

    simulation_data = {
        "dt": dt,
        "N": N,
        "A": A,
        "B": B,
        "H": H,
        "Q": Q,
        "R": R,
        "x0_true": x0_true,
        "x_0": x_0,
        "P_0": P_0,
        "t": t,
        "true_states": true_states,
        "measurement_noise": measurement_noise,
        "measured_data": measured_data,
        "measurements": measurements
    }

    return simulation_data

if __name__ == "__main__":
    data = simulate_position_velocity_1d(N=101, dt=1.0)
    
    A = data["A"]
    B = data["B"]
    H = data["H"]
    Q = data["Q"]
    R = data["R"]
    x_0 = data["x_0"]
    P_0 = data["P_0"]
    measurements = data["measurements"]
    
    kf = KalmanFilter(A, B, H, Q, R, x_0, P_0)
    estimated_states, error_covariances = run_kalman_filter(kf, measurements)
    
    estimated_states_arr = np.array(estimated_states).reshape(data["N"], 2).T
    
    t = data["t"]
    true_states = data["true_states"]
    measured_data = data["measured_data"]
    
    # Draw graph
    plt.figure(figsize=(14, 6))
    
    # Graph 1: Position
    plt.subplot(1, 2, 1)
    plt.plot(t, true_states[0, :], label="True position", color='green', linewidth=2)
    plt.scatter(t, measured_data[0, :], label="Errors", color='red', marker='x', alpha=0.6)
    plt.plot(t, estimated_states_arr[0, :], label="Estimated position", color='blue', linestyle='--', linewidth=2)
    plt.title("Position")
    plt.xlabel("Time (t)")
    plt.ylabel("Position")
    plt.legend()
    plt.grid(True)
    
    # Graph 2: Velocity
    plt.subplot(1, 2, 2)
    plt.plot(t, true_states[1, :], label="True velocity", color='green', linewidth=2)
    plt.plot(t, estimated_states_arr[1, :], label="Estimated velocity", color='blue', linestyle='--', linewidth=2)
    plt.title("Velocity")
    plt.xlabel("Time (t)")
    plt.ylabel("Velocity")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()