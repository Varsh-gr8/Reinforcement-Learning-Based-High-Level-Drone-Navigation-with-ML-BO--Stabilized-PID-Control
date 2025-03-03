'''
import os
import time
import numpy as np
import csv
import pybullet as p
from env1 import DroneNavigationAviary
from gym_pybullet_drones.utils.enums import DroneModel

def collect_pid_data():
    # Ensure data directory exists.
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_filename = os.path.join(data_dir, "pid_data.csv")
    print(f"CSV file will be saved at: {csv_filename}")

    # Check CSV file creation.
    try:
        with open(csv_filename, "w", newline="", encoding="utf-8") as test_file:
            test_file.write("Test Write\n")
        print("CSV file is writable!")
    except Exception as e:
        print(f"Error creating CSV file: {e}")
        return

    # Initialize environment.
    try:
        env = DroneNavigationAviary(drone_model=DroneModel.CF2X, num_drones=1, gui=True)
        print("Environment initialized successfully!")
    except Exception as e:
        print(f"Error initializing environment: {e}")
        return

    time.sleep(1)
    try:
        obs, _ = env.reset()
        print("Environment reset successfully!")
    except Exception as e:
        print(f"Error resetting environment: {e}")
        return

    dt = 1.0 / env.CTRL_FREQ  # Control update frequency.
    start_time = time.time()
    noise_std = 0.05         # Standard deviation for control noise.

    # Initialize PID variables.
    error_integral = 0.0
    prev_error = 0.0

    # Define strict boundaries: x,y ∈ [-5,5] and z ∈ [0,2].
    bounds_min = np.array([-5, -5, 0])
    bounds_max = np.array([5, 5, 2])

    with open(csv_filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Time", "x", "y", "z", "Roll", "Pitch", "Yaw",
            "vx", "vy", "vz", "wx", "wy", "wz",
            "Target_x", "Target_y", "Target_z", "Error_Dist",
            "RPM_1", "RPM_2", "RPM_3", "RPM_4",
            "Error_Int", "Error_Deri"
        ])
        file.flush()

        try:
            while True:
                current_time = time.time() - start_time

                # Get current state.
                pos = np.array(obs[0:3])
                vel = np.array(obs[7:10])
                orientation = obs[3:7]
                roll, pitch, yaw = p.getEulerFromQuaternion(orientation)

                # Get angular velocities.
                lin_vel, ang_vel = p.getBaseVelocity(env.DRONE_IDS[0])
                if ang_vel is None:
                    wx, wy, wz = 0.0, 0.0, 0.0
                else:
                    wx, wy, wz = ang_vel

                # Use the current dynamic target.
                target_pos = np.array(env.current_target)
                error_dist = np.linalg.norm(target_pos - pos)

                # Update PID integral/derivative.
                error_integral += error_dist * dt
                error_derivative = (error_dist - prev_error) / dt
                prev_error = error_dist

                # Compute RPM values.
                try:
                    rpm_values = env.compute_motor_rpms(0.5, roll, pitch, yaw)
                except Exception as e:
                    print(f"RPM Calculation Error: {e}")
                    rpm_values = [0, 0, 0, 0]

                writer.writerow([
                    current_time,
                    *pos,
                    roll, pitch, yaw,
                    *vel,
                    wx, wy, wz,
                    *target_pos,
                    error_dist,
                    *rpm_values,
                    error_integral,
                    error_derivative
                ])
                file.flush()
                print(f"Logged at {current_time:.2f}s: pos={pos}, target={target_pos}, error={error_dist:.2f}")

                # Enforce boundaries.
                pos_clamped = np.clip(pos, bounds_min, bounds_max)
                if (pos < bounds_min).any() or (pos > bounds_max).any():
                    print("Drone out of bounds! Resetting environment.")
                    p.resetBasePositionAndOrientation(env.DRONE_IDS[0], pos_clamped, orientation)
                    obs, _ = env.reset()
                    continue

                # Generate a randomized but controlled action.
                base_action = np.array([0.5, roll, pitch, yaw])
                noise = np.random.normal(0, noise_std, 4)
                thrust_action = np.clip(base_action[0] + noise[0], 0.0, 1.0)
                roll_action = np.clip(base_action[1] + noise[1], -0.5, 0.5)
                pitch_action = np.clip(base_action[2] + noise[2], -0.5, 0.5)
                yaw_action = np.clip(base_action[3] + noise[3], -0.5, 0.5)
                action = np.array([thrust_action, roll_action, pitch_action, yaw_action])

                # Update the camera to follow the drone.
                p.resetDebugVisualizerCamera(
                    cameraDistance=2.0,
                    cameraYaw=0,
                    cameraPitch=-30,
                    cameraTargetPosition=pos
                )

                obs, reward, done, _, info = env.step(action)
                print("Step Info:", info)

                if done:
                    print("Collision detected! Resetting environment.")
                    obs, _ = env.reset()
                    continue

                time.sleep(dt)

                # (For dynamic target, the environment itself updates the target when reached.)
                if error_dist < env.waypoint_threshold:
                    print("Target reached, new target generated (if applicable).")

                # Optionally, break after a certain time or number of samples.
                if current_time > 2000:  # For instance, running for 2000 seconds.
                    break
        except KeyboardInterrupt:
            print("Data collection interrupted by user!")
        finally:
            env.close()
            print(f"Data successfully saved to {csv_filename}")

if __name__ == "__main__":
    collect_pid_data()
    '''
'''
    /\/\/\/\/\/\/
import os
import time
import numpy as np
import csv
import pybullet as p
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from env1 import DroneNavigationAviary
from gym_pybullet_drones.utils.enums import DroneModel

def collect_pid_data():
    # ===============================
    # Data Collection Setup
    # ===============================
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    raw_csv_filename = os.path.join(data_dir, "pid_data.csv")
    print(f"Raw CSV will be saved at: {raw_csv_filename}")

    # Check CSV writability.
    try:
        with open(raw_csv_filename, "w", newline="", encoding="utf-8") as test_file:
            test_file.write("Test Write\n")
        print("CSV file is writable!")
    except Exception as e:
        print(f"Error creating CSV file: {e}")
        return

    # Initialize Environment.
    try:
        env = DroneNavigationAviary(drone_model=DroneModel.CF2X, num_drones=1, gui=True)
        print("Environment initialized successfully!")
    except Exception as e:
        print(f"Error initializing environment: {e}")
        return

    time.sleep(1)
    try:
        obs, _ = env.reset()
        print("Environment reset successfully!")
    except Exception as e:
        print(f"Error resetting environment: {e}")
        return

    dt = 1.0 / env.CTRL_FREQ  # Control update frequency.
    start_time = time.time()
    noise_std = 0.05         # Standard deviation for control noise.

    # Initialize PID variables.
    error_integral = 0.0
    prev_error = 0.0

    # Define strict boundaries: x,y ∈ [-5,5] and z ∈ [0,2].
    bounds_min = np.array([-5, -5, 0])
    bounds_max = np.array([5, 5, 2])

    # Open CSV file for logging raw data.
    with open(raw_csv_filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Time", "x", "y", "z", "Roll", "Pitch", "Yaw",
            "vx", "vy", "vz", "wx", "wy", "wz",
            "Target_x", "Target_y", "Target_z", "Error_Dist",
            "RPM_1", "RPM_2", "RPM_3", "RPM_4",
            "Error_Int", "Error_Deri"
        ])
        file.flush()

        try:
            while True:
                current_time = time.time() - start_time

                # ---------- Obtain Raw Drone State ----------
                pos = np.array(obs[0:3])
                vel = np.array(obs[7:10])
                orientation = obs[3:7]
                roll, pitch, yaw = p.getEulerFromQuaternion(orientation)

                # Retrieve angular velocities in a try/except block.
                try:
                    lin_vel, ang_vel = p.getBaseVelocity(env.DRONE_IDS[0])
                    if ang_vel is None:
                        wx, wy, wz = 0.0, 0.0, 0.0
                    else:
                        wx, wy, wz = ang_vel
                except Exception as e:
                    print("Error retrieving base velocity (physics server may be disconnected):", e)
                    break  # Exit the loop if we are no longer connected.

                # Use the current dynamic target.
                target_pos = np.array(env.current_target)
                error_dist = np.linalg.norm(target_pos - pos)

                # ---------- Update PID Errors ----------
                error_integral += error_dist * dt
                error_derivative = (error_dist - prev_error) / dt
                prev_error = error_dist

                # ---------- Compute Motor RPMs ----------
                try:
                    rpm_values = env.compute_motor_rpms(0.5, roll, pitch, yaw)
                except Exception as e:
                    print(f"RPM Calculation Error: {e}")
                    rpm_values = [0, 0, 0, 0]

                # Log the current data row.
                writer.writerow([
                    current_time,
                    *pos,
                    roll, pitch, yaw,
                    *vel,
                    wx, wy, wz,
                    *target_pos,
                    error_dist,
                    *rpm_values,
                    error_integral,
                    error_derivative
                ])
                file.flush()
                print(f"Logged at {current_time:.2f}s: pos={pos}, target={target_pos}, error={error_dist:.2f}")

                # ---------- Enforce Boundaries ----------
                pos_clamped = np.clip(pos, bounds_min, bounds_max)
                if (pos < bounds_min).any() or (pos > bounds_max).any():
                    print("Drone out of bounds! Resetting environment.")
                    p.resetBasePositionAndOrientation(env.DRONE_IDS[0], pos_clamped, orientation)
                    try:
                        obs, _ = env.reset()
                    except Exception as e:
                        print("Error during environment reset after boundary violation:", e)
                        break
                    continue  # Skip remaining processing for this cycle.

                # ---------- Generate Randomized but Controlled Action ----------
                base_action = np.array([0.5, roll, pitch, yaw])
                noise = np.random.normal(0, noise_std, size=4)
                thrust_action = np.clip(base_action[0] + noise[0], 0.0, 1.0)
                roll_action = np.clip(base_action[1] + noise[1], -0.5, 0.5)
                pitch_action = np.clip(base_action[2] + noise[2], -0.5, 0.5)
                yaw_action = np.clip(base_action[3] + noise[3], -0.5, 0.5)
                action = np.array([thrust_action, roll_action, pitch_action, yaw_action])

                # ---------- Update Camera ----------
                p.resetDebugVisualizerCamera(
                    cameraDistance=2.0,
                    cameraYaw=0,
                    cameraPitch=-30,
                    cameraTargetPosition=pos
                )

                # ---------- Step the Simulation ----------
                try:
                    obs, reward, done, _, info = env.step(action)
                    print("Step Info:", info)
                except Exception as e:
                    print(f"Step function error: {e}")
                    break

                # If a collision is detected, reset the environment.
                if done:
                    print("Collision detected! Resetting environment.")
                    try:
                        obs, _ = env.reset()
                    except Exception as e:
                        print("Error during environment reset after collision:", e)
                        break
                    continue

                time.sleep(dt)

                # (Dynamic target update is handled internally in env.step() when the target is reached.)
                if error_dist < env.waypoint_threshold:
                    print("Target reached – new target generated (if applicable).")

                # Optionally, break after some duration—here, after 2000 seconds.
                if current_time > 2000:
                    break

        except KeyboardInterrupt:
            print("Data collection interrupted by user!")
        finally:
            # Wrap the environment close in a try/except block.
            try:
                env.close()
            except Exception as e:
                print("Error closing environment:", e)
            print(f"Raw data successfully saved to {raw_csv_filename}")

    # ===============================
    # Post-Collection Preprocessing
    # ===============================
    print("Starting preprocessing of the collected CSV data...")

    try:
        df = pd.read_csv(raw_csv_filename)
    except Exception as e:
        print(f"Error reading CSV data: {e}")
        return

    # --- Angular Velocity Filtering ---
    df[['wx', 'wy', 'wz']] = df[['wx', 'wy', 'wz']].replace(-100, np.nan).interpolate(method='linear')
    print("Angular velocity filtering complete.")

    # --- Input Normalization ---
    features = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    scaler = MinMaxScaler(feature_range=(-1, 1))
    try:
        scaled_array = scaler.fit_transform(df[features])
        df_scaled = pd.DataFrame(scaled_array, columns=[f"{f}_norm" for f in features])
        for col in df_scaled.columns:
            df[col] = df_scaled[col]
        print("Input normalization complete.")
    except Exception as e:
        print(f"Normalization error: {e}")

    # --- Action-Error Alignment ---
    df['Error_Dist_prev'] = df['Error_Dist'].shift(1)
    df['Error_Dist_diff'] = df['Error_Dist'] - df['Error_Dist_prev']
    df['Error_Dist_prev'] = df['Error_Dist_prev'].fillna(method='bfill')
    print("Action-Error alignment complete.")

    preprocessed_csv = os.path.join(data_dir, "pid_data_preprocessed.csv")
    try:
        df.to_csv(preprocessed_csv, index=False)
        print(f"Preprocessed data successfully saved to: {preprocessed_csv}")
    except Exception as e:
        print(f"Error saving preprocessed CSV: {e}")

if __name__ == "__main__":
    collect_pid_data()
'''
'''
import os
import time
import numpy as np
import csv
import pybullet as p
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from env1 import DroneNavigationAviary
from gym_pybullet_drones.utils.enums import DroneModel

def collect_pid_data():
  
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    raw_csv_filename = os.path.join(data_dir, "pid_data.csv")
    print(f"Raw CSV will be saved at: {raw_csv_filename}")

    # Test creation of the CSV file.
    try:
        with open(raw_csv_filename, "w", newline="", encoding="utf-8") as test_file:
            test_file.write("Test Write\n")
        print("CSV file is writable!")
    except Exception as e:
        print(f"Error creating CSV file: {e}")
        return

    # -------------------------------
    # Initialize the Environment
    # -------------------------------
    try:
        env = DroneNavigationAviary(drone_model=DroneModel.CF2X, num_drones=1, gui=True)
        print("Environment initialized successfully!")
    except Exception as e:
        print(f"Error initializing environment: {e}")
        return

    time.sleep(1)
    try:
        obs, _ = env.reset()
        print("Environment reset successfully!")
    except Exception as e:
        print(f"Error resetting environment: {e}")
        return

    dt = 1.0 / env.CTRL_FREQ  # Control update frequency.
    start_time = time.time()
    noise_std = 0.05         # Standard deviation for the control noise.

    # Initialize PID error variables.
    error_integral = 0.0
    prev_error = 0.0

    # Define strict boundaries for the drone (x-y: [-5,5], z: [0,2]).
    bounds_min = np.array([-5, -5, 0])
    bounds_max = np.array([5, 5, 2])

    with open(raw_csv_filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # CSV header.
        writer.writerow([
            "Time", "x", "y", "z", "Roll", "Pitch", "Yaw",
            "vx", "vy", "vz", "wx", "wy", "wz",
            "Target_x", "Target_y", "Target_z", "Error_Dist",
            "RPM_1", "RPM_2", "RPM_3", "RPM_4",
            "Error_Int", "Error_Deri"
        ])
        file.flush()

        try:
            while True:
                current_time = time.time() - start_time

                # --- Get Raw Drone State ---
                pos = np.array(obs[0:3])
                vel = np.array(obs[7:10])
                orientation = obs[3:7]
                roll, pitch, yaw = p.getEulerFromQuaternion(orientation)

                # Retrieve angular velocities with error handling.
                try:
                    lin_vel, ang_vel = p.getBaseVelocity(env.DRONE_IDS[0])
                    if ang_vel is None:
                        wx, wy, wz = 0.0, 0.0, 0.0
                    else:
                        wx, wy, wz = ang_vel
                except Exception as e:
                    print("Error retrieving base velocity (physics server may be disconnected):", e)
                    break  # Exit loop if server is disconnected.

                # Use dynamic target from environment.
                target_pos = np.array(env.current_target)  # Dynamic target.
                error_dist = np.linalg.norm(target_pos - pos)

                # --- Update PID Error Terms ---
                error_integral += error_dist * dt
                error_derivative = (error_dist - prev_error) / dt
                prev_error = error_dist

                # --- Compute Motor RPM Values ---
                try:
                    rpm_values = env.compute_motor_rpms(0.5, roll, pitch, yaw)
                except Exception as e:
                    print(f"RPM Calculation Error: {e}")
                    rpm_values = [0, 0, 0, 0]

                writer.writerow([
                    current_time,
                    *pos,
                    roll, pitch, yaw,
                    *vel,
                    wx, wy, wz,
                    *target_pos,
                    error_dist,
                    *rpm_values,
                    error_integral,
                    error_derivative
                ])
                file.flush()
                print(f"Logged at {current_time:.2f}s: pos={pos}, target={target_pos}, error={error_dist:.2f}")
 
                pos_clamped = np.clip(pos, bounds_min, bounds_max)
                if (pos < bounds_min).any() or (pos > bounds_max).any():
                    print("Drone out of bounds! Resetting environment.")
                    p.resetBasePositionAndOrientation(env.DRONE_IDS[0], pos_clamped, orientation)
                    try:
                        obs, _ = env.reset()  # Also resets dynamic target.
                    except Exception as e:
                        print("Error during environment reset after out-of-bounds:", e)
                        break
                    continue 

                base_action = np.array([0.5, roll, pitch, yaw])
                noise = np.random.normal(0, noise_std, size=4)
                thrust_action = np.clip(base_action[0] + noise[0], 0.0, 1.0)
                roll_action = np.clip(base_action[1] + noise[1], -0.5, 0.5)
                pitch_action = np.clip(base_action[2] + noise[2], -0.5, 0.5)
                yaw_action = np.clip(base_action[3] + noise[3], -0.5, 0.5)
                action = np.array([thrust_action, roll_action, pitch_action, yaw_action])

       
                p.resetDebugVisualizerCamera(
                    cameraDistance=2.0,
                    cameraYaw=0,
                    cameraPitch=-30,
                    cameraTargetPosition=pos
                )


                try:
                    obs, reward, done, _, info = env.step(action)
                    print("Step Info:", info)
                except Exception as e:
                    print(f"Step function error: {e}")
                    break

                # If collision is detected, reset the environment.
                if done:
                    print("Collision detected! Resetting environment.")
                    try:
                        obs, _ = env.reset()
                    except Exception as e:
                        print("Error during environment reset after collision:", e)
                        break
                    continue

                time.sleep(dt)

                if error_dist < env.waypoint_threshold:
                    print("Target reached – new target generated (if applicable).")

                # Stop data collection after a set time (e.g. 2000 seconds).
                if current_time > 2000:
                    break

        except KeyboardInterrupt:
            print("Data collection interrupted by user!")
        finally:
            try:
                env.close()
            except Exception as e:
                print("Error closing environment:", e)
            print(f"Raw data successfully saved to {raw_csv_filename}")

 
    print("Starting preprocessing of the collected CSV data...")

    try:
        df = pd.read_csv(raw_csv_filename)
    except Exception as e:
        print(f"Error reading CSV data: {e}")
        return

    df[['wx', 'wy', 'wz']] = df[['wx', 'wy', 'wz']].replace(-100, np.nan).interpolate(method='linear')
    # Smooth noisy features using a rolling mean (window=3).
    for col in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
        df[col] = df[col].rolling(window=3, min_periods=1).mean()
    print("Angular velocity filtering and smoothing complete.")

    features = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    scaler = MinMaxScaler(feature_range=(-1, 1))
    try:
        scaled_array = scaler.fit_transform(df[features])
        df_scaled = pd.DataFrame(scaled_array, columns=[f"{f}_norm" for f in features])
        for col in df_scaled.columns:
            df[col] = df_scaled[col]
        print("Input normalization complete.")
    except Exception as e:
        print(f"Normalization error: {e}")

    df['Error_Dist_prev'] = df['Error_Dist'].shift(1)
    df['Error_Dist_diff'] = df['Error_Dist'] - df['Error_Dist_prev']
    df['Error_Dist_prev'] = df['Error_Dist_prev'].fillna(method='bfill')
    print("Action-Error alignment complete.")

    df['RPM_avg'] = df[['RPM_1', 'RPM_2', 'RPM_3', 'RPM_4']].mean(axis=1)
    # Create bins (e.g. 5 bins) from the RPM average.
    df['RPM_bin'] = pd.cut(df['RPM_avg'], bins=5, duplicates='drop')
    #
    group_counts = df['RPM_bin'].value_counts()
    if group_counts.empty:
        print("Warning: No RPM bins found; skipping stratified sampling.")
        df_stratified = df.copy()
    else:
        min_count = group_counts.min()
        # Use stratified sampling: sample equal number of rows from each RPM bin.
        df_stratified = df.groupby('RPM_bin', group_keys=False).apply(lambda x: x.sample(n=min_count, random_state=42))
    print("Stratified sampling complete.")

    
    preprocessed_csv = os.path.join(data_dir, "pid_data_preprocessed_stratified.csv")
    try:
        df_stratified.to_csv(preprocessed_csv, index=False)
        print(f"Preprocessed (stratified) data successfully saved to: {preprocessed_csv}")
    except Exception as e:
        print(f"Error saving preprocessed CSV: {e}")

if __name__ == "__main__":
    collect_pid_data()
'''
import os
import time
import numpy as np
import csv
import pybullet as p
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from env1 import DroneNavigationAviary
from gym_pybullet_drones.utils.enums import DroneModel


def collect_pid_data():
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    raw_csv_filename = os.path.join(data_dir, "pid_data.csv")
    print(f"Raw CSV will be saved at: {raw_csv_filename}")

    try:
        with open(raw_csv_filename, "w", newline="", encoding="utf-8") as test_file:
            test_file.write("Test Write\n")
        print("CSV file is writable!")
    except Exception as e:
        print(f"Error creating CSV file: {e}")
        return

    try:
        env = DroneNavigationAviary(drone_model=DroneModel.CF2X, num_drones=1, gui=True, demo_mode=False)
        print("Environment initialized successfully!")
    except Exception as e:
        print(f"Error initializing environment: {e}")
        return

    time.sleep(1)

    try:
        obs, _ = env.reset()
        print("Environment reset successfully!")
    except Exception as e:
        print(f"Error resetting environment: {e}")
        return

    dt = 1.0 / env.CTRL_FREQ
    start_time = time.time()
    noise_std = 0.05
    error_integral = 0.0
    prev_error = 0.0

    bounds_min = np.array([-5, -5, 0])
    bounds_max = np.array([5, 5, 2])

    with open(raw_csv_filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Time", "x", "y", "z", "Roll", "Pitch", "Yaw", "vx", "vy", "vz",
            "wx", "wy", "wz", "Target_x", "Target_y", "Target_z", "Error_Dist",
            "RPM_1", "RPM_2", "RPM_3", "RPM_4", "Error_Int", "Error_Deri"
        ])
        file.flush()

        try:
            while True:
                current_time = time.time() - start_time
                pos = np.array(obs[0:3])
                vel = np.array(obs[7:10])
                orientation = obs[3:7]
                roll, pitch, yaw = p.getEulerFromQuaternion(orientation)

                try:
                    lin_vel, ang_vel = p.getBaseVelocity(env.DRONE_IDS[0])
                    if ang_vel is None:
                        wx, wy, wz = 0.0, 0.0, 0.0
                    else:
                        wx, wy, wz = ang_vel
                except Exception as e:
                    print("Error retrieving base velocity:", e)
                    break

                target_pos = np.array(env.current_target)
                error_dist = np.linalg.norm(target_pos - pos)
                error_integral += error_dist * dt
                error_derivative = (error_dist - prev_error) / dt
                prev_error = error_dist

                try:
                    rpm_values = env.compute_motor_rpms(0.5, roll, pitch, yaw)
                except Exception as e:
                    print(f"RPM Calculation Error: {e}")
                    rpm_values = [0, 0, 0, 0]

                writer.writerow([
                    current_time, *pos, roll, pitch, yaw, *vel, wx, wy, wz,
                    *target_pos, error_dist, *rpm_values, error_integral, error_derivative
                ])
                file.flush()

                print(f"Logged at {current_time:.2f}s: pos={pos}, target={target_pos}, error={error_dist:.2f}")

                pos_clamped = np.clip(pos, bounds_min, bounds_max)
                if (pos < bounds_min).any() or (pos > bounds_max).any():
                    print("Drone out of bounds! Resetting environment.")
                    p.resetBasePositionAndOrientation(env.DRONE_IDS[0], pos_clamped, orientation)

                    try:
                        obs, _ = env.reset()
                    except Exception as e:
                        print("Error during environment reset after out-of-bounds:", e)
                        break
                    continue

                base_action = np.array([0.5, roll, pitch, yaw])
                noise = np.random.normal(0, noise_std, size=4)
                thrust_action = np.clip(base_action[0] + noise[0], 0.0, 1.0)
                roll_action = np.clip(base_action[1] + noise[1], -0.5, 0.5)
                pitch_action = np.clip(base_action[2] + noise[2], -0.5, 0.5)
                yaw_action = np.clip(base_action[3] + noise[3], -0.5, 0.5)

                action = np.array([thrust_action, roll_action, pitch_action, yaw_action])

                p.resetDebugVisualizerCamera(
                    cameraDistance=2.0, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=pos
                )

                try:
                    obs, reward, done, _, info = env.step(action)
                    print("Step Info:", info)
                except Exception as e:
                    print(f"Step function error: {e}")
                    break

                if done:
                    print("Collision detected! Resetting environment.")
                    try:
                        obs, _ = env.reset()
                    except Exception as e:
                        print("Error during environment reset after collision:", e)
                        break

                time.sleep(dt)

                if error_dist < env.waypoint_threshold:
                    print("Target reached – new target generated.")

                if current_time > 2000:
                    break

        except KeyboardInterrupt:
            print("Data collection interrupted by user!")

        finally:
            try:
                env.close()
            except Exception as e:
                print("Error closing environment:", e)

    print(f"Raw data successfully saved to {raw_csv_filename}")
    print("Starting preprocessing of the collected CSV data...")

    try:
        df = pd.read_csv(raw_csv_filename)
    except Exception as e:
        print(f"Error reading CSV data: {e}")
        return

    df[['wx', 'wy', 'wz']] = df[['wx', 'wy', 'wz']].replace(-100, np.nan).interpolate(method='linear')

    for col in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
        df[col] = df[col].rolling(window=3, min_periods=1).mean()

    print("Angular velocity filtering and smoothing complete.")

    features = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    scaler = MinMaxScaler(feature_range=(-1, 1))

    try:
        scaled_array = scaler.fit_transform(df[features])
        df_scaled = pd.DataFrame(scaled_array, columns=[f"{f}_norm" for f in features])

        for col in df_scaled.columns:
            df[col] = df_scaled[col]

        print("Input normalization complete.")

    except Exception as e:
        print(f"Normalization error: {e}")

    df['Error_Dist_prev'] = df['Error_Dist'].shift(1)
    df['Error_Dist_diff'] = df['Error_Dist'] - df['Error_Dist_prev']
    df['Error_Dist_prev'] = df['Error_Dist_prev'].fillna(method='bfill')

    print("Action-Error alignment complete.")

    df['RPM_avg'] = df[['RPM_1', 'RPM_2', 'RPM_3', 'RPM_4']].mean(axis=1)
    df['RPM_bin'] = pd.cut(df['RPM_avg'], bins=5, duplicates='drop')

    group_counts = df['RPM_bin'].value_counts()

    if group_counts.empty:
        print("Warning: No RPM bins found; skipping stratified sampling.")
        df_stratified = df.copy()
    else:
        min_count = group_counts.min()
        df_stratified = df.groupby('RPM_bin', group_keys=False).apply(lambda x: x.sample(n=min_count, random_state=42))

    print("Stratified sampling complete.")

    preprocessed_csv = os.path.join(data_dir, "pid_data_preprocessed_stratified.csv")

    try:
        df_stratified.to_csv(preprocessed_csv, index=False)
        print(f"Preprocessed (stratified) data successfully saved to: {preprocessed_csv}")
    except Exception as e:
        print(f"Error saving preprocessed CSV: {e}")


if __name__ == "__main__":
    collect_pid_data()
