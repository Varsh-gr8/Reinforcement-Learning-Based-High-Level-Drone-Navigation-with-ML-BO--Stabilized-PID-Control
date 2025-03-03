'''
import time
import numpy as np
import pybullet as p
import pybullet_data

# ----------------- PID Controller for Altitude -----------------
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0.0
        self.integral = 0.0
        
    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# ----------------- Initialize PyBullet -----------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)
dt = 1.0 / 240.0
p.setTimeStep(dt)

# Load ground plane.
plane = p.loadURDF("plane.urdf")

# ✅ FIXED: Correct URDF path
drone_urdf_path = r""
start_pos = [0, 0, 1.0]

try:
    drone = p.loadURDF(drone_urdf_path, basePosition=start_pos)
    print("✅ Drone URDF loaded successfully!")
except:
    raise FileNotFoundError(f"Error: Could not load URDF at {drone_urdf_path}")

# ----------------- Altitude Controller Setup -----------------
pid_altitude = PIDController(Kp=10.0, Ki=0.05, Kd=3.0)
target_altitude = 2.0  # Desired hover altitude

force_scaling = 5.0  # Converts PID output into Newtons

# ----------------- Simulation Loop (External Force Application) -----------------
for i in range(1000):
    pos, _ = p.getBasePositionAndOrientation(drone)
    current_alt = pos[2]
    error_alt = target_altitude - current_alt
    
    # Compute PID output.
    control_output = pid_altitude.compute(error_alt, dt)
    
    # Convert PID output to a force (in Newtons).
    force = np.clip(force_scaling * control_output, -100, 100)
    
    # Debug prints for monitoring the altitude.
    print(f"Step {i:03d}: Altitude = {current_alt:.3f}, Error = {error_alt:.3f}, PID = {control_output:.3f}, Force = {force:.3f}")
    
    # Apply external force in the upward (z) direction.
    p.applyExternalForce(objectUniqueId=drone,
                         linkIndex=-1, 
                         forceObj=[0, 0, force],
                         posObj=pos,
                         flags=p.WORLD_FRAME)
    
    p.stepSimulation()
    time.sleep(dt)

p.disconnect()
'''
'''
/\/\/\/\/\/
import pybullet as p
import pybullet_data
import time
import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=2.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0

    def compute(self, current_value, dt):
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.prev_error = error
        return output

def setup_simulation():
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    plane_id = p.loadURDF("plane.urdf")
    drone_id = p.loadURDF("C:/Users/varsh/OneDrive/Documents/Drone_navi/gym-pybullet-drones/gym_pybullet_drones/assets/cf2x.urdf", [0, 0, 0.2], useFixedBase=False)


    return drone_id

def apply_control(drone_id, thrust):
    max_force = 50  # Limit max thrust to prevent instability
    force = np.clip(thrust, 0, max_force)
    for i in range(4):  # Apply force to all 4 rotors
        p.applyExternalForce(drone_id, i, (0, 0, force), (0, 0, 0), p.LINK_FRAME)

def main():
    drone_id = setup_simulation()
    pid = PIDController(Kp=10, Ki=0.5, Kd=5, setpoint=2.0)

    dt = 1 / 240.0
    time.sleep(1)  # Allow physics engine to stabilize

    for step in range(1000):
        pos, _ = p.getBasePositionAndOrientation(drone_id)
        altitude = pos[2]

        thrust = pid.compute(altitude, dt)
        apply_control(drone_id, thrust)

        print(f"Step {step}, Altitude: {altitude:.2f}, Thrust: {thrust:.2f}")
        p.stepSimulation()
        time.sleep(dt)

    p.disconnect()

if __name__ == "__main__":
    main()
'''
'''
import pybullet as p
import pybullet_data
import time

# Connect to PyBullet
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set gravity
gravity_value = (0, 0, -9.81)
p.setGravity(*gravity_value)

# Load plane and drone
plane_id = p.loadURDF("plane.urdf")
drone_path = "C:/Users/varsh/OneDrive/Documents/Drone_navi/gym-pybullet-drones/gym_pybullet_drones/assets/cf2x.urdf"
drone_id = p.loadURDF(drone_path, [0, 0, 0.1], useFixedBase=False)

# Drone properties
target_altitude = 1.0
mass = 0.027  # From getDynamicsInfo
hover_thrust = mass * abs(gravity_value[2])  # Approximate thrust needed to hover

# PID Controller parameters (Adjusted for smoother ascent)
Kp = 10.0  # Lowered to reduce initial thrust spike
Ki = 0.2   # Slight integral effect
Kd = 5.0   # Higher damping effect

error_sum = 0
last_error = 0
dt = 0.01  # Time step

p.setRealTimeSimulation(0)

for step in range(500):
    pos, _ = p.getBasePositionAndOrientation(drone_id)
    altitude = pos[2]
    
    error = target_altitude - altitude
    error_sum += error * dt
    error_derivative = (error - last_error) / dt
    last_error = error

    # PID Control for thrust
    thrust = hover_thrust + (Kp * error) + (Ki * error_sum) + (Kd * error_derivative)
    
    # Keep thrust within reasonable limits
    thrust = max(hover_thrust * 0.8, min(thrust, hover_thrust * 1.5))  # Prevent extreme thrust

    p.applyExternalForce(drone_id, -1, [0, 0, thrust], [0, 0, 0], p.LINK_FRAME)
    
    p.stepSimulation()
    print(f"Step {step}, Altitude: {altitude:.2f}, Thrust: {thrust:.2f}")

    if abs(error) < 0.01 and abs(error_derivative) < 0.01:  # Stop if altitude stabilizes
        print("Drone stabilized at target altitude!")
        break

    time.sleep(dt)

p.disconnect()
'''
import pybullet as p
import pybullet_data
import time
import numpy as np

# Connect to PyBullet
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set gravity
gravity_value = (0, 0, -9.81)
p.setGravity(*gravity_value)

# Load plane and drone
plane_id = p.loadURDF("plane.urdf")
drone_path = "C:/Users/varsh/OneDrive/Documents/Drone_navi/gym-pybullet-drones/gym_pybullet_drones/assets/cf2x.urdf"
drone_id = p.loadURDF(drone_path, [0, 0, 0.1], useFixedBase=False)

# Drone properties
target_altitude = 1.0
mass = 0.027
hover_thrust = mass * abs(gravity_value[2])

# PID Controller parameters
Kp = 10.0
Ki = 0.2
Kd = 5.0

error_sum = 0
last_error = 0
dt = 0.01

p.setRealTimeSimulation(0)

for step in range(500):
    pos, _ = p.getBasePositionAndOrientation(drone_id)
    altitude = pos[2]

    error = target_altitude - altitude
    error_sum += error * dt
    error_derivative = (error - last_error) / dt
    last_error = error

    thrust = hover_thrust + (Kp * error) + (Ki * error_sum) + (Kd * error_derivative)
    thrust = max(hover_thrust * 0.8, min(thrust, hover_thrust * 1.5))

    p.applyExternalForce(drone_id, -1, [0, 0, thrust], [0, 0, 0], p.LINK_FRAME)
    p.stepSimulation()

    print(f"Step {step}, Altitude: {altitude:.2f}, Thrust: {thrust:.2f}")

    if abs(error) < 0.01 and abs(error_derivative) < 0.01:
        print("Drone stabilized at target altitude!")
        break

    time.sleep(dt)

p.disconnect()
