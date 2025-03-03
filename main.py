'''import time
from env1 import SimpleDroneNavigationAviary
from gym_pybullet_drones.utils.enums import DroneModel
import pybullet as p

def main():
    # Initialize the environment
    env = SimpleDroneNavigationAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        gui=True
    )

    env.reset()

    # Adjust camera for better visualization
    p.resetDebugVisualizerCamera(
        cameraDistance=6,   # Distance from the scene
        cameraYaw=45,       # Horizontal rotation
        cameraPitch=-30,    # Vertical rotation
        cameraTargetPosition=[0, 0, 0]  # Center focus
    )

    print("Environment Visualization Started")
    print("Press Ctrl+C to exit")

    try:
        while True:
            time.sleep(0.1)  # Keep running
            
    except KeyboardInterrupt:
        print("\nVisualization ended by user")
    
    env.close()

if __name__ == "__main__":
    main()
'''
import time
from env1 import DroneNavigationAviary  # ✅ Correct class name
from gym_pybullet_drones.utils.enums import DroneModel
import pybullet as p

def main():
    # Initialize the environment
    env = DroneNavigationAviary(  # ✅ Updated class name
        drone_model=DroneModel.CF2X,
        num_drones=1,
        gui=True
    )

    env.reset()

    # Adjust camera for better visualization
    p.resetDebugVisualizerCamera(
        cameraDistance=6,   # Distance from the scene
        cameraYaw=45,       # Horizontal rotation
        cameraPitch=-30,    # Vertical rotation
        cameraTargetPosition=[0, 0, 0]  # Center focus
    )

    print("Environment Visualization Started")
    print("Press Ctrl+C to exit")

    try:
        while True:
            time.sleep(0.1)  # Keep running
            
    except KeyboardInterrupt:
        print("\nVisualization ended by user")
    
    env.close()

if __name__ == "__main__":
    main()
