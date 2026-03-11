import os
import sys

# Add the project root to the python path so we can import 'env'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from env.panda_env import PandaEnv
import numpy as np
import time
from PIL import Image

try:
    import mujoco.viewer
    VIEWER_AVAILABLE = True
except ImportError:
    VIEWER_AVAILABLE = False
    print("WARNING: mujoco.viewer.launch requires GUI access. Running headless tests instead.")

def test_headless():
    """Validates the gym environment API conforms to expectations."""
    print("Testing Gym API...")
    env = PandaEnv()
    obs, info = env.reset()
    
    assert "qpos" in obs
    assert "qvel" in obs
    assert "image" in obs
    
    print("Observation space details:")
    print(f"qpos shape: {obs['qpos'].shape}")
    print(f"qvel shape: {obs['qvel'].shape}")
    print(f"image shape: {obs['image'].shape}")
    
    assert env.action_space.shape[0] == env.model.nu, f"Action space {env.action_space.shape[0]} != model.nu {env.model.nu}"
    print(f"Action space dimension: {env.action_space.shape[0]}")
    
    # Run a few randomized steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
            
    print("Gym API Test Passed!")

def test_camera():
    """Validates the camera rendering by popping open the first frame."""
    print("Testing Camera Rendering...")
    env = PandaEnv()
    obs, info = env.reset()
    
    print("Opening Camera Frame using PIL...")
    img = Image.fromarray(obs['image'])
    img.show()
    print("Camera Test Passed!")

def test_viewer():
    """Launches the interactive viewer to debug physics."""
    if not VIEWER_AVAILABLE:
        print("Viewer unavailable. Skipping interactive test.")
        return
        
    print("Launching interactive viewer. Press ESC to close.")
    env = PandaEnv(render_mode="human")
    obs, info = env.reset()
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        start_time = time.time()
        while viewer.is_running() and time.time() - start_time < 10.0: # Run for 10 seconds
            step_start = time.time()
            
            # Simple policy: just apply zeros
            action = np.zeros(env.action_space.shape)
            env.step(action)
            
            viewer.sync()
            
            # Wait to match real time
            time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
    print("Viewer Test Completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test MuJoCo Environment")
    parser.add_argument("--viewer", action="store_true", help="Launch the interactive viewer")
    parser.add_argument("--show-camera", action="store_true", help="Show the first frame of the camera using PIL")
    args = parser.parse_args()
    
    test_headless()
    
    if args.viewer:
        test_viewer()
        
    if args.show_camera:
        test_camera()
        
    print("\nAll tests ran successfully!")
