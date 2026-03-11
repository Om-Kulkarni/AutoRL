"""MuJoCo environment for the Franka Emika Panda arm with a gripper."""

import os
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class PandaEnv(gym.Env):
    """Franka Panda environment for manipulation tasks.

    Input State (Observations):
        - Joint positions (qpos)
        - Joint velocities (qvel)
        - Gripper state
        - Image from a fixed overhead camera

    Output State (Actions):
        - Actuator controls for the arm and gripper joints.
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, render_mode: Optional[str] = None):
        """Initializes the Panda environment."""
        super().__init__()
        self.render_mode = render_mode

        # Determine the path to the scene XML
        scene_path = os.path.join(
            os.path.dirname(__file__), "assets", "scene.xml"
        )

        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        
        self.observation_space = spaces.Dict(
            {
                "qpos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.model.nq,), dtype=np.float32
                ),
                "qvel": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.model.nv,), dtype=np.float32
                ),
                "image": spaces.Box(
                    low=0, high=255, shape=(240, 320, 3), dtype=np.uint8
                ),
            }
        )
        
        self.renderer = mujoco.Renderer(self.model, height=240, width=320)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Takes a step in the environment."""
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Gets the current observation."""
        self.renderer.update_scene(self.data, camera="overhead_cam")
        image = self.renderer.render()
        
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "image": image,
        }

    def render(self) -> Optional[np.ndarray]:
        """Renders the environment."""
        if self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data, camera="overhead_cam")
            return self.renderer.render()
        return None
