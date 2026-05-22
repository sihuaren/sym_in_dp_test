import os
import sys
import draccus
import numpy as np
import torch
import yaml
from PIL import Image
from collections import deque
from termcolor import colored
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from websocket import websocket_policy_server
from websocket import base_policy as _base_policy
from typing_extensions import override
import logging

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.franka_fr3_model import create_model

## rotation utils
from scipy.spatial.transform import Rotation as R
def normalize_vector(v):
    v_mag = np.linalg.norm(v, axis=-1, keepdims=True)
    v_mag = np.maximum(v_mag, 1e-8)
    return v / v_mag

def cross_product(u, v):
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = np.stack((i, j, k), axis=1)
    return out

def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]
        
    x = normalize_vector(x_raw)
    z = cross_product(x, y_raw)
    z = normalize_vector(z)
    y = cross_product(z, x)
    
    x = x.reshape(-1, 3, 1)
    y = y.reshape(-1, 3, 1)
    z = z.reshape(-1, 3, 1)
    matrix = np.concatenate((x, y, z), axis=2)
    return matrix

def convert_rotation_matrix_to_euler(rotmat):
    """
    Convert rotation matrix (3x3) to Euler angles (rpy).
    """
    r = R.from_matrix(rotmat)
    euler = r.as_euler('xyz', degrees=False)
    
    return euler

def convert_10d_to_7d(actions_10d):
    """
    快速将10D动作转换为7D动作
    """

    rotation_matrices = compute_rotation_matrix_from_ortho6d(actions_10d[:, 3:9])
    euler_angles = convert_rotation_matrix_to_euler(rotation_matrices)
    
    return np.concatenate([
        actions_10d[:, 0:3],
        euler_angles,
        actions_10d[:, 9:10]
    ], axis=1)

##

@dataclass
class GenerateConfig:
    # fmt: off
    port: int = 5001
    TASK_NAME: str = "goal_carrot_yellow_bowl"
    # Model parameters
    config: str = "/data/huangliqi/RoboticsDiffusionTransformer/configs/base.yaml"
    pretrained_model_name_or_path: str = "/hard_data/user_dataset/huangliqi_dataset/rdt-checkpoints/rdt-finetune-170m-bs24-fankafr3/checkpoint-100000/pytorch_model/mp_rank_00_model_states.pt"
    lang_embeddings_path: str = f"/hard_data/user_dataset/huangliqi_dataset/rdt_franka_fr3/{TASK_NAME}/lang_embed_0.pt"
    ctrl_freq: int = 20
    
    # Inference parameters
    chunk_size: int = 64 # RDT chunk size

    # Utils
    seed: int = 7
    # fmt: on

def initialize_model(cfg: GenerateConfig):
    """Initialize RDT model."""
    with open(cfg.config, "r") as fp:
        config = yaml.safe_load(fp)
    
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    
    print(f"Creating model with config from {cfg.config}...")
    model = create_model(
        args=config, 
        dtype=torch.bfloat16, 
        pretrained=cfg.pretrained_model_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=cfg.ctrl_freq,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device) # create_model might handle this, but ensure it
    
    # Load language embeddings
    lang_embeddings = None
    if cfg.lang_embeddings_path and os.path.exists(cfg.lang_embeddings_path):
        print(f"Loading language embeddings from {cfg.lang_embeddings_path}...")
        loaded_data = torch.load(cfg.lang_embeddings_path)
        if isinstance(loaded_data, dict) and "embeddings" in loaded_data:
            lang_embeddings = loaded_data["embeddings"]
        elif isinstance(loaded_data, torch.Tensor):
            lang_embeddings = loaded_data
        
        if isinstance(lang_embeddings, torch.Tensor):
            lang_embeddings = lang_embeddings.to(device=device, dtype=torch.bfloat16)
            if lang_embeddings.dim() == 2:
                lang_embeddings = lang_embeddings.unsqueeze(0)
    else:
        print("Using random embeddings.")
        lang_embeddings = torch.randn(1, 16, 4096).to(device=device, dtype=torch.bfloat16)
        
    return model, lang_embeddings

class Policy(_base_policy.BasePolicy):
    def __init__(self, cfg, model, lang_embeddings):
        self.cfg = cfg
        self.model = model
        self.lang_embeddings = lang_embeddings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # History buffer for observations (RDT needs history)
        # We'll maintain a small history here if needed, but RDT usually takes 
        # a sequence of images. simple_inference.py uses 2 frames.
        self.obs_history = deque(maxlen=2)

    @override
    def infer(self, obs):
        """
        obs structure from client:
        {
            "observation": {
                "left_image": np.array,
                "wrist_image": np.array,
                "qpos": np.array,      # Added in client
                "gripper_width": float # Added in client
            },
            "task_description": str
        }
        """
        outputs = {}
        observation = obs
        
        # 1. Process Images
        # Client sends numpy arrays (H, W, 3)
        img_high = observation["left_image"]
        img_right_wrist = observation["wrist_image"]
        img_left_wrist = np.zeros_like(img_high) # Dummy
        
        # 2. Process Proprioception
        # RDT expects [qpos (7), gripper_width (1)] -> 8 dim
        qpos = observation.get("qpos")
        gripper_width = observation.get("gripper_width")
        
        if qpos is None:
             # Fallback if client not updated yet (should not happen if we update client)
             # Try to use gripper_state (EEF) if that's what we have, but RDT needs joints.
             # Assuming client sends qpos.
             raise ValueError("qpos missing in observation")
             
        # Ensure shapes
        qpos = np.array(qpos)
        gripper_width = np.array([gripper_width])
        
        current_proprio = np.concatenate([qpos, gripper_width], axis=0) # (8,)
        
        # 3. Update History
        current_obs_entry = {
            'images': {
                'cam_high': img_high,
                'cam_right_wrist': img_right_wrist,
                'cam_left_wrist': img_left_wrist
            },
            'proprio': torch.from_numpy(current_proprio).float().to(self.device)
        }
        
        if len(self.obs_history) == 0:
            self.obs_history.append(current_obs_entry)
            self.obs_history.append(current_obs_entry)
        else:
            self.obs_history.append(current_obs_entry)
            
        # 4. Prepare Inputs for Model
        camera_names = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']
        image_arrs = []
        for entry in self.obs_history:
            for cam_name in camera_names:
                image_arrs.append(entry['images'][cam_name])
        
        images = [Image.fromarray(arr) for arr in image_arrs]
        proprio = self.obs_history[-1]['proprio'].unsqueeze(0) # (1, 8)
        
        # 5. Run Inference
        with torch.inference_mode():
            actions = self.model.step(
                proprio=proprio,
                images=images,
                text_embeds=self.lang_embeddings
            )
            # actions: (1, Chunk, Dim) -> (Chunk, Dim)
            actions = actions.squeeze(0).cpu().numpy()
            actions = convert_10d_to_7d(actions) # Convert to 7D
        # Return actions
        # Client expects "actions" to be a list or array
        outputs["actions"] = actions.tolist() 
        outputs["predicted_trajs"] = actions.tolist() # For visualization
        
        return outputs
    
@draccus.wrap()
def run(cfg: GenerateConfig):
    model, lang_embeddings = initialize_model(cfg)
    policy = Policy(cfg, model, lang_embeddings)
    print(colored("Starting websocket policy server...", "green"))
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=cfg.port,
    )
    server.serve_forever()



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    run()
