import os
from pathlib import Path
import sys
# 获取当前脚本的绝对路径
current_file = Path(__file__).resolve()
# 找到项目根目录 (假设 algo/ 在项目根目录下，所以是 .parent.parent)
project_root = current_file.parent.parent
# 将根目录添加到 Python 搜索路径中
sys.path.append(str(project_root))
import cv2
import time
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from typing import Optional
from roby.hardware.cameras.realsense.camera_realsense import RealSenseCameraConfig, RealSenseCamera
from roby.hardware.robots.fr3.robot_fr3 import FR3Robot, FR3RobotConfig, FR3RobotAction, FR3ActionMode
from utils import websocket_client_policy
from PIL import Image
from algo.utils.action_mapping import (
    delta_joint_action_mapping,
    absolute_position_action_mapping,
    absolute_joint_action_mapping,
    delta_absolute_position_action_mapping,
)

@dataclass
class FR3Config:
    # --- robot setting ---
    robot_id: str = "fr3"
    robot_ip: str = "172.16.0.2"
    load_gripper: bool = True
    relative_dynamics_factor: float = 0.05
    buffer_size: int = 10
    home: bool = True                 

    # --- camera setting ---
    scene_camera_id: Optional[str] = "938422072347" # left camera oringinal
    # scene_camera_id: Optional[str] = "112322077048" # left camera new
    wrist_camera_id: Optional[str] = "112322074840"
    fps: int = 15
    width: int = 640
    height: int = 480
    camera_buffer: int = 5

    # --- control mode ---
    action_mode: str = "POSITION_DELTA"  # ["POSITION_DELTA", "JOINT_DELTA", "POSITION_ABSOLUTE", "JOINT_ABSOLUTE"]
    img_update_rate: int = 15            
    asynchronous: bool = False           # move asynchronous


@dataclass
class TaskConfig:
    algorithm: str = "vla"
    is_online: bool = True
    server_host: str = "10.184.17.133"
    server_port: int = 5001
    object_to_manipulate: str = "banana"  # banana, carrot, oranges

class RobotClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(cfg.server_host, cfg.server_port)

    def inference(self, img_resized, wrist_img_resized, gripper_state, gripper_width, qpos, language_instruction):
        obs = {
            "left_image": img_resized, 
            "wrist_image": wrist_img_resized, 
            "gripper_state": gripper_state, 
            "gripper_width": gripper_width, 
            "qpos": qpos,
            "task_description": language_instruction
        }

        print(f"task description is {obs['task_description']}")
        actions = self.policy_client.infer(obs)["actions"]
        trajectories = self.policy_client.infer(obs)["predicted_trajs"]
        return actions, trajectories

class shw_franka:
    def __init__(self, robot_cfg, task_cfg):
        self.current_gripper_state = 0  # -1: open, 1: closed (跟踪当前夹爪实际状态)
        self.target_gripper_state = 0   # 0: open, 1: closed (策略要求的目标状态)
        self.cfg = robot_cfg
        self.action_mode = robot_cfg.action_mode
        self.delta_position_action_mapping = delta_absolute_position_action_mapping.__get__(self)
        self.delta_joint_action_mapping = delta_joint_action_mapping.__get__(self)
        self.absolute_position_action_mapping = absolute_position_action_mapping.__get__(self)
        self.absolute_joint_action_mapping = absolute_joint_action_mapping.__get__(self)
        self.action_mapping = {
            "POSITION_DELTA": self.delta_position_action_mapping,
            "JOINT_DELTA": self.delta_joint_action_mapping,
            "POSITION_ABSOLUTE": self.absolute_position_action_mapping,
            "JOINT_ABSOLUTE": self.absolute_joint_action_mapping,
        }
        self.robot = None
        self.client = RobotClient(task_cfg)
        self.object_to_manipulate = task_cfg.object_to_manipulate
        if self.object_to_manipulate == "banana":
            self.language_instruction = "place banana into red bowl"
        elif self.object_to_manipulate == "carrot":
            self.language_instruction = "pick up carrot and place it into red bowl"
        elif self.object_to_manipulate == "orange":
            self.language_instruction = "grasp orange and put it into red bowl"
        else:
            raise ValueError("Unsupported object to manipulate")
        # self.language_instruction = task_cfg.language_instruction
        self.scene_camera = None
        self.wrist_camera = None
        self.scene_camera_id = robot_cfg.scene_camera_id
        self.wrist_camera_id = robot_cfg.wrist_camera_id
        self.scene_camera_image = None
        self.wrist_camera_image = None
        self._img_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._threads = []
        self.devices = self.setup_hardwares()

    def setup_hardwares(self):
        robot_conf = FR3RobotConfig(
            id=self.cfg.robot_id,
            robot_ip=self.cfg.robot_ip,
            load_gripper=self.cfg.load_gripper,
            relative_dynamics_factor=self.cfg.relative_dynamics_factor,
            buffer_size=self.cfg.buffer_size,
            initial_joint = None,
            initial_end_pose=[0.4707543519985666, -0.02776023399946076, 0.29950666236991815, 1.0, 0.0, 0.0, 0.0]
        )
        self.robot = FR3Robot(robot_conf)
        self.robot.connect()
        self.robot.read_state()
        self.robot._start_read_thread()
        if self.cfg.home:
            self.robot.home()
            self.robot.gripper.open(0.1)
            self.current_gripper_state = 0  # 初始状态为开
        
        def create_camera(name, serial):
            if serial is None:
                return None
            cam_cfg = RealSenseCameraConfig(
                fps=self.cfg.fps,
                width=self.cfg.width,
                height=self.cfg.height,
                buffer_size=self.cfg.camera_buffer,
                serial_number_or_name=serial,
            )
            cam = RealSenseCamera(cam_cfg)
            cam.connect()
            cam._start_read_thread()
            record_devices[name] = cam
            return cam
        
        record_devices = {"fr3": self.robot}
        self.scene_camera = create_camera("scene_camera", self.cfg.scene_camera_id)
        self.wrist_camera = create_camera("wrist_camera", self.cfg.wrist_camera_id)
        return record_devices

    def move(self, action):
        action = self.action_mapping[self.action_mode](action)
        self.robot.send_action(action, asynchronous=self.cfg.asynchronous)

    def update_images(self):
        period = 1.0 / max(1, int(self.cfg.img_update_rate))
        next_t = time.time()
        while not self._stop_event.is_set():
            with self._img_lock:
                if self.scene_camera:
                    if not self.scene_camera.frame_buffer.empty():
                        img = self.scene_camera.frame_buffer.queue[-1].color
                        self.scene_camera_image = cv2.resize(img, (1280, 720))
                if self.wrist_camera:
                    if not self.wrist_camera.frame_buffer.empty():
                        img = self.wrist_camera.frame_buffer.queue[-1].color
                        self.wrist_camera_image = cv2.resize(img, (1280, 720))
            next_t += period
            dt = next_t - time.time()
            time.sleep(max(dt, 0.001))
    
    def _to_numpy_uint8_rgb(self, data: np.ndarray) -> np.ndarray:
        """Convert raw image to HxWx3 uint8 RGB."""
        resize_to = (256, 256)

        def _center_crop_pil(pil_img: Image.Image) -> Image.Image:
            w, h = pil_img.size
            s = min(w, h)
            left = (w - s) // 2
            top = (h - s) // 2
            return pil_img.crop((left, top, left + s, top + s))

        # 处理numpy array
        if data.ndim == 3:
            data_rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(data_rgb, 'RGB')
        
        if pil is None:
            return np.zeros((resize_to[0], resize_to[1], 3), dtype=np.uint8)

        # 中心裁剪和缩放
        pil = _center_crop_pil(pil)
        resample = getattr(Image, 'Resampling', Image).BILINEAR
        pil = pil.resize(resize_to, resample)
        pil = pil.convert('RGB')
        
        img = np.asarray(pil, dtype=np.uint8)
        assert img.ndim == 3 and img.shape[2] == 3, f"Shape mismatch: {img.shape}"
        return img
    
    def get_franka_image(self, images):
        """Extracts third-person image from observations and preprocesses it."""
        img = images["scene_image"]
        img = self._to_numpy_uint8_rgb(img)
        return img

    def get_franka_wrist_image(self, images):
        """Extracts wrist camera image from observations and preprocesses it."""
        img = images["wrist_image"]
        img = self._to_numpy_uint8_rgb(img)
        return img
            
    def normalize_gripper_action(self, action: np.ndarray, binarize: bool = True) -> np.ndarray:
        """Convert gripper action from [0,1] to binary [0,1] range"""
        action = np.array(action)
        normalized_action = action.copy()
        
        if binarize:
            # 二值化：>=0.5为关(1)，<0.5为开(0)
            normalized_action[:, -1] = np.where(normalized_action[:, -1] >= 0.9, 1, 0)
        return normalized_action

    def prepare_observation(self, images, state):
        """Prepare observation for policy input."""
        img_resized = self.get_franka_image(images)
        wrist_img_resized = self.get_franka_wrist_image(images)
        gripper_state = np.array(state.end_effector_position)
        gripper_width = np.array(state.gripper_width)
        qpos = np.array(state.joint_positions) # Assuming state has joint_positions
        # print("state:", state)
        return img_resized, wrist_img_resized, gripper_state, gripper_width, qpos

    def process_action(self, action):
        """Process action before sending to environment."""
        action = self.normalize_gripper_action(action, binarize=True)
        action[:, :3] = action[:, :3] / 100
        action[:, 3:-1] =[1e-3, 1e-3, 1e-3]
        return action
    
    def merge(self, actions_chunk):
        merged_pos = np.sum(actions_chunk[:, :3], axis=0)
        merged_gripper = actions_chunk[-1, 6]
        merged_euler = [1e-3, 1e-3, 1e-3]
        merged_action = np.concatenate([merged_pos, merged_euler, [merged_gripper]])
        return merged_action
    
    def merge_action_sequence(self, action_list, merge_count):
        flag = action_list.shape[0] // merge_count
        merged_actions = []
        for i in range(flag):
            start_idx = i * merge_count
            end_idx = start_idx + merge_count
            action_chunk = action_list[start_idx:end_idx]
            merged_action = self.merge(action_chunk)
            print("merged action:", merged_action)
            merged_actions.append(merged_action)
        return np.stack(merged_actions, axis=0)
    
    def get_action(self, control_hz: int = 10):
        period = 1.0 / max(1, control_hz)
        next_t = time.time()

        while not self._stop_event.is_set():
            with self._img_lock:
                scene_img = self.scene_camera_image.copy() 
                wrist_img = self.wrist_camera_image.copy()

            if all(x is None for x in [scene_img, wrist_img]):
                time.sleep(0.001)
                continue
            images = {
                'scene_image': scene_img,
                "wrist_image": wrist_img,
            }
            language_instruction = self.language_instruction
            state = self.robot.read_state()
            img_resized, wrist_img_resized, gripper_state, gripper_width, qpos = self.prepare_observation(images, state)
            
            action_list, trajectories = self.client.inference(img_resized, wrist_img_resized, gripper_state, gripper_width, qpos, language_instruction)
            print("raw_action", action_list[0])
            action_list = self.process_action(action_list)
            print("processed action", action_list[0])
            processed_action_list = self.merge_action_sequence(action_list[:8], 4)
            print("processed action shape:", processed_action_list.shape)  
            
            if action_list is None:
                print("No action")
                continue
            elif isinstance(action_list[0], (list, tuple, np.ndarray)):
                for i in range(2):
                    try:
                        self.move(list(processed_action_list[i]))
                        print("move once")
                    except Exception as e:
                        print(f"[get_action] apply action failed: {e}")
            else:
                try:
                    self.move(list(action_list))
                except Exception as e:
                    print(f"[get_action] apply action failed: {e}")
            
            next_t += period
            dt = next_t - time.time()
            if dt > 0:
                time.sleep(min(dt, 0.005))
            else:
                next_t = time.time()

    def run(self, seconds: float = None):
        t_img = threading.Thread(target=self.update_images, name="t_update_images", daemon=True)
        t_ctl = threading.Thread(target=self.get_action, name="t_get_action", daemon=True)
        self._threads = [t_img, t_ctl]

        for t in self._threads:
            t.start()

        start = time.time()
        try:
            while True:
                if seconds is not None and time.time() - start > seconds:
                    break
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("Stopping due to KeyboardInterrupt...")
        finally:
            self._stop_event.set()
            for t in self._threads:
                t.join(timeout=1.0)
            for cam in [self.scene_camera, self.wrist_camera]:
                try:
                    if cam: cam.disconnect()
                except Exception:
                    pass
            try:
                if self.robot: self.robot.disconnect()
            except Exception:
                pass

if __name__ == "__main__":
    franka_client = shw_franka(FR3Config(), TaskConfig())
    franka_client.run(seconds=600)

# import os
# from pathlib import Path
# import sys
# # 获取当前脚本的绝对路径
# current_file = Path(__file__).resolve()
# # 找到项目根目录 (假设 algo/ 在项目根目录下，所以是 .parent.parent)
# project_root = current_file.parent.parent
# # 将根目录添加到 Python 搜索路径中
# sys.path.append(str(project_root))
# import cv2
# import time
# import threading
# import numpy as np
# from scipy.spatial.transform import Rotation as R
# from dataclasses import dataclass
# from typing import Optional, List
# from roby.hardware.cameras.realsense.camera_realsense import RealSenseCameraConfig, RealSenseCamera
# from roby.hardware.robots.fr3.robot_fr3 import FR3Robot, FR3RobotConfig, FR3RobotAction, FR3ActionMode
# from utils import websocket_client_policy
# from PIL import Image
# from algo.utils.action_mapping import (
#     delta_joint_action_mapping,
#     absolute_position_action_mapping,
#     absolute_joint_action_mapping,
#     delta_absolute_position_action_mapping,
# )
# import datetime

# @dataclass
# class FR3Config:
#     # --- robot setting ---
#     robot_id: str = "fr3"
#     robot_ip: str = "172.16.0.2"
#     load_gripper: bool = True
#     relative_dynamics_factor: float = 0.05
#     buffer_size: int = 10
#     home: bool = True                 

#     # --- camera setting ---
#     scene_camera_id: Optional[str] = "938422072347" # left camera oringinals
#     # scene_camera_id: Optional[str] = "112322077048" # left camera new
#     wrist_camera_id: Optional[str] = "112322074840"
#     fps: int = 15
#     width: int = 640
#     height: int = 480
#     camera_buffer: int = 5

#     # --- control mode ---
#     action_mode: str = "POSITION_DELTA"  # ["POSITION_DELTA", "JOINT_DELTA", "POSITION_ABSOLUTE", "JOINT_ABSOLUTE"]
#     img_update_rate: int = 15            
#     asynchronous: bool = False           # move asynchronous


# @dataclass
# class TaskConfig:
#     algorithm: str = "vla"
#     is_online: bool = True
#     server_host: str = "10.184.17.177"
#     server_port: int = 8063
#     language_instruction: str = "place banana into red bowl"
#     save_video: bool = True
#     video_output_dir: str = "./robot_videos"

# class RobotClient:
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.policy_client = websocket_client_policy.WebsocketClientPolicy(cfg.server_host, cfg.server_port)

#     def inference(self, img_resized, wrist_img_resized, gripper_state, gripper_width, language_instruction):
#         obs = {"left_image": img_resized, "wrist_image": wrist_img_resized, "gripper_state": gripper_state, "gripper_width":gripper_width, "task_description":language_instruction}
#         print(f"task description is {obs['task_description']}")
#         actions = self.policy_client.infer(obs)["actions"]
#         trajectories = self.policy_client.infer(obs)["predicted_trajs"]
#         return actions, trajectories

# class TrajectoryVisualizer:
#     @staticmethod
#     def draw_trajectory_on_image(img: np.ndarray, predicted_traj: np.ndarray, is_rgb: bool = True) -> np.ndarray:
#         """
#         在图像上绘制预测的轨迹点
        
#         Args:
#             img: 原始图像 (H, W, 3) RGB格式
#             predicted_traj: 预测的轨迹点 [w, h, ...]
#             is_rgb: 图像是否为RGB格式
        
#         Returns:
#             绘制了轨迹的图像 (RGB格式)
#         """
#         if predicted_traj is None:
#             return img
        
#         # 复制图像，避免修改原始图像
#         img_with_traj = img.copy()
        
#         # 如果输入是BGR，转换为RGB用于处理
#         if not is_rgb and img_with_traj.ndim == 3 and img_with_traj.shape[2] == 3:
#             img_with_traj = cv2.cvtColor(img_with_traj, cv2.COLOR_BGR2RGB)
        
#         H, W = img_with_traj.shape[:2]
        
#         # 提取轨迹点的坐标 (w, h)
#         if len(predicted_traj) >= 2:
#             w = predicted_traj[0] * W + 20 # 归一化的宽度坐标
#             h = predicted_traj[1] * H + 35  # 归一化的高度坐标
            
#             w = int(np.clip(w, 0, W-1))
#             h = int(np.clip(h, 0, H-1))
            
#             # 绘制红色圆点标记轨迹点
#             radius = 5
#             color = (255, 0, 0)  # 红色 (RGB格式)
#             thickness = -1  # 实心圆
            
#             cv2.circle(img_with_traj, (w, h), radius, color, thickness)
            
#             # 添加文字标签
#             label = f"Traj: ({w}, {h})"
#             cv2.putText(img_with_traj, label, (w + 10, h - 10), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
#         # 如果有方向信息，绘制方向轴
#         if len(predicted_traj) >= 6:  # 假设轨迹包含 [w, h, z, rx, ry, rz, ...]
#             ori_vec = predicted_traj[3:6]
            
#             # 计算旋转矩阵
#             rx, ry, rz = ori_vec[:3]
#             cx, sx = np.cos(rx), np.sin(rx)
#             cy, sy = np.cos(ry), np.sin(ry)
#             cz, sz = np.cos(rz), np.sin(rz)
            
#             Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
#             Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
#             Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            
#             R = Rz @ Ry @ Rx
#             axes = R[:, :3].T  # shape (3,3): 三个轴向量
            
#             scale = 15  # 缩小比例，因为图像较小
#             origin_col, origin_row = w, h  # 注意：OpenCV使用(x, y)坐标，即(col, row)
            
#             # 定义轴颜色: X-红, Y-绿, Z-蓝 (RGB格式)
#             colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # RGB格式
            
#             for i, (vec, color) in enumerate(zip(axes, colors)):
#                 dx = float(vec[0]) * scale
#                 dy = -float(vec[1]) * scale  # 注意：图像坐标系中y轴向下
                
#                 end_col = int(np.clip(origin_col + dx, 0, W - 1))
#                 end_row = int(np.clip(origin_row + dy, 0, H - 1))
                
#                 # 绘制轴
#                 cv2.line(img_with_traj, (origin_col, origin_row), (end_col, end_row), 
#                         color, thickness=1, lineType=cv2.LINE_AA)  # 减小线宽
                
#                 # 在轴末端绘制小圆点
#                 cv2.circle(img_with_traj, (end_col, end_row), 2, color, -1, lineType=cv2.LINE_AA)
                
#                 # 添加轴标签
#                 axis_labels = ['X', 'Y', 'Z']
#                 label_pos = (end_col + 3, end_row + 3)  # 调整标签位置
#                 cv2.putText(img_with_traj, axis_labels[i], label_pos, 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)  # 减小字体
        
#         # 转换回BGR格式用于显示和保存
#         if img_with_traj.ndim == 3 and img_with_traj.shape[2] == 3:
#             img_with_traj = cv2.cvtColor(img_with_traj, cv2.COLOR_RGB2BGR)
        
#         return img_with_traj
    
#     @staticmethod
#     def create_video_writer(output_path: str, fps: int = 10, frame_size: tuple = (256, 256)):
#         """创建视频写入器，保存为AVI格式"""
#         # 使用AVI格式，XVID编码器
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         return cv2.VideoWriter(output_path, fourcc, fps, frame_size)

# class shw_franka:
#     def __init__(self, robot_cfg, task_cfg):
#         self.current_gripper_state = 0  # -1: open, 1: closed (跟踪当前夹爪实际状态)
#         self.target_gripper_state = 0   # 0: open, 1: closed (策略要求的目标状态)
#         self.cfg = robot_cfg
#         self.task_cfg = task_cfg
#         self.action_mode = robot_cfg.action_mode
#         self.delta_position_action_mapping = delta_absolute_position_action_mapping.__get__(self)
#         self.delta_joint_action_mapping = delta_joint_action_mapping.__get__(self)
#         self.absolute_position_action_mapping = absolute_position_action_mapping.__get__(self)
#         self.absolute_joint_action_mapping = absolute_joint_action_mapping.__get__(self)
#         self.action_mapping = {
#             "POSITION_DELTA": self.delta_position_action_mapping,
#             "JOINT_DELTA": self.delta_joint_action_mapping,
#             "POSITION_ABSOLUTE": self.absolute_position_action_mapping,
#             "JOINT_ABSOLUTE": self.absolute_joint_action_mapping,
#         }
#         self.robot = None
#         self.client = RobotClient(task_cfg)
#         self.language_instruction = task_cfg.language_instruction
#         self.scene_camera = None
#         self.wrist_camera = None
#         self.scene_camera_id = robot_cfg.scene_camera_id
#         self.wrist_camera_id = robot_cfg.wrist_camera_id
#         self.scene_camera_image = None
#         self.wrist_camera_image = None
#         self._img_lock = threading.Lock()
#         self._stop_event = threading.Event()
#         self._threads = []
        
#         # 可视化相关
#         self.visualizer = TrajectoryVisualizer()
#         self.replay_images: List[np.ndarray] = []  # 存储每一帧图像用于视频
#         self.video_writer = None
        
#         # 创建视频保存目录
#         if task_cfg.save_video:
#             os.makedirs(task_cfg.video_output_dir, exist_ok=True)
#             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             # 使用AVI格式
#             video_path = os.path.join(task_cfg.video_output_dir, f"robot_execution_{timestamp}.mp4")
#             # 使用预处理图像的大小 (256, 256)
#             self.video_writer = self.visualizer.create_video_writer(video_path, fps=10, frame_size=(256, 256))
#             print(f"Video will be saved to: {video_path}")
        
#         self.devices = self.setup_hardwares()

#     def setup_hardwares(self):
#         robot_conf = FR3RobotConfig(
#             id=self.cfg.robot_id,
#             robot_ip=self.cfg.robot_ip,
#             load_gripper=self.cfg.load_gripper,
#             relative_dynamics_factor=self.cfg.relative_dynamics_factor,
#             buffer_size=self.cfg.buffer_size,
#             initial_joint = None,
#             initial_end_pose=[0.4707543519985666, -0.02776023399946076, 0.29950666236991815, 1.0, 0.0, 0.0, 0.0]
#         )
#         self.robot = FR3Robot(robot_conf)
#         self.robot.connect()
#         self.robot.read_state()
#         self.robot._start_read_thread()
#         if self.cfg.home:
#             self.robot.home()
#             self.robot.gripper.open(0.1)
#             self.current_gripper_state = 0  # 初始状态为开
        
#         def create_camera(name, serial):
#             if serial is None:
#                 return None
#             cam_cfg = RealSenseCameraConfig(
#                 fps=self.cfg.fps,
#                 width=self.cfg.width,
#                 height=self.cfg.height,
#                 buffer_size=self.cfg.camera_buffer,
#                 serial_number_or_name=serial,
#             )
#             cam = RealSenseCamera(cam_cfg)
#             cam.connect()
#             cam._start_read_thread()
#             record_devices[name] = cam
#             return cam
        
#         record_devices = {"fr3": self.robot}
#         self.scene_camera = create_camera("scene_camera", self.cfg.scene_camera_id)
#         self.wrist_camera = create_camera("wrist_camera", self.cfg.wrist_camera_id)
#         return record_devices

#     def move(self, action):
#         action = self.action_mapping[self.action_mode](action)
#         self.robot.send_action(action, asynchronous=self.cfg.asynchronous)

#     def update_images(self):
#         period = 1.0 / max(1, int(self.cfg.img_update_rate))
#         next_t = time.time()
#         while not self._stop_event.is_set():
#             with self._img_lock:
#                 if self.scene_camera:
#                     if not self.scene_camera.frame_buffer.empty():
#                         img = self.scene_camera.frame_buffer.queue[-1].color
#                         self.scene_camera_image = cv2.resize(img, (640, 480))
#                 if self.wrist_camera:
#                     if not self.wrist_camera.frame_buffer.empty():
#                         img = self.wrist_camera.frame_buffer.queue[-1].color
#                         self.wrist_camera_image = cv2.resize(img, (640, 480))
#             next_t += period
#             dt = next_t - time.time()
#             time.sleep(max(dt, 0.001))
    
#     def _to_numpy_uint8_rgb(self, data: np.ndarray) -> np.ndarray:
#         """Convert raw image to HxWx3 uint8 RGB."""
#         resize_to = (256, 256)

#         def _center_crop_pil(pil_img: Image.Image) -> Image.Image:
#             w, h = pil_img.size
#             s = min(w, h)
#             left = (w - s) // 2
#             top = (h - s) // 2
#             return pil_img.crop((left, top, left + s, top + s))

#         # 处理numpy array
#         if data.ndim == 3:
#             data_rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
#             pil = Image.fromarray(data_rgb, 'RGB')
        
#         if pil is None:
#             return np.zeros((resize_to[0], resize_to[1], 3), dtype=np.uint8)

#         # 中心裁剪和缩放
#         pil = _center_crop_pil(pil)
#         resample = getattr(Image, 'Resampling', Image).BILINEAR
#         pil = pil.resize(resize_to, resample)
#         pil = pil.convert('RGB')
        
#         img = np.asarray(pil, dtype=np.uint8)
#         assert img.ndim == 3 and img.shape[2] == 3, f"Shape mismatch: {img.shape}"
#         return img
    
#     def get_franka_image(self, images):
#         """Extracts third-person image from observations and preprocesses it."""
#         img = images["scene_image"]
#         img = self._to_numpy_uint8_rgb(img)
#         return img

#     def get_franka_wrist_image(self, images):
#         """Extracts wrist camera image from observations and preprocesses it."""
#         img = images["wrist_image"]
#         img = self._to_numpy_uint8_rgb(img)
#         return img
            
#     def normalize_gripper_action(self, action: np.ndarray, binarize: bool = True) -> np.ndarray:
#         """Convert gripper action from [0,1] to binary [0,1] range"""
#         action = np.array(action)
#         normalized_action = action.copy()
        
#         if binarize:
#             # 二值化：>=0.5为关(1)，<0.5为开(0)
#             normalized_action[:, -1] = np.where((1 - normalized_action[:, -1]) >= 0.8, 1, 0)
        
#         return normalized_action

#     def prepare_observation(self, images, state):
#         """Prepare observation for policy input."""
#         img_resized = self.get_franka_image(images)
#         wrist_img_resized = self.get_franka_wrist_image(images)
#         gripper_state = np.array(state.end_effector_position)
#         gripper_width = np.array(state.gripper_width)
#         print("state:", state)
#         return img_resized, wrist_img_resized, gripper_state, gripper_width

#     def process_action(self, action):
#         """Process action before sending to environment."""
#         action = self.normalize_gripper_action(action, binarize=True)
#         action[:, :3] = action[:, :3] / 100
#         action[:, 3:-1] =[1e-3, 1e-3, 1e-3]
#         return action

#     def save_frame_with_trajectory(self, img_resized: np.ndarray, predicted_traj: np.ndarray, step: int):
#         """
#         在预处理后的图像上绘制轨迹并保存
        
#         Args:
#             img_resized: 预处理后的图像 (256x256 RGB)
#             predicted_traj: 预测的轨迹点
#             step: 当前步骤
#         """
#         if img_resized is None:
#             return
        
#         # 在预处理图像上绘制轨迹
#         img_with_traj_rgb = self.visualizer.draw_trajectory_on_image(img_resized, predicted_traj, is_rgb=True)
        
#         # 转换为PIL图像添加文本
#         pil_img = Image.fromarray(img_with_traj_rgb)
        
#         # 创建新的图像用于绘制文本
#         from PIL import ImageDraw, ImageFont
#         draw = ImageDraw.Draw(pil_img)
        
#         # 使用默认字体
#         try:
#             font = ImageFont.truetype("Arial.ttf", 12)
#         except:
#             font = ImageFont.load_default()
        
#         # 添加文本信息
#         draw.text((10, 10), f"Step: {step}", fill=(255, 255, 255), font=font)
#         draw.text((10, 30), f"Task: {self.language_instruction[:20]}...", fill=(255, 255, 255), font=font)
        
#         # 转换回numpy数组
#         img_with_info = np.array(pil_img)
        
#         # 转换回BGR格式用于显示和保存
#         img_with_info_bgr = cv2.cvtColor(img_with_info, cv2.COLOR_RGB2BGR)
        
#         # 保存到列表用于后续处理
#         self.replay_images.append(img_with_info_bgr)
        
#         # 写入视频文件
#         if self.video_writer is not None:
#             self.video_writer.write(img_with_info_bgr)
        
#         # 实时显示（可选）
#         cv2.imshow("Robot Execution with Trajectory (Processed Image)", img_with_info_bgr)
#         cv2.waitKey(1)

#     def get_action(self, control_hz: int = 10):
#         period = 1.0 / max(1, control_hz)
#         next_t = time.time()
#         step_counter = 0

#         while not self._stop_event.is_set():
#             with self._img_lock:
#                 scene_img = self.scene_camera_image.copy() 
#                 wrist_img = self.wrist_camera_image.copy()

#             if all(x is None for x in [scene_img, wrist_img]):
#                 time.sleep(0.001)
#                 continue
            
#             images = {
#                 'scene_image': scene_img,
#                 "wrist_image": wrist_img,
#             }
#             language_instruction = self.language_instruction
#             state = self.robot.read_state()
#             img_resized, wrist_img_resized, gripper_state, gripper_width = self.prepare_observation(images, state)
            
#             # 获取动作、轨迹和图像
#             action_list, trajectories = self.client.inference(img_resized, wrist_img_resized, gripper_state, gripper_width, language_instruction)
            
#             print("raw_action", action_list)
#             action_list = self.process_action(action_list)
#             print("processed action", action_list)
            
#             # 获取预测的轨迹（取最后一个轨迹点用于可视化）
#             predicted_traj = None
#             if trajectories is not None and len(trajectories) > 0:
#                 predicted_traj = trajectories[2]  # 取最后一个轨迹点
            
#             # 在预处理后的图像上保存带有轨迹可视化的帧
#             # 注意：这里使用img_resized而不是scene_img
#             self.save_frame_with_trajectory(img_resized, predicted_traj, step_counter)
            
#             if action_list is None:
#                 print("No action")
#                 continue
#             elif isinstance(action_list[0], (list, tuple, np.ndarray)):
#                 for i in range(8):
#                     try:
#                         self.move(list(action_list[i]))
#                         print(f"move step {i+1}")
#                     except Exception as e:
#                         print(f"[get_action] apply action failed: {e}")
#             else:
#                 try:
#                     self.move(list(action_list))
#                 except Exception as e:
#                     print(f"[get_action] apply action failed: {e}")
            
#             step_counter += 1
#             next_t += period
#             dt = next_t - time.time()
#             if dt > 0:
#                 time.sleep(min(dt, 0.005))
#             else:
#                 next_t = time.time()

#     def run(self, seconds: float = None):
#         t_img = threading.Thread(target=self.update_images, name="t_update_images", daemon=True)
#         t_ctl = threading.Thread(target=self.get_action, name="t_get_action", daemon=True)
#         self._threads = [t_img, t_ctl]

#         for t in self._threads:
#             t.start()

#         start = time.time()
#         try:
#             while True:
#                 if seconds is not None and time.time() - start > seconds:
#                     break
#                 time.sleep(0.1)
#         except KeyboardInterrupt:
#             print("Stopping due to KeyboardInterrupt...")
#         finally:
#             self._stop_event.set()
            
#             # 停止所有线程
#             for t in self._threads:
#                 t.join(timeout=1.0)
            
#             # 关闭视频写入器
#             if self.video_writer is not None:
#                 self.video_writer.release()
#                 print(f"Video saved with {len(self.replay_images)} frames")
            
#             # 保存最后一帧为图像（可选）
#             if len(self.replay_images) > 0:
#                 timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#                 last_frame_path = os.path.join(self.task_cfg.video_output_dir, f"last_frame_{timestamp}.jpg")
#                 cv2.imwrite(last_frame_path, self.replay_images[-1])
#                 print(f"Last frame saved to: {last_frame_path}")
            
#             # 关闭摄像头和机器人
#             for cam in [self.scene_camera, self.wrist_camera]:
#                 try:
#                     if cam: cam.disconnect()
#                 except Exception:
#                     pass
#             try:
#                 if self.robot: self.robot.disconnect()
#             except Exception:
#                 pass
            
#             # 关闭OpenCV窗口
#             cv2.destroyAllWindows()

# if __name__ == "__main__":
#     franka_client = shw_franka(FR3Config(), TaskConfig())
#     franka_client.run(seconds=200)