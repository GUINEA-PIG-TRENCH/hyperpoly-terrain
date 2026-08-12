"""
GoPro Mount Simulation System
Simulates first-person camera mounts (helmet, chest, handlebar, drone, suction)
with realistic FOV, lens distortion, stabilization, and motion characteristics.
Uses fractype prototype variants for terrain-aware camera paths.
"""

import numpy as np
import cv2
import yaml
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)


class StabilizationMode(Enum):
    HYPERSMOOTH_5 = "hypersmooth_5"
    GIMBAL = "gimbal_simulation"
    NONE = "none"


class LensModel(Enum):
    GOPRO_WIDE = "gopro_wide"  # ~122.6° diagonal FOV
    LINEAR = "linear"          # ~94.4° diagonal FOV
    SUPERVIEW = "superview"    # 155° ultra-wide


@dataclass
class MountConfig:
    id: str
    name: str
    position_offset: Tuple[float, float, float]
    rotation_offset: Tuple[float, float, float]
    fov_degrees: float
    aspect_ratio: float
    stabilization: StabilizationMode
    lens_distortion: LensModel
    motion_blur: bool = True
    exposure_mode: str = "auto"
    vibration_profile: Optional[str] = None
    enabled: bool = True


@dataclass
class CameraState:
    position: np.ndarray  # 3D world position
    rotation: np.ndarray  # 3x3 rotation matrix
    velocity: np.ndarray  # 3D velocity
    angular_velocity: np.ndarray  # 3D angular velocity
    timestamp: float


class GoProMountSimulator:
    """Simulates a GoPro camera mounted on a player/vehicle in the game world"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.mounts: Dict[str, MountConfig] = {}
        self.active_mount: Optional[str] = None
        self.camera_history: deque = deque(maxlen=300)  # ~5 sec at 60fps
        self.frame_history: deque = deque(maxlen=10)
        self.stabilization_buffers: Dict[str, deque] = {}
        
        self._load_mounts()
        
    def _load_mounts(self):
        mount_config = self.config.get("gopro_mounts", {})
        mounts_list = mount_config.get("mounts", [])
        
        for m in mounts_list:
            mount = MountConfig(
                id=m["id"],
                name=m["name"],
                position_offset=tuple(m.get("position_offset", [0, 0, 0])),
                rotation_offset=tuple(m.get("rotation_offset", [0, 0, 0])),
                fov_degrees=m.get("fov_degrees", 122.6),
                aspect_ratio=m.get("aspect_ratio", 16/9),
                stabilization=StabilizationMode(m.get("stabilization", "hypersmooth_5")),
                lens_distortion=LensModel(m.get("lens_distortion", "gopro_wide")),
                motion_blur=m.get("motion_blur", True),
                exposure_mode=m.get("exposure_mode", "auto"),
                vibration_profile=m.get("vibration_profile"),
                enabled=m.get("enabled", True)
            )
            self.mounts[mount.id] = mount
            self.stabilization_buffers[mount.id] = deque(maxlen=30)
            
        if self.mounts:
            self.active_mount = next(iter(self.mounts.keys()))
            
        logger.info(f"Loaded {len(self.mounts)} GoPro mounts")
        
    def set_active_mount(self, mount_id: str):
        if mount_id in self.mounts:
            self.active_mount = mount_id
            logger.info(f"Active mount: {self.mounts[mount_id].name}")
        else:
            logger.warning(f"Mount not found: {mount_id}")
            
    def get_active_mount(self) -> Optional[MountConfig]:
        return self.mounts.get(self.active_mount) if self.active_mount else None
        
    def compute_camera_transform(self, player_state: Dict[str, Any]) -> CameraState:
        """
        Compute camera transform from player state.
        player_state: {position: [x,y,z], rotation: [pitch,yaw,roll], velocity: [vx,vy,vz]}
        """
        mount = self.get_active_mount()
        if not mount:
            return None
            
        # Player transform
        p_pos = np.array(player_state.get("position", [0, 0, 0]), dtype=np.float32)
        p_rot = np.array(player_state.get("rotation", [0, 0, 0]), dtype=np.float32)
        p_vel = np.array(player_state.get("velocity", [0, 0, 0]), dtype=np.float32)
        
        # Mount offset in player local space (apply yaw rotation)
        yaw = np.radians(p_rot[1])
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        offset = np.array(mount.position_offset, dtype=np.float32)
        rotated_offset = np.array([
            offset[0] * cos_y - offset[2] * sin_y,
            offset[1],
            offset[0] * sin_y + offset[2] * cos_y
        ])
        
        cam_pos = p_pos + rotated_offset
        
        # Camera rotation = player rotation + mount rotation offset
        cam_rot_euler = p_rot + np.array(mount.rotation_offset, dtype=np.float32)
        cam_rot = self._euler_to_matrix(cam_rot_euler)
        
        # Angular velocity from rotation delta
        if self.camera_history:
            prev = self.camera_history[-1]
            dt = max(1e-3, player_state.get("timestamp", time.time()) - prev.timestamp)
            ang_vel = (cam_rot - prev.rotation) / dt
        else:
            ang_vel = np.zeros((3, 3), dtype=np.float32)
            
        state = CameraState(
            position=cam_pos,
            rotation=cam_rot,
            velocity=p_vel,
            angular_velocity=ang_vel,
            timestamp=player_state.get("timestamp", time.time())
        )
        self.camera_history.append(state)
        return state
        
    def _euler_to_matrix(self, euler: np.ndarray) -> np.ndarray:
        """Convert [pitch, yaw, roll] in degrees to 3x3 rotation matrix"""
        px, py, pz = np.radians(euler)
        cx, sx = np.cos(px), np.sin(px)
        cy, sy = np.cos(py), np.sin(py)
        cz, sz = np.cos(pz), np.sin(pz)
        
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        
        return Rz @ Ry @ Rx
        
    def apply_stabilization(self, frame: np.ndarray, mount_id: str) -> np.ndarray:
        """
        Apply HyperSmooth-style stabilization using gyro data from camera history.
        """
        mount = self.mounts.get(mount_id)
        if not mount or mount.stabilization == StabilizationMode.NONE:
            return frame
            
        buffer = self.stabilization_buffers[mount_id]
        buffer.append(frame.copy())
        
        if len(buffer) < 10:
            return frame
            
        # Estimate motion from camera history
        if len(self.camera_history) >= 2:
            curr = self.camera_history[-1]
            prev = self.camera_history[-2]
            # Translation delta
            dt = max(1e-3, curr.timestamp - prev.timestamp)
            trans_vel = (curr.position - prev.position) / dt
            ang_vel = (curr.rotation - prev.rotation) / dt
            
            # Compute stabilization crop/transform
            # Simple approach: counter-translate by velocity * smoothing_factor
            h, w = frame.shape[:2]
            # Estimate shake from angular velocity norm
            shake = np.linalg.norm(ang_vel)
            smooth_factor = min(1.0, shake * 0.1)  # More shake -> more correction
            
            # Apply subtle translation correction (simulated)
            dx = int(-trans_vel[0] * smooth_factor * 5)
            dy = int(-trans_vel[1] * smooth_factor * 5)
            
            # Clamp to small range
            dx = np.clip(dx, -20, 20)
            dy = np.clip(dy, -20, 20)
            
            # Apply translation via ROI crop with padding
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            stabilized = cv2.warpAffine(frame, M, (w, h))
            return stabilized
            
        return frame
        
    def apply_lens_distortion(self, frame: np.ndarray, mount_id: str) -> np.ndarray:
        """Apply GoPro-style lens distortion (barrel for wide, pincushion for linear)"""
        mount = self.mounts.get(mount_id)
        if not mount:
            return frame
            
        h, w = frame.shape[:2]
        cx, cy = w / 2, h / 2
        
        # Create distortion map
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        x_norm = (x - cx) / cx
        y_norm = (y - cy) / cy
        r2 = x_norm**2 + y_norm**2
        
        if mount.lens_distortion == LensModel.GOPRO_WIDE:
            # Barrel distortion (negative k1)
            k1 = -0.35
            k2 = 0.15
        elif mount.lens_distortion == LensModel.LINEAR:
            # Slight pincushion (positive k1)
            k1 = 0.05
            k2 = 0.0
        else:  # SUPERVIEW
            k1 = -0.5
            k2 = 0.25
            
        # Radial distortion
        factor = 1 + k1 * r2 + k2 * r2**2
        x_dist = cx + x_norm * factor * cx
        y_dist = cy + y_norm * factor * cy
        
        # Remap
        map_x = x_dist.astype(np.float32)
        map_y = y_dist.astype(np.float32)
        distorted = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        return distorted
        
    def apply_motion_blur(self, frame: np.ndarray, mount_id: str) -> np.ndarray:
        """Apply motion blur based on camera velocity"""
        mount = self.mounts.get(mount_id)
        if not mount or not mount.motion_blur:
            return frame
            
        if len(self.camera_history) >= 2:
            curr = self.camera_history[-1]
            prev = self.camera_history[-2]
            dt = max(1e-3, curr.timestamp - prev.timestamp)
            speed = np.linalg.norm(curr.velocity) / dt
            
            # Blur kernel size scales with speed
            kernel_size = int(min(31, speed * 0.5))
            if kernel_size < 3:
                return frame
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            # Directional blur (approximate)
            blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
            return blurred
            
        return frame
        
    def process_frame(self, frame: np.ndarray, player_state: Dict[str, Any]) -> np.ndarray:
        """Full GoPro mount processing pipeline"""
        mount_id = self.active_mount
        if not mount_id:
            return frame
            
        # 1. Compute camera transform
        self.compute_camera_transform(player_state)
        
        # 2. Apply stabilization
        frame = self.apply_stabilization(frame, mount_id)
        
        # 3. Apply lens distortion
        frame = self.apply_lens_distortion(frame, mount_id)
        
        # 4. Apply motion blur
        frame = self.apply_motion_blur(frame, mount_id)
        
        # 5. Apply chromatic aberration (subtle)
        frame = self.apply_chromatic_aberration(frame)
        
        # 6. Apply vignetting
        frame = self.apply_vignetting(frame)
        
        self.frame_history.append(frame)
        return frame
        
    def apply_chromatic_aberration(self, frame: np.ndarray) -> np.ndarray:
        """Apply subtle RGB channel shift at edges"""
        h, w = frame.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2, h / 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2) / np.sqrt(cx**2 + cy**2)
        
        shift = r * 2.0  # pixels at edge
        map_r = x + shift
        map_b = x - shift
        map_g = x
        
        # Separate channels
        b, g, r = cv2.split(frame.astype(np.float32))
        r_shifted = cv2.remap(r, map_r.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR)
        b_shifted = cv2.remap(b, map_b.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR)
        
        result = cv2.merge([b_shifted, g, r_shifted])
        return np.clip(result, 0, 255).astype(np.uint8)
        
    def apply_vignetting(self, frame: np.ndarray) -> np.ndarray:
        """Apply subtle vignette"""
        h, w = frame.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2, h / 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2) / np.sqrt(cx**2 + cy**2)
        
        vignette = 1.0 - 0.3 * r**2
        vignette = np.clip(vignette, 0, 1)[:, :, None]
        
        return np.clip(frame.astype(np.float32) * vignette, 0, 255).astype(np.uint8)
        
    def list_mounts(self) -> List[str]:
        return list(self.mounts.keys())
        
    def get_mount_info(self, mount_id: str) -> Dict[str, Any]:
        mount = self.mounts.get(mount_id)
        if not mount:
            return {}
        return {
            "id": mount.id,
            "name": mount.name,
            "fov_degrees": mount.fov_degrees,
            "stabilization": mount.stabilization.value,
            "lens": mount.lens_distortion.value,
            "enabled": mount.enabled
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    sim = GoProMountSimulator(str(config_path))
    
    print("Available mounts:")
    for mount_id in sim.list_mounts():
        print(f"  {mount_id}: {sim.get_mount_info(mount_id)}")
    
    # Test with synthetic frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    player_state = {
        "position": [0, 1.5, 0],
        "rotation": [0, 45, 0],
        "velocity": [2, 0, 1],
        "timestamp": time.time()
    }
    
    processed = sim.process_frame(test_frame, player_state)
    print(f"Processed frame shape: {processed.shape}")
