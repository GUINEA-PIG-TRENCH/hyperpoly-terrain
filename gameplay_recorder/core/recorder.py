"""
Core Gameplay Recorder
Multi-source capture with fractype integration, GoPro mount simulation,
Kaggle pipeline integration, and open-source asset management.
"""

import asyncio
import json
import time
import uuid
import logging
import yaml
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
import cv2
import threading
from queue import Queue, Empty
from collections import deque

# Lazy/optional imports
try:
    import mss
except ImportError:
    mss = None
try:
    import torch
except ImportError:
    torch = None
try:
    import psutil
except ImportError:
    psutil = None
try:
    import pygetwindow as gw
except ImportError:
    gw = None
try:
    import GPUtil
except ImportError:
    GPUtil = None

logger = logging.getLogger(__name__)


class CaptureSourceType(Enum):
    WINDOW_CAPTURE = "window_capture"
    SIMULATED_MOUNT = "simulated_mount"
    GPU_BUFFER = "gpu_buffer_capture"
    EXTERNAL_DEVICE = "external_device"


@dataclass
class FrameData:
    """Container for captured frame data"""
    timestamp: float
    frame_id: int
    source_id: str
    source_type: CaptureSourceType
    image: np.ndarray  # HWC, RGB, float32 [0,1]
    metadata: Dict[str, Any] = field(default_factory=dict)
    depth: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    motion_vectors: Optional[np.ndarray] = None


@dataclass
class CaptureSourceConfig:
    name: str
    source_type: CaptureSourceType
    priority: int
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class CaptureSource(ABC):
    """Abstract base class for capture sources"""
    
    def __init__(self, config: CaptureSourceConfig):
        self.config = config
        self.enabled = config.enabled
        self.frame_queue: Queue = Queue(maxsize=60)
        self.running = False
        self.frame_count = 0
        
    @abstractmethod
    async def start(self):
        pass
    
    @abstractmethod
    async def stop(self):
        pass
    
    @abstractmethod
    async def capture_frame(self) -> Optional[FrameData]:
        pass
    
    def get_latest_frame(self) -> Optional[FrameData]:
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None


class WindowCaptureSource(CaptureSource):
    """Captures game window using MSS"""
    
    def __init__(self, config: CaptureSourceConfig):
        super().__init__(config)
        if mss is None:
            raise RuntimeError("mss not installed; cannot use window capture")
        self.sct = mss.mss()
        self.monitor = config.config.get("monitor", 1)
        self.target_fps = config.config.get("fps", 60)
        self.region = config.config.get("region")  # left, top, width, height
        
    async def start(self):
        self.running = True
        logger.info(f"Window capture started: {self.config.name}")
        
    async def stop(self):
        self.running = False
        self.sct.close()
        logger.info(f"Window capture stopped: {self.config.name}")
        
    async def capture_frame(self) -> Optional[FrameData]:
        if not self.running:
            return None
            
        monitor = self.sct.monitors[self.monitor] if self.region is None else self.region
        screenshot = self.sct.grab(monitor)
        img = np.array(screenshot)[:, :, :3]  # BGRA -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = img.astype(np.float32) / 255.0
        
        frame = FrameData(
            timestamp=time.time(),
            frame_id=self.frame_count,
            source_id=self.config.name,
            source_type=CaptureSourceType.WINDOW_CAPTURE,
            image=img,
            metadata={
                "monitor": self.monitor,
                "region": monitor,
                "fps": self.target_fps
            }
        )
        self.frame_count += 1
        return frame


class GPUBufferCaptureSource(CaptureSource):
    """Captures GPU buffers (depth, normals, motion vectors) via compute shaders"""
    
    def __init__(self, config: CaptureSourceConfig):
        super().__init__(config)
        self.device = config.config.get("device")
        self.buffer_names = config.config.get("buffers", ["depth", "normals"])
        self.readback_buffers = {}
        
    async def start(self):
        self.running = True
        logger.info(f"GPU buffer capture started: {self.config.name}")
        
    async def stop(self):
        self.running = False
        logger.info(f"GPU buffer capture stopped: {self.config.name}")
        
    async def capture_frame(self) -> Optional[FrameData]:
        if not self.running or not self.device:
            return None
            
        # In production, this would use GPU readback via compute shaders
        # For now, return placeholder
        return None


class CaptureManager:
    """Manages multiple capture sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sources: Dict[str, CaptureSource] = {}
        self.running = False
        self.session_id = str(uuid.uuid4())[:8]
        self.frame_callbacks: List[Callable[[FrameData], None]] = []
        
    def add_source(self, source: CaptureSource):
        self.sources[source.config.name] = source
        
    def register_callback(self, callback: Callable[[FrameData], None]):
        self.frame_callbacks.append(callback)
        
    async def start_all(self):
        self.running = True
        for source in self.sources.values():
            if source.enabled:
                await source.start()
        logger.info(f"Capture manager started with {len(self.sources)} sources")
        
    async def stop_all(self):
        self.running = False
        for source in self.sources.values():
            await source.stop()
        logger.info("Capture manager stopped")
        
    async def capture_cycle(self):
        """Single capture cycle across all sources"""
        for source in sorted(self.sources.values(), key=lambda s: s.config.priority):
            if not source.enabled:
                continue
            try:
                frame = await source.capture_frame()
                if frame:
                    for callback in self.frame_callbacks:
                        callback(frame)
            except Exception as e:
                logger.error(f"Capture error from {source.config.name}: {e}")
                
    async def run(self, duration: Optional[float] = None):
        """Run capture loop"""
        await self.start_all()
        start_time = time.time()
        
        try:
            while self.running:
                cycle_start = time.time()
                await self.capture_cycle()
                
                # Frame rate limiting
                elapsed = time.time() - cycle_start
                target_frame_time = 1.0 / self.config.get("target_fps", 60)
                sleep_time = max(0, target_frame_time - elapsed)
                await asyncio.sleep(sleep_time)
                
                if duration and (time.time() - start_time) > duration:
                    break
                    
        finally:
            await self.stop_all()


class RecordingSession:
    """Manages a complete recording session with all subsystems"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.session_id = str(uuid.uuid4())[:8]
        self.start_time = time.time()
        self.recorder_config = self.config.get("recorder", {})
        self.capture_manager = CaptureManager(self.recorder_config)
        self.output_dir = Path(self.config.get("output", {}).get("directories", {}).get("recordings", "./output/recordings"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.frame_writers: Dict[str, cv2.VideoWriter] = {}
        self.metadata_buffer = []
        self.frame_count = 0
        
    def setup_writers(self):
        """Initialize video writers for each source"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.recorder_config.get("target_fps", 60)
        resolution = tuple(self.recorder_config.get("capture_resolution", [3840, 2160]))
        
        for source_name in self.capture_manager.sources:
            output_path = self.output_dir / f"{self.session_id}_{source_name}.mp4"
            writer = cv2.VideoWriter(
                str(output_path), fourcc, fps, resolution, isColor=True
            )
            self.frame_writers[source_name] = writer
            
    def frame_callback(self, frame: FrameData):
        """Process captured frame"""
        # Write to video
        if frame.source_id in self.frame_writers:
            writer = self.frame_writers[frame.source_id]
            img = (frame.image * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            writer.write(img)
            
        # Buffer metadata
        self.metadata_buffer.append({
            "timestamp": frame.timestamp,
            "frame_id": frame.frame_id,
            "source_id": frame.source_id,
            "source_type": frame.source_type.value,
            "metadata": frame.metadata
        })
        self.frame_count += 1
        
        # Periodic metadata flush
        if self.frame_count % 300 == 0:
            self.flush_metadata()
            
    def flush_metadata(self):
        if self.metadata_buffer:
            metadata_path = self.output_dir / f"{self.session_id}_metadata.jsonl"
            with open(metadata_path, 'a') as f:
                for entry in self.metadata_buffer:
                    f.write(json.dumps(entry) + '\n')
            self.metadata_buffer.clear()
            
    async def run(self, duration: Optional[float] = None):
        """Run recording session"""
        self.setup_writers()
        self.capture_manager.register_callback(self.frame_callback)
        
        max_duration = duration or self.config.get("session", {}).get("max_duration_hours", 8) * 3600
        
        try:
            await self.capture_manager.run(max_duration)
        finally:
            self.flush_metadata()
            for writer in self.frame_writers.values():
                writer.release()
            logger.info(f"Session {self.session_id} complete: {self.frame_count} frames")


async def main():
    logging.basicConfig(level=logging.INFO)
    
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    session = RecordingSession(str(config_path))
    
    # Add window capture source
    window_config = CaptureSourceConfig(
        name="game_window",
        source_type=CaptureSourceType.WINDOW_CAPTURE,
        priority=1,
        config={"monitor": 1, "fps": 60}
    )
    window_source = WindowCaptureSource(window_config)
    session.capture_manager.add_source(window_source)
    
    # Run for 60 seconds as demo
    await session.run(duration=60)


if __name__ == "__main__":
    asyncio.run(main())