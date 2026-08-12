"""
Gameplay Recorder - Main Orchestrator
Integrates: capture → GoPro mount simulation → visual analysis → fractype
continuous iteration → Kaggle learning → asset management in a single loop.
"""

import asyncio
import json
import logging
import time
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import numpy as np
import cv2

from core.recorder import (
    RecordingSession, CaptureSourceConfig, CaptureSourceType,
    WindowCaptureSource, FrameData
)
from mounts.mount_simulator import GoProMountSimulator
from analysis.visual_analysis import (
    AnalysisPipeline, AnalysisFrame, FractypeContinuousValidator
)
from fractype_integration.integration import FractypeIntegration
from kaggle_pipeline.pipeline import KagglePipeline
from assets.asset_manager import AssetManager

logger = logging.getLogger(__name__)


class GameplayRecorderOrchestrator:
    """Main orchestrator for the gameplay recording system"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.config_path = config_path
        self.session_id = str(Path(config_path).stem)
        self.running = False
        self.iteration_count = 0
        
        # Subsystems
        self.fractype = FractypeIntegration(config_path)
        self.kaggle = KagglePipeline(config_path)
        self.assets = AssetManager(config_path)
        self.mount_sim = GoProMountSimulator(config_path)
        self.analysis = AnalysisPipeline(config_path, self.fractype)
        
        # Recording session
        self.session = RecordingSession(config_path)
        self.session.capture_manager.add_source(
            WindowCaptureSource(CaptureSourceConfig(
                name="game_window",
                source_type=CaptureSourceType.WINDOW_CAPTURE,
                priority=1,
                config={"monitor": 1, "fps": 60}
            ))
        )
        
        # Frame history for mount simulation
        self.raw_frame_history: List[np.ndarray] = []
        self.processed_frame_history: List[np.ndarray] = []
        
        # Feedback
        self.feedback_score = 0.0
        self.quality_threshold = self.config.get("fractype_integration", {}).get(
            "iteration", {}).get("feedback_threshold", 0.85)
        
    async def start(self):
        """Start the orchestrator"""
        self.running = True
        logger.info(f"Gameplay Recorder Orchestrator started (session: {self.session_id})")
        
        # Initialize subsystems
        await self._init_subsystems()
        
        # Start capture
        self.session.capture_manager.register_callback(self._on_frame_captured)
        
        # Start main loop
        await self._main_loop()
        
    async def _init_subsystems(self):
        """Initialize all subsystems"""
        logger.info("Initializing subsystems...")
        
        # Fractype: check calibration status
        status = self.fractype.get_status()
        logger.info(f"Fractype: {len(status['active_variants'])} active variants")
        
        # Kaggle: check dataset status
        kaggle_status = self.kaggle.get_dataset_status()
        logger.info(f"Kaggle: {kaggle_status['enabled_datasets']} enabled datasets")
        
        # Assets: check asset status
        asset_status = self.assets.get_status()
        logger.info(f"Assets: {asset_status['total_assets']} registered assets")
        
        # Mounts: check mount status
        mounts = self.mount_sim.list_mounts()
        logger.info(f"GoPro mounts: {len(mounts)} available")
        
    def _on_frame_captured(self, frame: FrameData):
        """Callback for captured frames"""
        if frame.image is None:
            return
            
        # Store raw frame
        self.raw_frame_history.append(frame.image.copy())
        if len(self.raw_frame_history) > 30:
            self.raw_frame_history.pop(0)
            
        # Apply GoPro mount simulation
        player_state = self._estimate_player_state(frame)
        processed = self.mount_sim.process_frame(
            (frame.image * 255).astype(np.uint8), player_state
        )
        self.processed_frame_history.append(processed.copy())
        if len(self.processed_frame_history) > 30:
            self.processed_frame_history.pop(0)
            
        # Create analysis frame
        analysis_frame = AnalysisFrame(
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
            source_id=frame.source_id,
            image=processed,  # Already uint8 from mount sim
            depth=frame.depth,
            metadata=frame.metadata
        )
        
        # Run analysis
        raw_for_validation = self.raw_frame_history[-1] if self.raw_frame_history else None
        results = self.analysis.process_frame(analysis_frame, raw_for_validation)
        
        # Update feedback score
        self._update_feedback(results)
        
    def _estimate_player_state(self, frame: FrameData) -> Dict[str, Any]:
        """Estimate player state from frame metadata or heuristics"""
        # In production, this would read game state
        # For demo, use a simple oscillator
        t = frame.timestamp
        return {
            "position": [0, 1.5, 0],
            "rotation": [0, 45 + 10 * np.sin(t), 0],
            "velocity": [2 * np.cos(t), 0, 1 * np.sin(t)],
            "timestamp": t
        }
        
    def _update_feedback(self, results: Dict[str, Any]):
        """Update quality feedback score from analysis results"""
        # Combine quality metrics
        score = 0.0
        count = 0
        
        if "visual_quality" in results:
            vq = results["visual_quality"]
            # Higher sharpness, color accuracy, dynamic range = better
            score += min(1.0, vq.get("sharpness", 0) / 100.0) * 0.3
            score += min(1.0, vq.get("color_accuracy", 0) / 8.0) * 0.3
            score += min(1.0, vq.get("dynamic_range", 0) / 255.0) * 0.4
            count += 1
            
        if "gameplay_metrics" in results:
            gm = results["gameplay_metrics"]
            # Lower clutter, lower sickness risk = better
            score += (1.0 - gm.get("visual_clutter", 0.5)) * 0.2
            score += (1.0 - min(1.0, gm.get("motion_sickness_risk", 0) / 10.0)) * 0.2
            count += 1
            
        if count > 0:
            self.feedback_score = score / count
            
    async def _main_loop(self):
        """Main orchestration loop"""
        try:
            # Start capture
            await self.session.capture_manager.start_all()
            
            # Background tasks
            iteration_task = asyncio.create_task(self._iteration_loop())
            kaggle_task = asyncio.create_task(self._kaggle_loop())
            asset_task = asyncio.create_task(self._asset_loop())
            
            # Capture loop
            start_time = time.time()
            max_duration = self.config.get("session", {}).get("max_duration_hours", 8) * 3600
            
            while self.running:
                cycle_start = time.time()
                
                # Capture cycle
                await self.session.capture_manager.capture_cycle()
                
                # Frame rate limiting
                elapsed = time.time() - cycle_start
                target_frame_time = 1.0 / self.config.get("recorder", {}).get("target_fps", 60)
                sleep_time = max(0, target_frame_time - elapsed)
                await asyncio.sleep(sleep_time)
                
                if time.time() - start_time > max_duration:
                    logger.info("Max duration reached")
                    break
                    
        finally:
            await self.session.capture_manager.stop_all()
            iteration_task.cancel()
            kaggle_task.cancel()
            asset_task.cancel()
            
            # Flush analysis
            self.analysis.flush_results(
                self.config.get("output", {}).get("directories", {}).get("analysis", "./output/analysis")
            )
            
    async def _iteration_loop(self):
        """Continuous fractype iteration loop"""
        while self.running:
            if self.fractype.should_iterate():
                results = await self.fractype_iterate()
                self.iteration_count += 1
                logger.info(f"Iteration {self.iteration_count}: {results.get('actions', [])}")
                
            await asyncio.sleep(self.fractype.iteration_interval / 10)
            
    async def fractype_iterate(self):
        """Run fractype iteration with feedback"""
        return await self.fractype.iterate(feedback_score=self.feedback_score)
        
    async def _kaggle_loop(self):
        """Kaggle continuous learning loop"""
        while self.running:
            tick = await self.kaggle.continuous_learning_tick()
            if tick.get("datasets_updated") or tick.get("features_extracted"):
                logger.info(f"Kaggle tick: {len(tick.get('features_extracted', []))} features")
            await asyncio.sleep(3600)  # Check hourly
            
    async def _asset_loop(self):
        """Asset management loop"""
        while self.running:
            # Process any unprocessed assets
            for asset in self.assets.assets.values():
                if asset.downloaded and not asset.processed:
                    self.assets.process_asset(asset)
            await asyncio.sleep(300)  # Check every 5 min
            
    def stop(self):
        self.running = False
        
    async def run_demo(self, duration: int = 60):
        """Run a demo session for specified duration"""
        self.running = True
        logger.info(f"Demo run for {duration} seconds")
        
        # Initialize
        await self._init_subsystems()
        
        # Run analysis on synthetic frames for demo
        logger.info("Running analysis demo on synthetic frames...")
        for i in range(10):
            frame = AnalysisFrame(
                timestamp=time.time() + i,
                frame_id=i,
                source_id="demo",
                image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                depth=np.random.rand(480, 640).astype(np.float32)
            )
            self.analysis.process_frame(frame)
            
        # Fractype iteration
        await self.fractype.iterate(feedback_score=0.9)
        
        # Kaggle feature extraction
        for name in self.kaggle.datasets:
            self.kaggle.extract_features(name)
            
        # Asset processing
        self.assets.populate_default_catalog()
        self.assets.export_asset_manifest("./output/assets/manifest.json")
        
        # Flush analysis
        self.analysis.flush_results("./output/analysis")
        
        logger.info("Demo complete")
        

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    orchestrator = GameplayRecorderOrchestrator(str(config_path))
    
    # Run demo
    await orchestrator.run_demo(duration=60)
    
    # Print summary
    summary = orchestrator.analysis.get_summary()
    print("\n=== Analysis Summary ===")
    print(json.dumps(summary, indent=2))
    
    print("\n=== Fractype Status ===")
    print(json.dumps(orchestrator.fractype.get_status(), indent=2, default=str))
    
    print("\n=== Kaggle Status ===")
    print(json.dumps(orchestrator.kaggle.get_dataset_status(), indent=2, default=str))
    
    print("\n=== Asset Status ===")
    print(json.dumps(orchestrator.assets.get_status(), indent=2))
    

if __name__ == "__main__":
    asyncio.run(main())
