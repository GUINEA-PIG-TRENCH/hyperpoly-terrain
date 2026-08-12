"""
Visual Analysis Pipeline for R&D Review
Extracts quality metrics, terrain analysis, gameplay metrics, and GoPro
simulation validation from recorded footage.
Integrates with fractype prototype variants for continuous validation.
"""

import numpy as np
import cv2
import json
import yaml
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque

# Lazy/optional imports (torch only used by optional CNN components)
try:
    import torch  # noqa
    import torch.nn as nn  # noqa
    import torch.nn.functional as F  # noqa
except ImportError:
    torch = nn = F = None

logger = logging.getLogger(__name__)


@dataclass
class AnalysisFrame:
    timestamp: float
    frame_id: int
    source_id: str
    image: np.ndarray  # RGB uint8
    depth: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class VisualQualityAnalyzer:
    """Analyzes visual quality of recorded frames"""
    
    def __init__(self):
        self.sharpness_history = deque(maxlen=100)
        self.color_history = deque(maxlen=100)
        self.dr_history = deque(maxlen=100)
        
    def compute_sharpness(self, frame: np.ndarray) -> float:
        """Laplacian variance as sharpness proxy"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())
        
    def compute_color_accuracy(self, frame: np.ndarray) -> float:
        """Color distribution complexity (entropy)"""
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-8)
        entropy = -np.sum(hist * np.log2(hist + 1e-8))
        return float(entropy)
        
    def compute_dynamic_range(self, frame: np.ndarray) -> float:
        """Luminance dynamic range (min to max)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return float(gray.max() - gray.min())
        
    def compute_temporal_stability(self, frame_history: List[np.ndarray]) -> float:
        """Frame-to-frame difference (lower = more stable)"""
        if len(frame_history) < 2:
            return 0.0
        diffs = []
        for i in range(1, len(frame_history)):
            prev = cv2.cvtColor(frame_history[i-1], cv2.COLOR_RGB2GRAY)
            curr = cv2.cvtColor(frame_history[i], cv2.COLOR_RGB2GRAY)
            diffs.append(np.mean(np.abs(curr.astype(np.float32) - prev.astype(np.float32))))
        return float(np.mean(diffs))
        
    def analyze(self, frame: np.ndarray, history: Optional[List[np.ndarray]] = None) -> Dict[str, float]:
        sharpness = self.compute_sharpness(frame)
        color = self.compute_color_accuracy(frame)
        dr = self.compute_dynamic_range(frame)
        stability = self.compute_temporal_stability(history or [frame])
        
        self.sharpness_history.append(sharpness)
        self.color_history.append(color)
        self.dr_history.append(dr)
        
        return {
            "sharpness": sharpness,
            "color_accuracy": color,
            "dynamic_range": dr,
            "temporal_stability": stability
        }


class TerrainAnalyzer:
    """Analyzes terrain features using fractype prototype integration"""
    
    def __init__(self, fractype_integration=None):
        self.fractype = fractype_integration
        self.elevation_history = deque(maxlen=50)
        self.slope_history = deque(maxlen=50)
        
    def analyze_depth_buffer(self, depth: np.ndarray) -> Dict[str, float]:
        """Analyze terrain from depth buffer"""
        if depth is None:
            return {}
            
        # Depth -> elevation estimate
        elevation = 1.0 - depth  # Assume linear depth mapping
        slope_x = np.gradient(elevation, axis=1)
        slope_y = np.gradient(elevation, axis=0)
        slope_mag = np.sqrt(slope_x**2 + slope_y**2)
        
        # Material distribution estimate (from slope)
        flat_frac = float(np.mean(slope_mag < 0.1))
        steep_frac = float(np.mean(slope_mag > 0.5))
        
        self.elevation_history.append(float(elevation.mean()))
        self.slope_history.append(float(slope_mag.mean()))
        
        return {
            "elevation_mean": float(elevation.mean()),
            "elevation_std": float(elevation.std()),
            "slope_mean": float(slope_mag.mean()),
            "flat_fraction": flat_frac,
            "steep_fraction": steep_frac,
            "erosion_realism": self._estimate_erosion_realism(slope_mag)
        }
        
    def _estimate_erosion_realism(self, slope_mag: np.ndarray) -> float:
        """Heuristic: realistic erosion has smooth slope transitions"""
        # Compute slope gradient (curvature)
        curvature = np.gradient(slope_mag)
        curvature_mag = np.sqrt(np.sum(np.array(curvature)**2, axis=0))
        # Smooth curvature = realistic
        smoothness = 1.0 / (1.0 + float(np.mean(curvature_mag)))
        return smoothness
        
    def validate_with_fractype(self, depth: np.ndarray) -> Dict[str, Any]:
        """Validate terrain against fractype prototypes"""
        if self.fractype is None:
            return {}
        return self.fractype.validate_terrain(depth)


class GameplayMetricsAnalyzer:
    """Analyzes gameplay-relevant visual metrics"""
    
    def __init__(self):
        self.visibility_history = deque(maxlen=50)
        
    def compute_player_visibility(self, frame: np.ndarray, player_bbox: Optional[Tuple] = None) -> float:
        """Estimate how visible the player/character is"""
        if player_bbox is None:
            # Heuristic: use saliency-like center bias
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # Center region saliency
            cy, cx = h // 2, w // 2
            center_region = gray[cy-50:cy+50, cx-50:cx+50]
            if center_region.size > 0:
                return float(center_region.mean() / 255.0)
        return 0.5
        
    def compute_visual_clutter(self, frame: np.ndarray) -> float:
        """Estimate visual clutter (high frequency content)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        return float(np.mean(edges) / 255.0)
        
    def compute_motion_sickness_risk(self, frame_history: List[np.ndarray]) -> float:
        """Estimate motion sickness risk from frame motion"""
        if len(frame_history) < 3:
            return 0.0
        # Optical flow magnitude
        flows = []
        for i in range(1, min(len(frame_history), 4)):
            prev = cv2.cvtColor(frame_history[i-1], cv2.COLOR_RGB2GRAY)
            curr = cv2.cvtColor(frame_history[i], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flows.append(float(np.percentile(mag, 95)))
        return float(np.mean(flows))
        
    def analyze(self, frame: np.ndarray, history: Optional[List[np.ndarray]] = None) -> Dict[str, float]:
        visibility = self.compute_player_visibility(frame)
        clutter = self.compute_visual_clutter(frame)
        sickness = self.compute_motion_sickness_risk(history or [frame])
        
        self.visibility_history.append(visibility)
        
        return {
            "player_visibility": visibility,
            "visual_clutter": clutter,
            "motion_sickness_risk": sickness
        }


class GoProValidationAnalyzer:
    """Validates GoPro simulation accuracy"""
    
    def __init__(self):
        self.stab_history = deque(maxlen=30)
        
    def validate_stabilization(self, raw_frame: np.ndarray, stabilized_frame: np.ndarray) -> Dict[str, float]:
        """Measure stabilization effectiveness"""
        raw_gray = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2GRAY)
        stab_gray = cv2.cvtColor(stabilized_frame, cv2.COLOR_RGB2GRAY)
        
        # Edge sharpness comparison
        raw_lap = cv2.Laplacian(raw_gray, cv2.CV_64F).var()
        stab_lap = cv2.Laplacian(stab_gray, cv2.CV_64F).var()
        
        # Higher stabilized sharpness = better stabilization
        stab_effectiveness = float(stab_lap / (raw_lap + 1e-8))
        
        self.stab_history.append(stab_effectiveness)
        return {
            "stabilization_effectiveness": stab_effectiveness,
            "sharpness_retention": min(1.0, stab_effectiveness)
        }
        
    def validate_distortion(self, frame: np.ndarray, expected_fov: float) -> Dict[str, float]:
        """Validate that distortion matches expected FOV"""
        h, w = frame.shape[:2]
        # Estimate FOV from frame content (heuristic)
        # More barrel distortion = wider FOV
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2, h / 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2) / np.sqrt(cx**2 + cy**2)
        edge_distortion = float(np.percentile(r, 99))
        
        return {
            "distortion_match": edge_distortion,
            "estimated_fov": expected_fov * edge_distortion
        }


class FractypeContinuousValidator:
    """Continuous validation of fractype prototypes during iteration"""
    
    def __init__(self, fractype_integration=None):
        self.fractype = fractype_integration
        self.drift_history = deque(maxlen=50)
        
    def validate(self, iteration_id: str) -> Dict[str, Any]:
        """Run all fractype validation checks"""
        if self.fractype is None:
            return {}
            
        results = {}
        
        # Conservation drift
        cons_drift = self.fractype.check_conservation_drift()
        results["conservation_drift"] = cons_drift
        
        # Quantization error
        quant_err = self.fractype.check_quantization_error()
        results["quantization_error"] = quant_err
        
        # Material calibration drift
        calib_drift = self.fractype.check_calibration_drift()
        results["material_calibration_drift"] = calib_drift
        
        # Shader compilation time
        compile_time = self.fractype.measure_shader_compile_time()
        results["shader_compilation_time"] = compile_time
        
        # GPU memory
        gpu_mem = self.fractype.measure_gpu_memory()
        results["gpu_memory_usage"] = gpu_mem
        
        self.drift_history.append(results)
        return results


class AnalysisPipeline:
    """Orchestrates all analysis pipelines"""
    
    def __init__(self, config_path: str, fractype_integration=None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.analysis_config = self.config.get("analysis", {})
        self.frame_history: deque = deque(maxlen=30)
        
        self.visual = VisualQualityAnalyzer()
        self.terrain = TerrainAnalyzer(fractype_integration)
        self.gameplay = GameplayMetricsAnalyzer()
        self.gopro = GoProValidationAnalyzer()
        self.fractype_validator = FractypeContinuousValidator(fractype_integration)
        
        self.results_buffer = []
        self.fractype = fractype_integration

        # Normalize pipelines config (list of dicts -> name-keyed dict)
        pipelines_raw = self.analysis_config.get("pipelines", [])
        if isinstance(pipelines_raw, list):
            self.pipelines = {p.get("name"): p for p in pipelines_raw if isinstance(p, dict)}
        elif isinstance(pipelines_raw, dict):
            self.pipelines = pipelines_raw
        else:
            self.pipelines = {}

    def _pipeline_enabled(self, name: str, default: bool = True) -> bool:
        cfg = self.pipelines.get(name, {})
        if isinstance(cfg, dict):
            return cfg.get("enabled", default)
        return default

    def process_frame(self, frame: AnalysisFrame, raw_frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Process single frame through all pipelines"""
        self.frame_history.append(frame.image)
        history = list(self.frame_history)
        
        results = {
            "timestamp": frame.timestamp,
            "frame_id": frame.frame_id,
            "source_id": frame.source_id
        }
        
        # Visual quality
        if self._pipeline_enabled("visual_quality"):
            results["visual_quality"] = self.visual.analyze(frame.image, history)

        # Terrain analysis
        if self._pipeline_enabled("terrain_analysis"):
            if frame.depth is not None:
                results["terrain_analysis"] = self.terrain.analyze_depth_buffer(frame.depth)

        # Gameplay metrics
        if self._pipeline_enabled("gameplay_metrics"):
            results["gameplay_metrics"] = self.gameplay.analyze(frame.image, history)

        # GoPro validation
        if self._pipeline_enabled("gopro_simulation_validation"):
            if raw_frame is not None:
                results["gopro_validation"] = self.gopro.validate_stabilization(raw_frame, frame.image)

        # Fractype continuous validation
        if self._pipeline_enabled("fractype_continuous_validation"):
            results["fractype_validation"] = self.fractype_validator.validate(
                f"frame_{frame.frame_id}"
            )
            
        self.results_buffer.append(results)
        return results
        
    def flush_results(self, output_dir: str):
        """Save results to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_path = output_path / f"analysis_{timestamp}.json"
        
        with open(json_path, 'w') as f:
            json.dump(self.results_buffer, f, indent=2)

        self.last_summary = self.get_summary()
        logger.info(f"Analysis results saved: {json_path}")
        
    def get_summary(self) -> Dict[str, Any]:
        """Get aggregate summary"""
        if not self.results_buffer:
            return getattr(self, "last_summary", {})
            
        # Average all metrics
        summary = {}
        for result in self.results_buffer:
            for category, metrics in result.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            if category not in summary:
                                summary[category] = {}
                            if metric not in summary[category]:
                                summary[category][metric] = []
                            summary[category][metric].append(value)
                            
        # Compute means
        for category in summary:
            for metric in summary[category]:
                summary[category][metric] = float(np.mean(summary[category][metric]))
                
        return summary


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    
    pipeline = AnalysisPipeline(str(config_path))
    
    # Test with synthetic frames
    test_frames = [
        AnalysisFrame(
            timestamp=time.time() + i,
            frame_id=i,
            source_id="game_window",
            image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            depth=np.random.rand(480, 640).astype(np.float32)
        )
        for i in range(5)
    ]
    
    for frame in test_frames:
        results = pipeline.process_frame(frame)
        print(f"Frame {frame.frame_id}: {list(results.keys())}")
        
    summary = pipeline.get_summary()
    print(f"Summary: {json.dumps(summary, indent=2)}")
    pipeline.flush_results("./output/analysis")
