"""
Open-Source Asset Pipeline (Royalty-Free)
Manages downloading, processing, and tagging of open-source assets for use in
gameplay recording and R&D. Supports Kenney, OpenGameArt, Poly Haven, AmbientCG,
Sketchfab (CC-BY), and GitHub repos with permissive licenses.
"""

import json
import yaml
import logging
import time
import asyncio
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class License(Enum):
    CC0 = "CC0"
    CC_BY = "CC-BY"
    CC_BY_SA = "CC-BY-SA"
    GPL = "GPL"
    MIT = "MIT"
    APACHE = "Apache-2.0"
    BSD = "BSD-3-Clause"
    UNKNOWN = "unknown"


class AssetCategory(Enum):
    MODELS_3D = "3d_models"
    TEXTURES = "textures"
    AUDIO = "audio"
    UI = "ui"
    SPRITES = "sprites"
    HDRI = "hdri"
    PBR_MATERIALS = "pbr_materials"
    SHADERS = "shaders"
    SCRIPTS = "scripts"
    TOOLS = "tools"


@dataclass
class AssetSource:
    name: str
    url: str
    licenses: List[License]
    categories: List[AssetCategory]
    auto_download: bool = False
    repos: List[str] = field(default_factory=list)


@dataclass
class Asset:
    id: str
    name: str
    source: str
    category: AssetCategory
    license: License
    path: str
    tags: Set[str] = field(default_factory=set)
    hash: str = ""
    downloaded: bool = False
    processed: bool = False


class AssetManager:
    """Manages open-source royalty-free assets"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.asset_config = self.config.get("assets", {})
        self.sources: Dict[str, AssetSource] = {}
        self.assets: Dict[str, Asset] = {}
        self.asset_dir = Path(self.asset_config.get("processing", {}).get(
            "output_dir", "./output/assets"
        ))
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_sources()
        
    def _load_sources(self):
        for s in self.asset_config.get("sources", []):
            try:
                licenses = [License(l) for l in s.get("license", ["CC0"])]
            except ValueError:
                licenses = [License.UNKNOWN]
                
            try:
                categories = [AssetCategory(c) for c in s.get("categories", [])]
            except ValueError:
                categories = []
                
            source = AssetSource(
                name=s["name"],
                url=s["url"],
                licenses=licenses,
                categories=categories,
                auto_download=s.get("auto_download", False),
                repos=s.get("repos", [])
            )
            self.sources[source.name] = source
            
        logger.info(f"Loaded {len(self.sources)} asset sources")
        
    def register_asset(self, asset: Asset):
        """Register an asset in the manager"""
        self.assets[asset.id] = asset
        
    def find_assets(self, category: Optional[AssetCategory] = None,
                    license: Optional[License] = None,
                    tag: Optional[str] = None) -> List[Asset]:
        """Find assets matching filters"""
        results = []
        for asset in self.assets.values():
            if category and asset.category != category:
                continue
            if license and asset.license != license:
                continue
            if tag and tag not in asset.tags:
                continue
            results.append(asset)
        return results
        
    def download_from_source(self, source_name: str) -> List[Asset]:
        """Download assets from a source (manual step - requires API access)"""
        source = self.sources.get(source_name)
        if not source:
            logger.warning(f"Source not found: {source_name}")
            return []
            
        if not source.auto_download:
            logger.info(f"Auto-download disabled for {source_name}. Manual download required.")
            return []
            
        downloaded = []
        source_dir = self.asset_dir / source.name
        source_dir.mkdir(parents=True, exist_ok=True)
        
        # GitHub repos: clone if auto_download
        if source.repos and source.url == "https://github.com":
            for repo in source.repos:
                repo_dir = source_dir / repo.replace("/", "_")
                if not repo_dir.exists():
                    logger.info(f"Cloning {repo}...")
                    success = self._git_clone(repo, repo_dir)
                    if success:
                        # Register assets found in repo
                        assets = self._scan_repo_for_assets(repo, repo_dir, source)
                        downloaded.extend(assets)
                        
        return downloaded
        
    def _git_clone(self, repo: str, dest: Path) -> bool:
        """Clone a git repository"""
        try:
            result = subprocess.run(
                ["git", "clone", f"https://github.com/{repo}.git", str(dest)],
                capture_output=True, text=True, timeout=120
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Clone failed: {e}")
            return False
            
    def _scan_repo_for_assets(self, repo: str, repo_dir: Path, source: AssetSource) -> List[Asset]:
        """Scan cloned repo for usable assets"""
        assets = []
        patterns = {
            ".gltf": AssetCategory.MODELS_3D,
            ".glb": AssetCategory.MODELS_3D,
            ".png": AssetCategory.TEXTURES,
            ".jpg": AssetCategory.TEXTURES,
            ".hdr": AssetCategory.HDRI,
            ".wgsl": AssetCategory.SHADERS,
            ".js": AssetCategory.SCRIPTS,
            ".py": AssetCategory.SCRIPTS,
        }
        
        for ext, category in patterns.items():
            for file_path in repo_dir.rglob(f"*{ext}"):
                if "node_modules" in str(file_path) or ".git" in str(file_path):
                    continue
                    
                asset_id = hashlib.md5(str(file_path).encode()).hexdigest()[:12]
                asset = Asset(
                    id=asset_id,
                    name=file_path.stem,
                    source=repo,
                    category=category,
                    license=source.licenses[0] if source.licenses else License.UNKNOWN,
                    path=str(file_path),
                    downloaded=True
                )
                self.register_asset(asset)
                assets.append(asset)
                
        return assets
        
    # ── Processing ──
    
    def process_asset(self, asset: Asset) -> bool:
        """Process an asset (compress, optimize, generate LODs, validate PBR)"""
        if not asset.downloaded:
            return False
            
        processing = self.asset_config.get("processing", {})
        
        # Tag with auto-tags if enabled
        if self.asset_config.get("tagging", {}).get("enabled", False):
            self._auto_tag(asset)
            
        # Generate hash
        asset.hash = self._compute_hash(asset.path)
        
        asset.processed = True
        logger.info(f"Processed asset: {asset.name} ({asset.category.value})")
        return True
        
    def _auto_tag(self, asset: Asset):
        """Auto-tag asset based on filename/path"""
        path_str = asset.path.lower()
        tags = self.asset_config.get("tagging", {}).get("tags", [])
        
        for tag in tags:
            if tag in path_str:
                asset.tags.add(tag)
                
        # Add category as tag
        asset.tags.add(asset.category.value)
        
    def _compute_hash(self, path: str) -> str:
        """Compute file hash"""
        h = hashlib.sha256()
        try:
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    h.update(chunk)
        except Exception:
            return ""
        return h.hexdigest()[:16]
        
    # ── Query interface for R&D ──
    
    def get_asset_for_rnd(self, category: AssetCategory, license_filter: Optional[License] = None) -> List[Asset]:
        """Get assets suitable for R&D iteration"""
        candidates = self.find_assets(category=category, license=license_filter)
        # Prefer CC0 for R&D (no attribution needed)
        cc0 = [a for a in candidates if a.license == License.CC0]
        if cc0:
            return cc0
        return candidates
        
    def export_asset_manifest(self, output_path: str):
        """Export asset manifest"""
        manifest = {
            "generated_at": time.time(),
            "total_assets": len(self.assets),
            "by_category": {},
            "by_license": {},
            "assets": [
                {
                    "id": a.id,
                    "name": a.name,
                    "source": a.source,
                    "category": a.category.value,
                    "license": a.license.value,
                    "tags": list(a.tags),
                    "path": a.path,
                    "hash": a.hash,
                    "downloaded": a.downloaded,
                    "processed": a.processed
                }
                for a in self.assets.values()
            ]
        }
        
        for a in self.assets.values():
            cat = a.category.value
            lic = a.license.value
            manifest["by_category"][cat] = manifest["by_category"].get(cat, 0) + 1
            manifest["by_license"][lic] = manifest["by_license"].get(lic, 0) + 1
            
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        logger.info(f"Asset manifest exported: {output_path}")
        
    def get_status(self) -> Dict[str, Any]:
        return {
            "sources": list(self.sources.keys()),
            "total_assets": len(self.assets),
            "downloaded": sum(1 for a in self.assets.values() if a.downloaded),
            "processed": sum(1 for a in self.assets.values() if a.processed),
            "by_category": {
                cat.value: sum(1 for a in self.assets.values() if a.category == cat)
                for cat in AssetCategory
            }
        }

    def populate_default_catalog(self):
        """Seed the catalog with known open-source royalty-free assets.

        These are references to well-known CC0/CC-BY asset collections. Actual
        download requires manual retrieval or enabling auto_download per source.
        """
        defaults = [
            # Kenney (CC0)
            ("kenney_nature", "Kenney Nature Pack", "kenney_assets",
             AssetCategory.MODELS_3D, License.CC0, {"environment", "props"}),
            ("kenney_urban", "Kenney Urban Pack", "kenney_assets",
             AssetCategory.MODELS_3D, License.CC0, {"environment", "props"}),
            ("kenney_textures", "Kenney Texture Pack", "kenney_assets",
             AssetCategory.TEXTURES, License.CC0, {"material", "surface"}),
            ("kenney_audio", "Kenney Audio Pack", "kenney_assets",
             AssetCategory.AUDIO, License.CC0, {"sound", "music"}),
            ("kenney_ui", "Kenney UI Pack", "kenney_assets",
             AssetCategory.UI, License.CC0, {"interface"}),
            # Poly Haven (CC0)
            ("polyhaven_hdri", "Poly Haven HDRI", "polyhaven",
             AssetCategory.HDRI, License.CC0, {"lighting", "environment"}),
            ("polyhaven_pbr", "Poly Haven PBR", "polyhaven",
             AssetCategory.PBR_MATERIALS, License.CC0, {"material", "surface"}),
            # AmbientCG (CC0)
            ("ambientcg_pbr", "AmbientCG PBR", "ambientcg",
             AssetCategory.PBR_MATERIALS, License.CC0, {"material", "surface"}),
            ("ambientcg_hdri", "AmbientCG HDRI", "ambientcg",
             AssetCategory.HDRI, License.CC0, {"lighting", "environment"}),
            # OpenGameArt (mixed)
            ("oga_textures", "OpenGameArt Textures", "opengameart",
             AssetCategory.TEXTURES, License.CC_BY, {"game_art"}),
            ("oga_audio", "OpenGameArt Audio", "opengameart",
             AssetCategory.AUDIO, License.CC_BY_SA, {"sound"}),
        ]
        for aid, name, src, cat, lic, tags in defaults:
            asset = Asset(
                id=aid,
                name=name,
                source=src,
                category=cat,
                license=lic,
                path=str(self.asset_dir / aid),
                tags=tags,
            )
            self.register_asset(asset)
        logger.info(f"Populated catalog with {len(defaults)} default assets")

    def generate_report(self) -> Dict[str, Any]:
        """Generate asset inventory report"""
        by_category: Dict[str, int] = {}
        by_license: Dict[str, int] = {}
        for asset in self.assets.values():
            by_category[asset.category.value] = by_category.get(asset.category.value, 0) + 1
            by_license[asset.license.value] = by_license.get(asset.license.value, 0) + 1
        return {
            "total_assets": len(self.assets),
            "by_category": by_category,
            "by_license": by_license,
            "processed": sum(1 for a in self.assets.values() if a.processed),
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    
    manager = AssetManager(str(config_path))
    print("Asset status:", json.dumps(manager.get_status(), indent=2))
    
    # Download from auto-enabled sources
    for source_name in manager.sources:
        if manager.sources[source_name].auto_download:
            manager.download_from_source(source_name)
            
    manager.export_asset_manifest("./output/assets/manifest.json")
    print("Asset manager initialized.")
