import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectConfig(BaseModel):
    name: str = "CadastraAI Urban Parcel Extraction"
    default_crs_epsg: int = 32643
    max_gsd_cm: float = 10.0
    min_gsd_cm: float = 1.0


class IngestionConfig(BaseModel):
    allowed_raster_types: List[str] = ["ORI", "DSM", "DTM", "NDSM"]
    cog_block_size: int = 512
    cog_compression: str = "DEFLATE"
    cog_overview_resampling: str = "bilinear"
    bounds_overlap_min_ratio: float = 0.90
    control_points_max_rmse_cm: float = 50.0


class PreprocessingConfig(BaseModel):
    ndsm_noise_min_height_m: float = 0.5
    ndsm_median_filter_size: int = 3
    percentile_min: float = 2.0
    percentile_max: float = 98.0
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    hillshade_azimuth_deg: float = 315.0
    hillshade_altitude_deg: float = 45.0
    stack_channels: str = "rgb+ndsm"


class TilingConfig(BaseModel):
    tile_size_px: int = 1024
    overlap_ratio: float = 0.25
    training_chip_size_px: int = 512
    training_nodata_max_ratio: float = 0.70
    blending_window: str = "hann"


class TopologyConfig(BaseModel):
    node_snap_tolerance_m: float = 0.30
    min_face_area_sqm: float = 5.0
    sliver_compactness_threshold: float = 0.05
    voronoi_building_seed_weight: bool = True


class ValidationConfig(BaseModel):
    overlap_area_tolerance_sqm: float = 0.05
    gap_area_threshold_sqm: float = 10.0
    spike_angle_min_deg: float = 15.0
    max_relative_area_diff: float = 0.25


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "CadastraAI"
    app_version: str = "0.1.0"
    debug: bool = False

    # Base Paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent.parent
    data_dir: Path = base_dir / "data"
    weights_dir: Path = base_dir / "weights"
    config_path: Path = base_dir / "configs" / "default.yaml"

    # Database
    database_url: str = Field(
        default="postgresql+psycopg://cadastra:cadastra_pw@localhost:5432/cadastra_db",
        alias="DATABASE_URL",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        alias="REDIS_URL",
    )

    # Sub-configurations
    project: ProjectConfig = ProjectConfig()
    ingestion: IngestionConfig = IngestionConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    tiling: TilingConfig = TilingConfig()
    topology: TopologyConfig = TopologyConfig()
    validation: ValidationConfig = ValidationConfig()

    @classmethod
    def load_from_yaml(cls, yaml_path: Optional[Path] = None) -> "Settings":
        """Loads settings with YAML file values taking precedence over default schema values."""
        target_path = yaml_path or (Path(__file__).resolve().parent.parent.parent.parent / "configs" / "default.yaml")
        yaml_data: Dict[str, Any] = {}
        if target_path.exists():
            with open(target_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}

        settings = cls()
        if "project" in yaml_data:
            settings.project = ProjectConfig(**yaml_data["project"])
        if "ingestion" in yaml_data:
            settings.ingestion = IngestionConfig(**yaml_data["ingestion"])
        if "preprocessing" in yaml_data:
            settings.preprocessing = PreprocessingConfig(**yaml_data["preprocessing"])
        if "tiling" in yaml_data:
            settings.tiling = TilingConfig(**yaml_data["tiling"])
        if "topology" in yaml_data:
            settings.topology = TopologyConfig(**yaml_data["topology"])
        if "validation" in yaml_data:
            settings.validation = ValidationConfig(**yaml_data["validation"])
        return settings


settings = Settings.load_from_yaml()
