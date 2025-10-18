from pathlib import Path
import os
import yaml
from pydantic import BaseModel
from typing import List, Dict


class ProjectConfig(BaseModel):
    data_dir: Path
    countries_primary: List[str]
    countries_backup: List[str]
    window_days: int
    spike_thresholds: Dict[str, Dict[str, float]]
    quality_rules: Dict[str, float | int]


def load_config(path: str = "config/project.yml") -> ProjectConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    data_dir = os.path.expandvars(os.path.expanduser(raw.get("data_dir", "./data")))
    raw["data_dir"] = Path(data_dir).resolve()
    return ProjectConfig(**raw)
