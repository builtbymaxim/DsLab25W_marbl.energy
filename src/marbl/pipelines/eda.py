from typing import List
from marbl.core.config import load_config, ProjectConfig


def run(countries: List[str] | None = None) -> dict:
    cfg: ProjectConfig = load_config()
    sel = countries or cfg.countries_primary
    # Platzhalter: hier Daten laden, Qualität prüfen, Spike-Stats berechnen
    return {"countries": sel, "data_dir": str(cfg.data_dir)}
