from dataclasses import dataclass
from pathlib import Path

from src.core.utils import load_yaml


@dataclass
class RuntimeSettings:
    fps_input: int
    fps_output: int
    window_sec: int
    stride_sec: int
    resize_width: int
    resize_height: int
    draw_overlay: bool
    save_figure: bool


def load_runtime_settings(config_path: str = "configs/app.yaml") -> tuple[RuntimeSettings, dict]:
    cfg = load_yaml(config_path)
    rt = cfg["runtime"]
    return RuntimeSettings(**rt), cfg


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
