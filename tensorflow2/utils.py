import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import polars as pl
import tomli

COLUMNS = [
    "user_id",
    "item_id",
    "language",
    "is_ebook",
    "format",
    "publisher",
    "pub_decade",
    "avg_rating",
    "num_pages",
]


@dataclass
class Config:
    data_dir: Path
    train_data: str
    eval_data: str
    num_workers: int
    size_map: Dict[str, int]
    n_epochs: int
    learning_rate: float
    weight_decay: float
    embed_dim: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    jit_xla: bool
    seed: int
    use_tpu: bool


def read_configs() -> Config:
    toml_str = (Path(__file__).parent / "config.toml").read_text()
    config = tomli.loads(toml_str)
    config["data_dir"] = Path(config["data_dir"]).absolute()
    size_map = Path.read_text(config["data_dir"] / "size_map.json")
    config["size_map"] = json.loads(size_map)
    config["jit_xla"] = config["jit_xla"] or None
    return Config(**config)


def get_data_size(data_path: str) -> int:
    data = pl.scan_parquet(data_path)
    return data.select(pl.count()).collect(streaming=True).item()
