import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal

import polars as pl
import tomli


@dataclass
class Config:
    data_dir: Path
    train_data: str
    eval_data: str
    write_format: Literal["tfrecord", "parquet"]
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
    assert config["write_format"] in ("tfrecord", "parquet")
    return Config(**config)


def get_data_size(data_path: Path) -> int:
    if str(data_path).endswith("tfrecord"):
        name = "train" if "train" in str(data_path) else "eval"
        path = data_path.parent / f"{name}_data_size.json"
        return json.loads(path.read_text())[f"{name}_data_size"]
    else:
        data = pl.scan_parquet(data_path)
        return data.select(pl.count()).collect(streaming=True).item()
