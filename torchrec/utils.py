from dataclasses import dataclass
from pathlib import Path

import polars as pl
import tomli


@dataclass
class Config:
    data_dir: Path
    train_data: str
    eval_data: str
    # streaming: bool
    n_epochs: int
    learning_rate: float
    weight_decay: float
    embed_dim: int
    n_heads: int
    n_layers: int
    max_len: int
    sliding_step: int
    mask_prob: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    model_parallel: bool
    # mixed_precision: bool
    seed: int


def read_configs() -> Config:
    toml_str = (Path(__file__).parent / "config.toml").read_text()
    config = tomli.loads(toml_str)
    config["data_dir"] = Path(config["data_dir"]).absolute()
    assert config["max_len"] >= config["sliding_step"]
    return Config(**config)


def get_data_size(data_path: str) -> int:
    data = pl.scan_parquet(data_path)
    return data.select(pl.count()).collect(streaming=True).item()
