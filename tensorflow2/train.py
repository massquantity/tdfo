from pathlib import Path

import tensorflow as tf
from datasets import load_dataset

from models import TwoTower
from utils import COLUMNS, Config, get_data_size, read_configs


def build_model(config: Config):
    model = TwoTower(config.size_map, config.embed_dim)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.AdamW(config.learning_rate, config.weight_decay)
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[tf.keras.metrics.AUC(curve="ROC", from_logits=True)],
        jit_compile=config.jit_xla,
    )
    return model


def build_data(
    train_data_path: Path,
    eval_data_path: Path,
    cache_dir: Path,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
):
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": str(train_data_path),
            "eval": str(eval_data_path),
        },
        cache_dir=str(cache_dir),
    )
    train_dataset = dataset["train"].to_tf_dataset(
        train_batch_size,
        shuffle=True,
        drop_remainder=True,
        prefetch=True,
        columns=COLUMNS,
        label_cols="label",
        num_workers=num_workers,
    )
    eval_dataset = dataset["eval"].to_tf_dataset(
        eval_batch_size,
        shuffle=False,
        drop_remainder=False,
        prefetch=True,
        columns=COLUMNS,
        label_cols="label",
        num_workers=num_workers,
    )
    return train_dataset, eval_dataset


def main():
    config = read_configs()
    train_data_path = config.data_dir / config.train_data
    eval_data_path = config.data_dir / config.eval_data
    cache_dir = config.data_dir / "huggingface"
    train_batch_size = config.per_device_train_batch_size
    eval_batch_size = config.per_device_eval_batch_size
    num_workers = config.num_workers

    train_data_size = get_data_size(train_data_path)
    eval_data_size = get_data_size(eval_data_path)
    print(f"===== train size: {train_data_size}, eval size: {eval_data_size} =====\n")

    tf.random.set_seed(config.seed)
    train_data, eval_data = build_data(
        train_data_path,
        eval_data_path,
        cache_dir,
        train_batch_size,
        eval_batch_size,
        num_workers,
    )

    model = build_model(config)
    model.fit(train_data, epochs=config.n_epochs, validation_data=eval_data)


if __name__ == "__main__":
    main()
