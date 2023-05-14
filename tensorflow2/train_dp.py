from pathlib import Path
from typing import Callable

import tensorflow as tf
from datasets import load_dataset
from tqdm import tqdm

from models import TwoTower
from utils import COLUMNS, Config, get_data_size, read_configs


def setup_strategy(config: Config) -> tf.distribute.Strategy:
    if config.use_tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) == 0:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        elif len(gpus) == 1:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            strategy = tf.distribute.MirroredStrategy()

    return strategy


class DistDataset:
    def __init__(self, strategy: tf.distribute.Strategy, data: tf.data.Dataset):
        self.data = strategy.experimental_distribute_dataset(data)
        self.data_size = len(data)

    def __len__(self):
        return self.data_size

    def __iter__(self):
        yield from self.data


def build_data(
    train_data_path: Path,
    eval_data_path: Path,
    cache_dir: Path,
    train_batch_size: int,
    eval_batch_size: int,
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
    )
    eval_dataset = dataset["eval"].to_tf_dataset(
        eval_batch_size,
        shuffle=False,
        drop_remainder=False,
        prefetch=True,
        columns=COLUMNS,
        label_cols="label",
    )
    return train_dataset, eval_dataset


def get_train_func(
    config: Config,
    strategy: tf.distribute.Strategy,
    model: tf.keras.Model,
    loss_fn: Callable,
    optimizer: tf.keras.optimizers.Optimizer,
    auc_metric: tf.keras.metrics.Metric,
    batch_size: int,
):
    def train_step(inputs: tf.Tensor, labels: tf.Tensor):
        with tf.GradientTape() as tape:
            logits = model(inputs)
            # add new dim for loss reduction
            logits, labels = logits[:, tf.newaxis], labels[:, tf.newaxis]
            loss = tf.nn.compute_average_loss(
                per_example_loss=loss_fn(labels, logits), global_batch_size=batch_size
            )
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        auc_metric.update_state(labels, logits)
        return loss

    @tf.function(jit_compile=config.jit_xla)
    def train_dist_step(inputs: tf.Tensor, labels: tf.Tensor):
        per_replica_losses = strategy.run(train_step, args=(inputs, labels))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    return train_dist_step


def get_eval_func(
    config: Config,
    strategy: tf.distribute.Strategy,
    model: tf.keras.Model,
    loss_fn: Callable,
    loss_metric: tf.keras.metrics.Metric,
    auc_metric: tf.keras.metrics.Metric,
):
    def eval_step(inputs: tf.Tensor, labels: tf.Tensor):
        logits = model(inputs)
        logits, labels = logits[:, tf.newaxis], labels[:, tf.newaxis]
        loss = loss_fn(labels, logits)
        loss_metric.update_state(loss)
        auc_metric.update_state(labels, logits)
        return loss

    @tf.function(jit_compile=config.jit_xla)
    def eval_dist_step(inputs: tf.Tensor, labels: tf.Tensor):
        strategy.run(eval_step, args=(inputs, labels))

    return eval_dist_step


def main():
    config = read_configs()
    train_data_path = config.data_dir / config.train_data
    eval_data_path = config.data_dir / config.eval_data
    cache_dir = config.data_dir / "huggingface"

    strategy = setup_strategy(config)
    n_replicas = strategy.num_replicas_in_sync
    train_batch_size = config.per_device_train_batch_size * n_replicas
    eval_batch_size = config.per_device_eval_batch_size * n_replicas

    train_data_size = get_data_size(train_data_path)
    eval_data_size = get_data_size(eval_data_path)
    print(f"===== train size: {train_data_size}, eval size: {eval_data_size} =====")
    print(f"===== num devices: {n_replicas} =====\n")

    tf.random.set_seed(config.seed)
    train_data, eval_data = build_data(
        train_data_path,
        eval_data_path,
        cache_dir,
        train_batch_size,
        eval_batch_size,
    )
    train_dist_data = DistDataset(strategy, train_data)
    eval_dist_data = DistDataset(strategy, eval_data)

    with strategy.scope():
        model = TwoTower(config.size_map, config.embed_dim)
        # `AdamW` became experimental since tf2.10
        optimizer = tf.keras.optimizers.experimental.AdamW(
            config.learning_rate, config.weight_decay
        )
        loss_fn = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        eval_loss_metric = tf.keras.metrics.Mean()
        train_auc_metric = tf.keras.metrics.AUC(curve="ROC", from_logits=True)
        eval_auc_metric = tf.keras.metrics.AUC(curve="ROC", from_logits=True)

        train_dist_step = get_train_func(
            config, strategy, model, loss_fn, optimizer, train_auc_metric, train_batch_size
        )
        eval_dist_step = get_eval_func(
            config, strategy, model, loss_fn, eval_loss_metric, eval_auc_metric
        )

    for epoch in range(1, config.n_epochs + 1):
        train_loss, n_batches = 0.0, 0
        for inputs, labels in tqdm(train_dist_data, desc="Training..."):
            train_loss += train_dist_step(inputs, labels)
            n_batches += 1
        train_loss /= n_batches
        train_auc = train_auc_metric.result()
        print(f"\nEpoch {epoch} train loss: {train_loss:.4f}, train auc: {train_auc:.4f}\n")

        for inputs, labels in tqdm(eval_dist_data, desc="Evaluating..."):
            eval_dist_step(inputs, labels)
        eval_loss = eval_loss_metric.result()
        eval_auc = eval_auc_metric.result()
        print(f"\nEpoch {epoch} eval loss: {eval_loss:.4f}, eval auc: {eval_auc:.4f}\n")

        eval_loss_metric.reset_states()
        train_auc_metric.reset_states()
        eval_auc_metric.reset_states()


if __name__ == "__main__":
    main()
