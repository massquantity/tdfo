import math
from typing import Callable

import tensorflow as tf
from tqdm import tqdm

from data import read_parquet_data, read_tfrecord_data
from models import TwoTower
from utils import Config, get_data_size, read_configs

# N_VIRTUAL_DEVICES = 2
# physical_devices = tf.config.list_physical_devices("CPU")
# tf.config.set_logical_device_configuration(
#   physical_devices[0],
#    [tf.config.LogicalDeviceConfiguration() for _ in range(N_VIRTUAL_DEVICES)]
# )
# print("Simulated devices:",  tf.config.list_logical_devices())
# strategy = tf.distribute.MirroredStrategy()


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

    def __init__(
        self, strategy: tf.distribute.Strategy, data: tf.data.Dataset, n_steps: int
    ):
        self.data = strategy.experimental_distribute_dataset(data)
        self.n_steps = n_steps

    def __len__(self):
        return self.n_steps

    def __iter__(self):
        yield from self.data


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
    train_data_path = config.data_dir / config.write_format / config.train_data
    eval_data_path = config.data_dir / config.write_format / config.eval_data
    cache_dir = config.data_dir / "huggingface"

    strategy = setup_strategy(config)
    n_replicas = strategy.num_replicas_in_sync
    train_batch_size = config.per_device_train_batch_size * n_replicas
    eval_batch_size = config.per_device_eval_batch_size * n_replicas
    num_workers = config.num_workers
    # `jit_compile` has a few issues:
    # 1. XLA does not work well for multi-gpus:
    # https://github.com/tensorflow/tensorflow/issues/45940
    # 2. Since XLA compilation will be performed on TPU implicitly, set `jit_compile=True` may lead to strange behavior:
    # https://huggingface.co/docs/transformers/main/perf_train_tpu_tf#i-keep-hearing-about-this-xla-thing-whats-xla-and-how-does-it-relate-to-tpus
    # https://github.com/keras-team/keras-nlp/issues/443
    if config.use_tpu or n_replicas > 1:
        config.jit_xla = None

    train_data_size = get_data_size(train_data_path)
    eval_data_size = get_data_size(eval_data_path)
    print(f"===== train size: {train_data_size:,}, eval size: {eval_data_size:,} =====")
    print(f"===== num devices: {n_replicas} =====\n")

    tf.random.set_seed(config.seed)
    if config.write_format == "tfrecord":
        train_data, eval_data = read_tfrecord_data(
            train_data_path, eval_data_path, train_batch_size, eval_batch_size
        )
    else:
        train_data, eval_data = read_parquet_data(
            train_data_path,
            eval_data_path,
            cache_dir,
            train_batch_size,
            eval_batch_size,
            num_workers,
        )
    n_train_steps = train_data_size // train_batch_size
    n_eval_steps = math.ceil(eval_data_size / eval_batch_size)
    train_dist_data = DistDataset(strategy, train_data, n_train_steps)
    eval_dist_data = DistDataset(strategy, eval_data, n_eval_steps)

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
