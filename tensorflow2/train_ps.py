import json
import multiprocessing
import os
from pathlib import Path

import tensorflow as tf

from models import TwoTower
from utils import Config, get_data_size, read_configs


def create_in_process_cluster(cluster_spec, num_workers, num_ps):
    """Creates and starts local servers and returns the cluster_resolver."""
    # noinspection PyUnresolvedReferences
    worker_config = tf.compat.v1.ConfigProto()
    if multiprocessing.cpu_count() < num_workers + 1:
        worker_config.inter_op_parallelism_threads = num_workers + 1

    for i in range(num_workers):
        tf.distribute.Server(
            cluster_spec,
            job_name="worker",
            task_index=i,
            config=worker_config,
            protocol="grpc",
        )

    for i in range(num_ps):
        tf.distribute.Server(
            cluster_spec,
            job_name="ps",
            task_index=i,
            protocol="grpc",
        )

    cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
        cluster_spec, rpc_layer="grpc"
    )
    os.environ["GRPC_FAIL_FAST"] = "use_caller"
    return cluster_resolver


def setup_strategy():
    cluster_dict = json.loads((Path(".").absolute() / "cluster.json").read_text())
    cluster_spec = tf.train.ClusterSpec(cluster_dict["cluster"])
    num_workers = len(cluster_dict["cluster"]["worker"])
    num_ps = len(cluster_dict["cluster"]["ps"])
    cluster_resolver = create_in_process_cluster(cluster_spec, num_workers, num_ps)

    variable_partitioner = tf.distribute.experimental.partitioners.MinSizePartitioner(
        min_shard_bytes=(256 << 10),
        max_shards=num_ps,
    )
    strategy = tf.distribute.ParameterServerStrategy(
        cluster_resolver, variable_partitioner=variable_partitioner
    )
    return strategy


def build_data(data_path: Path, global_batch_size: int, shuffle: bool = False):
    feature_description = {
        "user_id": tf.io.FixedLenFeature([], tf.int64),
        "item_id": tf.io.FixedLenFeature([], tf.int64),
        "language": tf.io.FixedLenFeature([], tf.int64),
        "is_ebook": tf.io.FixedLenFeature([], tf.int64),
        "format": tf.io.FixedLenFeature([], tf.int64),
        "publisher": tf.io.FixedLenFeature([], tf.int64),
        "pub_decade": tf.io.FixedLenFeature([], tf.int64),
        "avg_rating": tf.io.FixedLenFeature([], tf.float32),
        "num_pages": tf.io.FixedLenFeature([], tf.float32),
        "label": tf.io.FixedLenFeature([], tf.float32),
    }

    def _parse(serialized_examples):
        example = tf.io.parse_example(serialized_examples, feature_description)
        label = example.pop("label")
        return example, label

    def dataset_fn(input_context):
        batch_size = input_context.get_per_replica_batch_size(global_batch_size)
        dataset = tf.data.TFRecordDataset(
            filenames=tf.data.TFRecordDataset.list_files(str(data_path), seed=42),
            compression_type="GZIP",
            num_parallel_reads=tf.data.AUTOTUNE,
        )
        if shuffle:
            dataset = dataset.shuffle(buffer_size=2_000_000)
        dataset = dataset.repeat()
        dataset = dataset.shard(
            input_context.num_input_pipelines, input_context.input_pipeline_id
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    input_options = tf.distribute.InputOptions(
        experimental_fetch_to_device=True, experimental_per_replica_buffer_size=2
    )
    data = tf.keras.utils.experimental.DatasetCreator(dataset_fn, input_options)
    return data


def build_model(config: Config):
    model = TwoTower(config.size_map, config.embed_dim)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.AdamW(config.learning_rate, config.weight_decay)
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[tf.keras.metrics.AUC(curve="ROC", from_logits=True)],
        # jit_compile=config.jit_xla,
        steps_per_execution=config.steps_per_execution,
    )
    return model


def main():
    config = read_configs()
    train_data_path = config.data_dir / config.write_format / config.train_data
    eval_data_path = config.data_dir / config.write_format / config.eval_data

    strategy = setup_strategy()
    n_replicas = strategy.num_replicas_in_sync
    train_batch_size = config.per_device_train_batch_size * n_replicas
    eval_batch_size = config.per_device_eval_batch_size * n_replicas

    train_data_size = get_data_size(train_data_path)
    eval_data_size = get_data_size(eval_data_path)
    print(f"===== train size: {train_data_size:,}, eval size: {eval_data_size:,} =====")
    print(f"===== num devices: {n_replicas} =====\n")

    tf.random.set_seed(config.seed)
    train_data = build_data(train_data_path, train_batch_size, shuffle=True)
    eval_data = build_data(eval_data_path, eval_batch_size)
    n_train_steps = train_data_size // train_batch_size

    with strategy.scope():
        model = build_model(config)

    model.fit(
        train_data,
        epochs=config.n_epochs,
        validation_data=eval_data,
        steps_per_epoch=n_train_steps,
    )


if __name__ == "__main__":
    main()
