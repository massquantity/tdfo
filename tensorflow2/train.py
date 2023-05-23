import tensorflow as tf

from data import read_parquet_data, read_tfrecord_data
from models import TwoTower
from utils import Config, get_data_size, read_configs


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


def main():
    config = read_configs()
    train_data_path = config.data_dir / config.write_format / config.train_data
    eval_data_path = config.data_dir / config.write_format / config.eval_data
    cache_dir = config.data_dir / "huggingface"
    train_batch_size = config.per_device_train_batch_size
    eval_batch_size = config.per_device_eval_batch_size
    num_workers = config.num_workers

    train_data_size = get_data_size(train_data_path)
    eval_data_size = get_data_size(eval_data_path)
    print(f"===== train size: {train_data_size:,}, eval size: {eval_data_size:,} =====\n")

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

    model = build_model(config)
    model.fit(
        train_data.repeat(config.n_epochs),
        epochs=config.n_epochs,
        validation_data=eval_data,
        steps_per_epoch=n_train_steps,
    )


if __name__ == "__main__":
    main()
