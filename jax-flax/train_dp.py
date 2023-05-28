import functools
import math
from typing import Dict, Optional

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import polars as pl
from datasets import Dataset, IterableDataset, load_dataset
from flax.training import train_state
from flax.training.common_utils import get_metrics, shard
from flax.training.dynamic_scale import DynamicScale
from tqdm import tqdm

from models import init_model
from utils import read_configs

# import os
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


# noinspection PyAbstractClass
class TrainState(train_state.TrainState):
    dynamic_scale: DynamicScale


def create_train_state(
    rng: jax.random.PRNGKey,
    size_map: dict,
    learning_rate: float,
    weight_decay: float,
    embed_dim: int,
    mixed_precision: bool
):
    model, params = init_model(rng, size_map, embed_dim, mixed_precision)
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    dynamic_scale = DynamicScale() if mixed_precision else None
    return TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer, dynamic_scale=dynamic_scale
    )


def train_step(state: TrainState, batch: Dict[str, jnp.ndarray]):
    def loss_fn(params):
        labels = batch.pop("label")
        logits = state.apply_fn({"params": params}, batch)
        loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))
        return loss

    dynamic_scale = state.dynamic_scale
    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn, axis_name="batch")
        # dynamic loss takes care of averaging gradients across replicas
        dynamic_scale, is_finite_grads, loss, grads = grad_fn(state.params)
    else:
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        grads = jax.lax.pmean(grads, axis_name="batch")

    new_state = state.apply_gradients(grads=grads)
    metrics = jax.lax.pmean({"loss": loss}, axis_name="batch")
    if dynamic_scale:
        # noinspection PyUnboundLocalVariable
        new_state = new_state.replace(
            opt_state=jax.tree_util.tree_map(
                functools.partial(jnp.where, is_finite_grads),
                new_state.opt_state,
                state.opt_state,
            ),
            params=jax.tree_util.tree_map(
                functools.partial(jnp.where, is_finite_grads),
                new_state.params,
                state.params,
            ),
            dynamic_scale=dynamic_scale,
        )
        # jax.debug.print("scale: {}", dynamic_scale.scale)
    return new_state, metrics


def eval_step(state: TrainState, batch: Dict[str, jnp.ndarray]):
    labels = batch.pop("label")
    logits = state.apply_fn({"params": state.params}, batch)
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))
    metrics = jax.lax.pmean({"loss": loss}, axis_name="batch")
    return metrics


def data_loader(
    dataset: Dataset,
    batch_size: int,
    drop_last_batch: bool,
    shuffle: bool,
    rng: Optional[jax.random.PRNGKey] = None,
):
    if shuffle:
        if "cpu" in jax.default_backend():  # permutation is slow on cpu jax, use numpy instead
            np_rng = np.random.default_rng()
            batch_idx = np_rng.permutation(len(dataset))
        else:
            batch_idx = jax.random.permutation(rng, len(dataset))
            batch_idx = np.asarray(batch_idx)
    else:
        batch_idx = np.arange(len(dataset))

    if drop_last_batch:  # avoid jit recompilation
        total_num = (len(dataset) // batch_size) * batch_size
    else:
        total_num = len(dataset)

    for i in range(0, total_num, batch_size):
        batch = dataset[batch_idx[i : i + batch_size]]
        yield {k: np.array(v) for k, v in batch.items()}


# https://huggingface.co/docs/datasets/v2.11.0/en/about_mapstyle_vs_iterable#exact-and-fast-approximate-shuffling
def lazy_data_loader(
    dataset: IterableDataset,
    batch_size: int,
    drop_last_batch: bool,
    shuffle: bool,
    seed: Optional[int] = None,
    epoch: Optional[int] = None,
):
    # iter_dataset = dataset.to_iterable_dataset(num_shards=1024)
    # print("data num shards: ", iter_dataset.n_shards)
    if shuffle:
        dataset = dataset.shuffle(seed, buffer_size=2_000_000)
        dataset.set_epoch(epoch)
    for batch in dataset.iter(batch_size, drop_last_batch=drop_last_batch):
        yield {k: np.array(v) for k, v in batch.items()}


def get_data_size(data_path: str):
    data = pl.scan_parquet(data_path)
    return data.select(pl.count()).collect(streaming=True).item()


def main():
    config = read_configs()
    train_data_path = config.data_dir / "parquet" / config.train_data
    eval_data_path = config.data_dir / "parquet" / config.eval_data
    cache_dir = config.data_dir / "huggingface"
    train_batch_size = config.per_device_train_batch_size * jax.device_count()
    eval_batch_size = config.per_device_eval_batch_size * jax.device_count()

    train_data_size = get_data_size(train_data_path)
    eval_data_size = get_data_size(eval_data_path)
    print(f"===== train size: {train_data_size:,}, eval size: {eval_data_size:,} =====")
    print(f"===== num devices: {jax.device_count()} =====\n")

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": str(train_data_path),
            "eval": str(eval_data_path),
        },
        cache_dir=str(cache_dir),
        streaming=config.streaming,
    )
    n_train_steps = train_data_size // train_batch_size
    n_eval_steps = math.ceil(eval_data_size / eval_batch_size)
    rng = jax.random.PRNGKey(config.seed)
    rng, state_rng = jax.random.split(rng)
    state = create_train_state(
        state_rng,
        config.size_map,
        config.learning_rate,
        config.weight_decay,
        config.embed_dim,
        config.mixed_precision,
    )

    p_train_step = jax.pmap(train_step, axis_name="batch", donate_argnums=(0,))
    p_eval_step = jax.pmap(eval_step, axis_name="batch")
    # https://flax.readthedocs.io/en/latest/guides/full_eval.html
    p_eval_step_pad = flax.jax_utils.pad_shard_unpad(
        p_eval_step, static_argnums=(0,), static_return=True
    )
    # copy parameters to each device
    state = flax.jax_utils.replicate(state)

    for epoch in range(1, config.n_epochs + 1):
        train_metrics = []
        if config.streaming:
            train_loader = lazy_data_loader(
                dataset["train"],
                train_batch_size,
                drop_last_batch=True,
                shuffle=True,
                seed=config.seed,
                epoch=epoch,
            )
        else:
            rng, train_rng = jax.random.split(rng)
            train_loader = data_loader(
                dataset["train"],
                train_batch_size,
                drop_last_batch=True,
                shuffle=True,
                rng=train_rng,
            )

        train_loader = map(shard, train_loader)
        train_loader = flax.jax_utils.prefetch_to_device(train_loader, size=2)
        for batch in tqdm(train_loader, total=n_train_steps, desc="Training..."):
            state, metrics = p_train_step(state, batch)
            train_metrics.append(metrics)

        train_metrics = get_metrics(train_metrics)
        train_metrics = jax.tree_util.tree_map(jnp.mean, train_metrics)
        print(f"\nEpoch {epoch} train loss: {(train_metrics['loss']):.4f}")

        eval_metrics = []
        if config.streaming:
            eval_loader = lazy_data_loader(
                dataset["eval"], eval_batch_size, drop_last_batch=False, shuffle=False
            )
        else:
            eval_loader = data_loader(
                dataset["eval"], eval_batch_size, drop_last_batch=False, shuffle=False
            )

        for batch in tqdm(eval_loader, total=n_eval_steps, desc="Evaluating..."):
            metrics = p_eval_step_pad(
                state, batch, min_device_batch=config.per_device_eval_batch_size
            )
            eval_metrics.append(metrics)

        eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_util.tree_map(jnp.mean, eval_metrics)
        print(f"\nEpoch {epoch} eval loss: {(eval_metrics['loss']):.4f}")


if __name__ == "__main__":
    main()
