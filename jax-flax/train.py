import json
import math
from pathlib import Path
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import polars as pl
from datasets import Dataset, IterableDataset, load_dataset
from flax.training.train_state import TrainState
from tqdm import trange

from models import init_model

DATA_DIR = Path("data/goodreads").absolute()


def create_train_state(
    rng: jax.random.PRNGKey, size_map: dict, learning_rate: float, weight_decay: float
):
    model, params = init_model(rng, size_map)
    # optimizer = optax.sgd(learning_rate=0.001, momentum=0.9)
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    return TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


@jax.jit
def train_step(state: TrainState, batch: Dict[str, jnp.ndarray]):
    def loss_fn(params):
        labels = batch.pop("label")
        logits = state.apply_fn({"params": params}, batch)
        loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


@jax.jit
def eval_step(state: TrainState, batch: Dict[str, jnp.ndarray]):
    labels = batch.pop("label")
    logits = state.apply_fn({"params": state.params}, batch)
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))
    return loss


def data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    rng: Optional[jax.random.PRNGKey] = None,
):
    if shuffle:
        if "cpu" in jax.default_backend():  # permutation is slow on cpu jax
            np_rng = np.random.default_rng()
            batch_idx = np_rng.permutation(len(dataset))
        else:
            batch_idx = jax.random.permutation(rng, len(dataset))
            batch_idx = np.asarray(batch_idx)
    else:
        batch_idx = np.arange(len(dataset))

    for i in range(0, len(dataset), batch_size):
        batch = dataset[batch_idx[i : i + batch_size]]
        yield {k: np.array(v) for k, v in batch.items()}


# https://huggingface.co/docs/datasets/v2.11.0/en/about_mapstyle_vs_iterable#exact-and-fast-approximate-shuffling
def lazy_data_loader(
    dataset: IterableDataset,
    batch_size: int,
    shuffle: bool,
    seed: Optional[int] = None,
    epoch: Optional[int] = None,
):
    # iter_dataset = dataset.to_iterable_dataset(num_shards=1024)
    # print("data num shards: ", iter_dataset.n_shards)
    if shuffle:
        dataset = dataset.shuffle(seed, buffer_size=20000)
        dataset.set_epoch(epoch)
    for batch in dataset.iter(batch_size, drop_last_batch=False):
        yield {k: np.array(v) for k, v in batch.items()}


def get_data_size(data_path: str):
    data = pl.scan_parquet(data_path)
    return data.select(pl.count()).collect(streaming=True).item()


def main():
    n_epochs = 10
    learning_rate = 3e-4
    weight_decay = 0.0  # 1e-4
    seed = 42
    batch_size, eval_batch_size = 2048, 8192
    streaming = True
    train_data_path = DATA_DIR / "train_1m.parquet"
    eval_data_path = DATA_DIR / "eval_1m.parquet"
    cache_dir = DATA_DIR / "huggingface"
    size_map_path = DATA_DIR / "size_map.json"

    train_data_size = get_data_size(train_data_path)
    eval_data_size = get_data_size(eval_data_path)
    print(f"train size: {train_data_size}, eval size: {eval_data_size}")

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": str(train_data_path),
            "eval": str(eval_data_path),
        },
        cache_dir=str(cache_dir),
        streaming=streaming,
    )
    n_train_steps = math.ceil(train_data_size / batch_size)
    n_eval_steps = math.ceil(eval_data_size / eval_batch_size)
    size_map = json.loads(Path.read_text(size_map_path))
    rng = jax.random.PRNGKey(seed)
    rng, state_rng = jax.random.split(rng)
    state = create_train_state(state_rng, size_map, learning_rate, weight_decay)

    for epoch in range(n_epochs):
        train_loss = []
        if streaming:
            train_loader = lazy_data_loader(
                dataset["train"], batch_size, shuffle=True, seed=seed, epoch=epoch
            )
        else:
            rng, train_rng = jax.random.split(rng)
            train_loader = data_loader(
                dataset["train"], batch_size, shuffle=True, rng=train_rng
            )
        for _ in trange(n_train_steps, desc="Training..."):
            batch = next(train_loader)
            state, loss = train_step(state, batch)
            train_loss.append(loss)
        print(f"train loss: {(jnp.mean(np.array(train_loss)).item()):.4f}")

        eval_loss = []
        if streaming:
            eval_loader = lazy_data_loader(
                dataset["eval"], eval_batch_size, shuffle=False
            )
        else:
            eval_loader = data_loader(
                dataset["eval"], eval_batch_size, shuffle=False
            )
        for _ in trange(n_eval_steps, desc="Evaluating..."):
            batch = next(eval_loader)
            eval_loss.append(eval_step(state, batch))
        print(f"eval loss: {(np.mean(eval_loss)):.4f}")


if __name__ == "__main__":
    main()
