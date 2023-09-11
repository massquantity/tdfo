import json
import math
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import polars as pl

from utils import read_configs

MIN_INTERACTIONS = 20
MAX_INTERACTIONS = 200
PAD_ID = 0
MASK_ID = -1  # set when n_items is available
EVAL_NEG_NUM = 100  # follow the paper setting
FILE_NUM = 2

DTYPES = {
    "user_id": pl.Int32,
    "book_id": pl.Int32,
    "is_read": pl.Int8,
    "is_reviewed": pl.Int8,
    "rating": pl.Int8,
}


def read_original_data(data_dir: Path) -> pl.DataFrame:
    """Read and transform data based on the following steps:

    1. Only keep users with MIN_INTERACTIONS to MAX_INTERACTIONS interactions.
    2. Sort `book_id` for every user for further data splitting.
    """
    data = pl.scan_csv(data_dir / "goodreads_interactions.csv", dtypes=DTYPES)
    data = data.filter(
        (
            (pl.col("book_id").count().over("user_id") >= MIN_INTERACTIONS)
            & (pl.col("book_id").count().over("user_id") <= MAX_INTERACTIONS)
        ).alias("num_interactions")
    )
    item_sort_expr = pl.col("book_id").sort().over("user_id")
    data = data.select("user_id", item_sort_expr).collect(streaming=True)
    return data


def map_ids(df: pl.DataFrame) -> Tuple[pl.DataFrame, int, int]:
    def _get_sparse_mapping(col_name: str) -> dict:
        unique_vals = df.select(pl.col(col_name).unique()).to_series().to_list()
        mapping = dict(zip(sorted(unique_vals), range(1, len(unique_vals) + 1)))
        return mapping

    user_id_mapping = _get_sparse_mapping("user_id")
    item_id_mapping = _get_sparse_mapping("book_id")
    data = df.select(
        pl.col("user_id").map_dict(
            user_id_mapping, default=PAD_ID, return_dtype=pl.Int32
        ),
        pl.col("book_id").map_dict(
            item_id_mapping, default=PAD_ID, return_dtype=pl.Int32
        ),
    )

    n_users = len(user_id_mapping)
    n_items = len(item_id_mapping)
    global MASK_ID
    MASK_ID = n_items + 1  # item id range: [1, n_items]

    assert data.get_column("user_id").min() == 1
    assert data.get_column("user_id").max() == n_users
    assert data.get_column("book_id").min() == 1
    assert data.get_column("book_id").max() == n_items
    return data, n_users, n_items


def get_item_popularity(df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    popularity = df.groupby("book_id").count().sort("count", descending=True)
    items = popularity.get_column("book_id").to_numpy()
    counts = popularity.get_column("count").to_numpy()
    item_probs = counts / np.sum(counts)
    return items, item_probs


def split_data(df: pl.DataFrame) -> pl.DataFrame:
    """Split interaction items for each user.

    eval and test data will keep the last two items, the rest items are used for training.
    """

    def _split_train_eval_test(interaction: pl.Series) -> pl.Series:
        interaction = interaction.sort()
        length = len(interaction)
        train_items = interaction.head(length - 2)
        eval_items = interaction.take(length - 2)
        test_items = interaction.take(length - 1)
        return train_items, eval_items, test_items

    data = (
        df.lazy()
        .groupby("user_id")
        .agg(
            pl.col("book_id").apply(_split_train_eval_test).alias("total_interactions")
        )
    )
    data = data.select(
        "user_id",
        pl.col("total_interactions").list.get(0).alias("train_interactions"),
        pl.col("total_interactions").list.get(1).alias("eval_interactions"),
    ).collect()
    return data


def add_last_item_mask(df: pl.DataFrame) -> pl.DataFrame:
    """Add mask for last item in every sequence based on the paper."""
    seq_lens = df.select(pl.col("train_interactions").list.lengths().alias("seq_lens"))
    last_item_indices = seq_lens.to_series().cumsum().to_numpy() - 1
    data = df.select("user_id", "train_interactions").explode("train_interactions")
    indices = np.zeros(len(data), dtype=np.int32)
    indices[last_item_indices] = -1
    return data.with_columns(last_item_mask=pl.lit(indices))


def gen_train_mask_data(
    df: pl.DataFrame, mask_prob: float, rng: np.random.Generator
) -> pl.DataFrame:
    """Bert-like data masking.

    Items with probability of `mask_prob` are masked as MASK_ID.
    According to the paper, last items in sequence are also masked.

    Labels are the original items if masking else PAD_ID,
    and grads of PAD_ID will be ignored in `nn.CrossEntropyLoss`
    """
    data = add_last_item_mask(df)
    data = data.lazy().with_columns(
        mask_prob_col=pl.lit(rng.random(size=len(data), dtype=np.float32)),
    )
    mask_condition = (pl.col("mask_prob_col") <= mask_prob) | (pl.col("last_item_mask") == -1)  # fmt: skip
    masked_items_col = (
        pl.when(mask_condition)
        .then(pl.lit(MASK_ID, dtype=pl.Int32))
        .otherwise(pl.col("train_interactions"))
        .alias("train_interactions")
    )
    label_col = (
        pl.when(mask_condition)
        .then(pl.col("train_interactions"))
        .otherwise(pl.lit(PAD_ID, dtype=pl.Int32))
        .alias("labels")
    )
    return data.select("user_id", masked_items_col, label_col).collect()


def get_masked_ratio(train_data: pl.DataFrame) -> float:
    masked_length = len(train_data.filter(pl.col("train_interactions") == MASK_ID))
    return masked_length / len(train_data["user_id"])


def gen_sliding_seq_slow(
    df: pl.DataFrame, seq_len: int, sliding_step: int
) -> pl.DataFrame:
    """This function is slow in polars due to the python runtime."""

    def _sliding_window(seq: pl.Series) -> List[pl.List]:
        length = len(seq)
        # total = pl.concat([seq, pl.Series("mask", [PAD_ID] * seq_len, dtype=x.dtype)])
        total = seq.extend_constant(PAD_ID, n=seq_len)
        return [total.slice(i, seq_len) for i in range(0, length, sliding_step)]

    data = (
        df.lazy()
        .groupby("user_id")
        .agg(
            pl.col("train_interactions").apply(_sliding_window),
            pl.col("labels").apply(_sliding_window),
        )
    )
    # explode nested list to list
    return data.explode(["train_interactions", "labels"]).collect()


# def gen_sliding_seq_unstack(df: pl.DataFrame, seq_len: int) -> pl.DataFrame:
#    data = df.unstack(step=seq_len, how="horizontal", fill_values=PAD_ID)
#    user_col = [col for col in data.columns if "user" in col][0]
#    item_cols = sorted([col for col in data.columns if "interactions" in col])
#    label_cols = sorted([col for col in data.columns if "labels" in col])
#    data = data.select(
#        pl.col(user_col).alias("user_id"),
#        pl.concat_list(item_cols).alias("train_interactions"),
#        pl.concat_list(label_cols).alias("labels"),
#    )
#    return data


def gen_sliding_seq(df: pl.DataFrame, seq_len: int, sliding_step: int) -> pl.DataFrame:
    user_groupby = df.groupby("user_id").agg("train_interactions", "labels")
    user_ids = user_groupby.get_column("user_id").to_numpy().tolist()
    item_ids = user_groupby.get_column("train_interactions").to_numpy().tolist()
    labels = user_groupby.get_column("labels").to_numpy().tolist()
    total_users, total_items, total_labels = [], [], []
    for user, seqs, lbs in zip(user_ids, item_ids, labels):
        pad_len = _get_pad_size(seqs, seq_len, sliding_step)
        padded_seqs = np.append(seqs, ([PAD_ID] * pad_len))
        padded_labels = np.append(lbs, ([PAD_ID] * pad_len))
        for i in range(0, len(seqs), sliding_step):
            total_users.append(user)
            total_items.append(padded_seqs[i : i + seq_len])
            total_labels.append(padded_labels[i : i + seq_len])

    seq_data = pl.DataFrame(
        {
            "user_id": total_users,
            "train_interactions": total_items,
            "labels": total_labels,
        },
        schema={
            "user_id": pl.Int32,
            "train_interactions": pl.List(pl.Int32),
            "labels": pl.List(pl.Int32),
        },
    )
    return seq_data


def _get_pad_size(seqs: List[int], seq_len: int, sliding_step: int) -> int:
    border_index = (len(seqs) - 1) // sliding_step * sliding_step
    return border_index + seq_len - len(seqs)


def get_eval_seqs(df: pl.LazyFrame, seq_len: int, data_size: int) -> pl.LazyFrame:
    """Get interactions for evaluation from last `seq_len` train_interactions, padded with PAD_ID"""
    pad_items = np.full([data_size, seq_len], fill_value=PAD_ID, dtype=np.int32)
    data = df.with_columns(
        pad_items=pl.lit(pad_items, dtype=pl.Int32),
        mask_items=pl.lit(MASK_ID, dtype=pl.Int32),
    )
    eval_seqs = pl.concat_list(["pad_items", "train_interactions", "mask_items"])
    return data.select(
        "user_id", eval_seqs.list.tail(seq_len).alias("eval_seqs"), "eval_interactions"
    )


def sample_negatives(
    data_size: int,
    rng: np.random.Generator,
    items: np.ndarray,
    probs: np.ndarray,
) -> np.ndarray:
    """Sample `EVAL_NEG_NUM` negatives for each user."""
    total_size = data_size * EVAL_NEG_NUM
    # avoid exceeding total item num in `rng.choice`
    step = len(items) // 5 // EVAL_NEG_NUM
    negative_samples = []
    for i in range(0, data_size, step):
        negs = rng.choice(items, step * EVAL_NEG_NUM, replace=False, p=probs)
        negative_samples.extend(negs)
    negative_samples = np.array(negative_samples[:total_size], dtype=np.int32)
    return negative_samples.reshape(data_size, EVAL_NEG_NUM)


def sample_negs_without_pos(
    df: pl.DataFrame,
    rng: np.random.Generator,
    items: np.ndarray,
    probs: np.ndarray,
) -> pl.Series:
    """Sample `EVAL_NEG_NUM` negatives for each user and exclude items from training data."""
    n_users = len(df)
    pos_n_items = df.select(pl.col("train_interactions").list.lengths()).to_series()
    total_size = pos_n_items.sum() + n_users * EVAL_NEG_NUM
    # avoid exceeding total item num in `rng.choice`
    step = len(items) // 5
    all_negative_samples = []
    for i in range(0, total_size, step):
        negs = rng.choice(items, step, replace=False, p=probs)
        all_negative_samples.extend(negs)

    neg_indices = (pos_n_items + EVAL_NEG_NUM).cumsum().to_list()
    negative_samples = np.split(all_negative_samples, neg_indices)[:-1]

    data = (
        df.lazy()
        .with_columns(
            positive_samples=pl.concat_list(
                ["train_interactions", "eval_interactions"]
            ),
            negative_samples=pl.Series(
                values=negative_samples, dtype=pl.List(pl.Int32)
            ),
        )
        .select(
            pl.col("negative_samples")
            .list.set_difference("positive_samples")
            .list.head(EVAL_NEG_NUM)
            .alias("negatives")
        )
    )
    # print(data.select(pl.col("negatives").list.lengths().sort()).collect())
    data = data.collect().to_series()
    return data


def gen_eval_data(
    df: pl.DataFrame,
    seq_len: int,
    rng: np.random.Generator,
    items: np.ndarray,
    probs: np.ndarray,
) -> pl.DataFrame:
    data = get_eval_seqs(df.lazy(), seq_len, len(df))
    # negative_samples = sample_negatives(len(df), rng, items, probs)
    data = data.with_columns(neg_items=sample_negs_without_pos(df, rng, items, probs))
    candidate_items = pl.concat_list(["eval_interactions", "neg_items"])
    return data.select(
        "user_id", "eval_seqs", candidate_items.alias("candidate_items")
    ).collect()


def write_parquet_data(data_dir: Path, data: pl.DataFrame, prefix: str):
    write_dir = data_dir.joinpath("parquet_bert4rec")
    Path.mkdir(write_dir, exist_ok=True)
    file_unit = math.ceil(len(data) / FILE_NUM)
    for i, offset in enumerate(range(0, len(data), file_unit), start=1):
        print(f"writing {prefix} part_{i}...")
        start = time.perf_counter()
        part = data.slice(offset, file_unit)
        if prefix == "train":
            print("shuffling...")
            part = part.sample(fraction=1.0, shuffle=True, seed=42)

        # `datasets` only supports pandas parquet when using List[int]
        part.to_pandas().to_parquet(
            write_dir.joinpath(f"{prefix}_part_{i}.parquet"), index=False
        )
        print(f"{prefix} part_{i} finished in {(time.perf_counter() - start):.2f}s")


def main():
    config = read_configs()
    np_rng = np.random.default_rng(config.seed)

    start_time = time.perf_counter()
    interaction_data = read_original_data(config.data_dir)
    print(f"data reading finished in {(time.perf_counter() - start_time):.2f}s")
    print(f"data size: {interaction_data['user_id'].len():,}")

    data, n_users, n_items = map_ids(interaction_data)
    items, item_probs = get_item_popularity(data)
    print(f"n_users: {n_users:,}, n_items: {n_items:,}")
    with open(config.data_dir / "size_map_bert4rec.json", "w") as f:
        json.dump({"n_users": n_users, "n_items": n_items}, f, indent=4)

    start_time = time.perf_counter()
    data = split_data(data)
    print(f"data split finished in {(time.perf_counter() - start_time):.2f}s")
    # print(data.row(by_predicate=pl.col("user_id") == 1))

    start_time = time.perf_counter()
    masked_train_data = gen_train_mask_data(data, config.mask_prob, np_rng)
    print(f"total masked ratio: {get_masked_ratio(masked_train_data):.4f}")
    masked_seq_train_data = gen_sliding_seq(
        masked_train_data, config.max_len, config.sliding_step
    )
    print(f"train seq data finished in {(time.perf_counter() - start_time):.2f}s")
    print(f"train seq data size: {len(masked_seq_train_data):,}\n")
    # print(masked_seq_train_data.filter(pl.col("user_id") == 2).glimpse())
    write_parquet_data(config.data_dir, masked_seq_train_data, prefix="train")

    start_time = time.perf_counter()
    eval_data = gen_eval_data(data, config.max_len, np_rng, items, item_probs)
    print(f"\neval seq data finished in {(time.perf_counter() - start_time):.2f}s")
    print(f"eval seq data size: {len(eval_data):,}")
    print(
        f"eval_seq len: {len(eval_data.row(0)[1])}, candidate num: {len(eval_data.row(0)[2])}"
    )
    write_parquet_data(config.data_dir, eval_data, prefix="eval")


if __name__ == "__main__":
    main()
