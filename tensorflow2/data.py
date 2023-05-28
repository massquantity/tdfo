import json
import math
import time
from pathlib import Path

import polars as pl
import tensorflow as tf
from datasets import load_dataset
from tqdm import tqdm

FILE_NUM = 8

DTYPES = {
    "user_id": pl.Int32,
    "book_id": pl.Int32,
    "is_read": pl.Int8,
    "is_reviewed": pl.Int8,
    "rating": pl.Int8,
}

FEAT_COLUMNS = [
    "user_id",
    "item_id",
    "language",
    "is_ebook",
    "format",
    "publisher",
    "pub_decade",
    "avg_rating",
    "num_pages",
]

WRITE_COLUMNS = FEAT_COLUMNS + ["is_read", "is_reviewed", "label"]


def read_original_data(data_dir: Path) -> pl.DataFrame:
    """Read and transform data based on the following steps:

    1. Only keep users with 10 to 250 interactions.
    2. Convert rating >= 4 to label 1, and rating < 4 to label 0.
    3. Sort `book_id` for every user for further data splitting.
    """
    data = pl.scan_csv(data_dir / "goodreads_interactions.csv", dtypes=DTYPES)
    data = (
        data.filter(
            (
                (pl.col("book_id").count().over("user_id") >= 10)
                & (pl.col("book_id").count().over("user_id") <= 250)
            ).alias("num_interactions")
        )
        .with_columns(
            pl.when(pl.col("rating") >= 4)
            .then(1)
            .otherwise(0)
            .alias("label")
            .cast(pl.Int8)
        )
        .drop("rating")
    ).collect(streaming=True)
    data = data.select(
        "user_id",
        pl.col("book_id").sort().over("user_id"),
        "label",
        "is_read",
        "is_reviewed",
    )
    return data


def write_data(
    data_dir: Path,
    reindex_ids: pl.Series,
    interaction_data: pl.DataFrame,
    book_features: pl.DataFrame,
    write_format: str,
    prefix: str,
):
    """Divide data to multiple parts according to reindex_ids and join with book features."""
    write_dir = data_dir / write_format
    Path.mkdir(write_dir, exist_ok=True)
    file_unit = math.ceil(len(reindex_ids) / FILE_NUM)
    if write_format == "tfrecord":
        size_str = json.dumps({f"{prefix}_data_size": len(reindex_ids)})
        (write_dir / f"{prefix}_data_size.json").write_text(size_str)

    for i, offset in enumerate(range(0, len(reindex_ids), file_unit), start=1):
        print(f"writing {prefix} part_{i}...")
        start = time.perf_counter()
        ids = reindex_ids[offset : offset + file_unit]
        part = interaction_data[ids]
        if prefix == "train":
            print("shuffling...")
            part = part.sample(fraction=1.0, shuffle=True, seed=42)

        part = (
            part.join(book_features, on="book_id", how="left")
            .rename({"book_id": "item_id"})
            .select(WRITE_COLUMNS)
        )
        write_path = str(write_dir / f"{prefix}_part_{i}.{write_format}")
        if write_format == "tfrecord":
            write_tfrecord(part, write_path)
        else:
            part.write_parquet(write_path)
        print(f"{prefix} part_{i} finished in {(time.perf_counter() - start):.2f}s")


def write_tfrecord(data: pl.DataFrame, file_path: str):
    # noinspection PyUnresolvedReferences
    def _serialize_example(row):
        feature = {
            "user_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[row[0]])),
            "item_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[row[1]])),
            "language": tf.train.Feature(int64_list=tf.train.Int64List(value=[row[2]])),
            "is_ebook": tf.train.Feature(int64_list=tf.train.Int64List(value=[row[3]])),
            "format": tf.train.Feature(int64_list=tf.train.Int64List(value=[row[4]])),
            "publisher": tf.train.Feature(int64_list=tf.train.Int64List(value=[row[5]])),
            "pub_decade": tf.train.Feature(int64_list=tf.train.Int64List(value=[row[6]])),
            "avg_rating": tf.train.Feature(float_list=tf.train.FloatList(value=[row[7]])),
            "num_pages": tf.train.Feature(float_list=tf.train.FloatList(value=[row[8]])),
            "label": tf.train.Feature(float_list=tf.train.FloatList(value=[row[11]])),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(file_path, options) as writer:
        data_iter = data.iter_rows(named=False)
        for _ in tqdm(range(len(data)), desc="Writing tfrecord...", leave=False):
            row = next(data_iter)
            writer.write(_serialize_example(row))


def read_parquet_data(
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
        columns=FEAT_COLUMNS,
        label_cols="label",
        num_workers=num_workers,
    )
    eval_dataset = dataset["eval"].to_tf_dataset(
        eval_batch_size,
        shuffle=False,
        drop_remainder=False,
        prefetch=True,
        columns=FEAT_COLUMNS,
        label_cols="label",
        num_workers=num_workers,
    )
    return train_dataset, eval_dataset


def read_tfrecord_data(
    train_data_path: Path,
    eval_data_path: Path,
    train_batch_size: int,
    eval_batch_size: int,
):
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

    train_files = tf.data.TFRecordDataset.list_files(str(train_data_path), seed=42)
    train_dataset = (
        tf.data.TFRecordDataset(train_files, compression_type="GZIP", num_parallel_reads=8)
        .shuffle(buffer_size=2_000_000)
        .batch(train_batch_size, drop_remainder=True)
        .map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    eval_files = tf.data.TFRecordDataset.list_files(str(eval_data_path), seed=42)
    eval_dataset = (
        tf.data.TFRecordDataset(eval_files, compression_type="GZIP", num_parallel_reads=8)
        .batch(eval_batch_size, drop_remainder=False)
        .map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_dataset, eval_dataset
