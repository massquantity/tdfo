import datetime
import json
import math
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import polars as pl

from data import read_original_data, write_parquet_data
from utils import read_configs

SPLIT_RATIO = 0.8


def publication_year_to_decade() -> pl.Expr:
    data_col = pl.col("pub_year").str.strptime(pl.Date, "%Y", strict=False)
    data_col = data_col.fill_null(pl.date(1000, 1, 1))
    return (
        pl.when(
            data_col.is_between(datetime.date(1900, 1, 1), datetime.date(1910, 1, 1))
        )
        .then("1900s")
        .when(data_col.is_between(datetime.date(1910, 1, 1), datetime.date(1920, 1, 1)))
        .then("1910s")
        .when(data_col.is_between(datetime.date(1920, 1, 1), datetime.date(1930, 1, 1)))
        .then("1920s")
        .when(data_col.is_between(datetime.date(1930, 1, 1), datetime.date(1940, 1, 1)))
        .then("1930s")
        .when(data_col.is_between(datetime.date(1940, 1, 1), datetime.date(1950, 1, 1)))
        .then("1940s")
        .when(data_col.is_between(datetime.date(1950, 1, 1), datetime.date(1960, 1, 1)))
        .then("1950s")
        .when(data_col.is_between(datetime.date(1960, 1, 1), datetime.date(1970, 1, 1)))
        .then("1960s")
        .when(data_col.is_between(datetime.date(1970, 1, 1), datetime.date(1980, 1, 1)))
        .then("1970s")
        .when(data_col.is_between(datetime.date(1980, 1, 1), datetime.date(1990, 1, 1)))
        .then("1980s")
        .when(data_col.is_between(datetime.date(1990, 1, 1), datetime.date(2000, 1, 1)))
        .then("1990s")
        .when(data_col.is_between(datetime.date(2000, 1, 1), datetime.date(2010, 1, 1)))
        .then("2000s")
        .when(data_col.is_between(datetime.date(2010, 1, 1), datetime.date(2020, 1, 1)))
        .then("2010s")
        .when(data_col.is_between(datetime.date(2020, 1, 1), datetime.date(2030, 1, 1)))
        .then("2020s")
        .otherwise("unknown")
    )


def transform_continuous(df: pl.DataFrame, col_name: str) -> pl.Expr:
    """Remove nulls and outliers, then min-max normalize"""
    values = (
        df.filter(pl.col(col_name) != "")
        .filter(pl.col(col_name).cast(pl.Float32) <= 2000)
        .get_column(col_name)
        .cast(pl.Float32)
    )
    min_val, max_val = values.min(), values.max()
    median_val = round(values.median(), 4)
    data_col = (
        pl.when(pl.col(col_name) == "")
        .then(str(median_val))
        .otherwise(pl.col(col_name))
        .cast(pl.Float32)
        .alias(col_name)
    )
    data_col = pl.when(data_col > 2000).then(median_val).otherwise(data_col)
    return (data_col - min_val) / (max_val - min_val)


def get_sparse_mapping(df: pl.DataFrame, col_name: str) -> dict:
    data_col = (
        pl.when(pl.col(col_name) == "").then("unknown").otherwise(pl.col(col_name))
    )
    unique_vals = sorted(df.select(data_col.unique()).to_series().to_list())
    mapping = dict(zip(unique_vals, range(len(unique_vals))))
    return mapping


def transform_categorical(mapping: dict, col_name: str, dtype: pl.DataType) -> pl.Expr:
    data_col = (
        pl.when(pl.col(col_name) == "").then("unknown").otherwise(pl.col(col_name))
    )
    return data_col.map_dict(mapping, return_dtype=dtype)


def get_user_num(data_dir: Path) -> int:
    user_id_map = pl.read_csv(
        data_dir / "user_id_map.csv",
        new_columns=["user_id", "user_original_id"],
        dtypes={"user_id": pl.Int32},
    )
    return pl.count(user_id_map["user_id"])


def read_item_id_data(data_dir: Path) -> Tuple[pl.DataFrame, int]:
    book_id_map = pl.read_csv(
        data_dir / "book_id_map.csv",
        new_columns=["book_id", "book_original_id"],
        dtypes={"book_id": pl.Int32, "book_original_id": pl.Utf8},
    )
    return book_id_map, pl.count(book_id_map["book_id"])


def get_book_info(data_dir: Path) -> pl.DataFrame:
    size_map = dict()
    size_map["user"] = get_user_num(data_dir)
    book_id_map, book_num = read_item_id_data(data_dir)
    size_map["item"] = book_num

    books = pl.scan_ndjson(data_dir / "goodreads_books.json")
    books_df = books.select(
        [
            pl.col("book_id").alias("book_original_id"),
            pl.col("language_code").alias("language"),
            pl.col("is_ebook"),
            pl.col("average_rating").alias("avg_rating"),
            pl.col("format"),
            pl.col("publisher"),
            pl.col("num_pages"),
            pl.col("publication_year").alias("pub_year"),
        ]
    ).collect(streaming=True)
    books_df = books_df.with_columns(publication_year_to_decade().alias("pub_decade"))

    category_col_exprs = []
    category_cols = [
        ("language", pl.Int16),
        ("is_ebook", pl.Int8),
        ("format", pl.Int16),
        ("publisher", pl.Int32),
        ("pub_decade", pl.Int8),
    ]
    for col, dtype in category_cols:
        mapping = get_sparse_mapping(books_df, col)
        trans_col = transform_categorical(mapping, col, dtype).alias(col)
        category_col_exprs.append(trans_col)
        size_map[col] = len(mapping)

    books_df = books_df.select(
        "book_original_id",
        *category_col_exprs,
        transform_continuous(books_df, "avg_rating").alias("avg_rating"),
        transform_continuous(books_df, "num_pages").alias("num_pages"),
    )
    book_features = book_id_map.join(books_df, on="book_original_id", how="left")
    book_features = book_features.drop("book_original_id")
    assert not np.any(book_features.null_count().row(0))  # no nulls left in data
    return book_features, size_map


def split_train_test(interaction: pl.Series, ratio: float, is_train: bool) -> pl.Series:
    """Split interaction items for one user."""
    train_offset = math.ceil(interaction.len() * ratio)
    interaction = interaction.sort()
    return (
        interaction.head(train_offset)
        if is_train
        else interaction.tail(len(interaction) - train_offset)
    )


def split_data_by_index(data: pl.DataFrame, is_train: bool) -> pl.Series:
    """Split interaction items for each user then flatten lists and get row ids."""
    return (
        data.lazy()
        .groupby("user_id")
        .agg(
            pl.col("book_id")
            .agg_groups()
            .apply(lambda x: split_train_test(x, SPLIT_RATIO, is_train=is_train))
            .alias("reindex_ids")
        )
        .explode("reindex_ids")
        .select("reindex_ids")
        .collect(streaming=True)
        .get_column("reindex_ids")
    )


def write_size_map(data_dir: Path, size_map: dict):
    with open(data_dir / "size_map.json", "w") as f:
        json.dump(size_map, f, indent=4)


def main():
    config = read_configs()
    book_features, size_map = get_book_info(config.data_dir)
    write_size_map(config.data_dir, size_map)

    interaction_data = read_original_data(config.data_dir)
    start_split = time.perf_counter()
    reindex_ids = split_data_by_index(interaction_data, is_train=True)
    print(f"train split finished in {(time.perf_counter() - start_split):.2f}s")
    print(f"train data size: {len(reindex_ids)}\n")
    write_parquet_data(
        config.data_dir, reindex_ids, interaction_data, book_features, "train"
    )

    start_split = time.perf_counter()
    reindex_ids = split_data_by_index(interaction_data, is_train=False)
    print(f"\neval split finished in {(time.perf_counter() - start_split):.2f}s")
    print(f"eval data size: {len(reindex_ids)}")
    write_parquet_data(
        config.data_dir, reindex_ids, interaction_data, book_features, "eval"
    )


if __name__ == "__main__":
    main()
