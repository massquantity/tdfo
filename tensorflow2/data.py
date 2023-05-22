import math
import time
from pathlib import Path

import polars as pl

FILE_NUM = 8

DTYPES = {
    "user_id": pl.Int32,
    "book_id": pl.Int32,
    "is_read": pl.Int8,
    "is_reviewed": pl.Int8,
    "rating": pl.Int8,
}

FINAL_COLUMNS = [
    "user_id",
    "item_id",
    "language",
    "is_ebook",
    "format",
    "publisher",
    "pub_decade",
    "avg_rating",
    "num_pages",
    "is_read",
    "is_reviewed",
    "label",
]


def read_original_data(data_dir: Path) -> pl.DataFrame:
    """Read and transform data based on the following steps:

    1. Only keep users with more than 10 interactions.
    2. Convert to rating >= 4 to label 1, and rating < 4 to label 0.
    3. Sort `book_id` for every user for further data splitting.
    """
    data = pl.scan_csv(data_dir / "goodreads_interactions.csv", dtypes=DTYPES)
    data = (
        data.filter(
            pl.col("book_id").count().over("user_id").alias("num_interactions") >= 10
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
    print(f"data size: {pl.count(data['user_id'])}")
    return data


def write_parquet_data(
    data_dir: Path,
    reindex_ids: pl.Series,
    interaction_data: pl.DataFrame,
    book_features: pl.DataFrame,
    prefix: str,
):
    """Divide data to multiple parts according to reindex_ids and join with book features."""
    parquet_dir = data_dir / "parquet"
    Path.mkdir(parquet_dir, exist_ok=True)
    file_unit = math.ceil(len(reindex_ids) / FILE_NUM)
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
            .select(FINAL_COLUMNS)
        )
        part.write_parquet(parquet_dir / f"{prefix}_part_{i}.parquet")
        print(f"{prefix} part_{i} finished in {(time.perf_counter() - start):.2f}s")
