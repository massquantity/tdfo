# Distributed Training using TensorFlow2

Before running, one should set some hyperparameters in [config.toml](https://github.com/massquantity/tdfo/blob/main/tensorflow2/config.toml). Some of these are worth mentioning:

+ `data_dir`: Directory of the downloaded data. It is also where the preprocessed data will be saved.
+ `write_format`: The preprocessed file saving format, which must be `tfrecord` or `parquet`. `tfrecord` is faster for data loading, whereas `parquet` uses less disk space.
+ `num_workers`: Number of workers to use for loading the dataset. Only supported on Python versions >= 3.8.
+ `train_data`: The preprocessed training data is saved in multiple parts, and one can use wild card(*) to specify all the files.
+ `eval_data`: The preprocessed evaluation data.
+ `weight_decay`: Weight decay value in the [AdamW](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/AdamW) optimizer.
+ `per_device_train_batch_size`: Training batch size in each device. The total training batch size would be `per_device_train_batch_size * num_devices`.
+ `per_device_eval_batch_size`: Evaluation batch size in each device.
+ `jit_xla`: Whether to compile the model training step with [XLA](https://www.tensorflow.org/xla), which *may* speed up training multiple times.
+ `use_tpu`: Whether to use Google Cloud TPU for training if available.



## Requirements

+ python >= 3.8
+ tensorflow == 2.12.0
+ polars == 0.17.8
+ datasets == 2.11.0
+ tqdm
+ tomli

[Polars](https://github.com/pola-rs/polars) is used for fast and streaming data processing. [🤗 Datasets](https://github.com/huggingface/datasets) is used for efficient data loading and memory-mapping. These libraries are helpful for dealing with large datasets.



## Data preprocessing

Original data comes from the [Goodreads Datasets](https://github.com/MengtingWan/goodreads), which contains multiple versions.

We use `goodreads_interactions.csv`, `user_id_map.csv`, `book_id_map.csv` in [link](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/shelves) and `goodreads_books.json.gz` in [link](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/books).

Make sure they are downloaded into the `data_dir` folder in `config.toml`, then run the data preprocessing script:

```shell
$ python preprocessing.py
```



## Training

Run the `train.py` script if you are using CPU or single GPU.

```shell
$ python train.py
```

Run the `train_dp.py` script if you are using multiple GPUs or TPU. `dp` stands for [*data parallelism*](https://en.wikipedia.org/wiki/Data_parallelism).

```shell
$ python train_dp.py
```

