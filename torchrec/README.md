# Distributed Training using TorchRec

Before running, one should set some hyperparameters in [config.toml](https://github.com/massquantity/tdfo/blob/main/torchrec/config.toml). Some of these are worth mentioning:

+ `data_dir`: Directory of the downloaded data. It is also where the preprocessed data will be saved.
+ `train_data`: The preprocessed training data is saved in multiple parts, and one can use wild card(*) to specify all the files.
+ `eval_data`: The preprocessed evaluation data.
+ `n_heads`: Number of heads in multi-head attention.
+ `n_layers`: Number of transformer layers.
+ `max_len`: Maximum sequence length.
+ `sliding_step`: Sliding window size for building sequences.
+ `mask_prob`: Probability of masking items, Bert style.
+ `per_device_train_batch_size`: Training batch size in each device. The total training batch size would be `per_device_train_batch_size * num_devices`.
+ `per_device_eval_batch_size`: Evaluation batch size in each device.
+ `model_parallel`: Whether to use embedding model parallelism. If False, data parallelism will be used.
+ `num_workers`: Number of workers in data loading.



## Requirements

+ python >= 3.8
+ torchrec == 0.4.0
+ torchx == 0.5.0
+ polars == 0.18.15
+ datasets == 2.14.5
+ tqdm
+ tomli

[Polars](https://github.com/pola-rs/polars) is used for fast and streaming data processing. [🤗 Datasets](https://github.com/huggingface/datasets) is used for efficient data loading and memory-mapping. These libraries are helpful for dealing with large datasets.

[TorchRec](https://github.com/pytorch/torchrec) is used for data-parallelism/model-parallelism distributed training. 
[torchx](https://github.com/pytorch/torchx) is a universal job launcher for PyTorch applications.



## Data preprocessing

Original data comes from the [Goodreads Datasets](https://github.com/MengtingWan/goodreads), which contains multiple versions.

We use `goodreads_interactions.csv` in [link](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/shelves).

Make sure they are downloaded into the `data_dir` folder in `config.toml`, then run the data preprocessing script:

```shell
$ python preprocessing.py
```


## Training

Using torchx to run the `train.py` script. 
Note that if you are using CPU, `model_parallel` can't be used.

```shell
$ torchx run -s local_cwd dist.ddp -j 1x2 --script train.py
```
