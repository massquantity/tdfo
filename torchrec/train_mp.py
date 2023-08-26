import json
import os
from collections import defaultdict
from pathlib import Path
from typing import cast, Dict, List, Union

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel as DMP
from torchrec.distributed.types import ModuleSharder
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from tqdm import tqdm

from data import Bert4RecDataLoader
from models import Bert4Rec
from utils import get_data_size, read_configs

PAD_ID = 0
EVAL_NEG_NUM = 100
METRICS_K = (10, 20, 50)


def _to_kjt(seqs: torch.LongTensor, device: torch.device) -> KeyedJaggedTensor:
    seqs_list = list(seqs)
    lengths = torch.IntTensor([value.size(0) for value in seqs_list])
    values = torch.cat(seqs_list, dim=0)

    kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=["item"], values=values, lengths=lengths
    ).to(device)
    return kjt


def _calculate_metrics(
    model: Union[DDP, DMP],
    batch: List[torch.LongTensor],
    device: torch.device,
) -> Dict[str, float]:
    seqs, candidates = batch
    kjt = _to_kjt(seqs, device)
    scores = model(kjt)  # B * T * V
    scores = scores[:, -1, :]  # B * V
    scores = torch.gather(scores, dim=1, index=candidates)
    metrics = recalls_and_ndcgs_for_ks(scores)
    return metrics


def recalls_and_ndcgs_for_ks(scores: torch.Tensor):
    metrics = dict()
    batch_size = scores.size(dim=0)
    labels = [1] + [0] * EVAL_NEG_NUM
    labels = torch.tensor(labels, dtype=torch.float32).repeat(batch_size, 1)
    answer_counts = labels.sum(dim=1)
    _, cut = torch.sort(-scores, dim=1)
    for k in METRICS_K:
        k_cut = cut[:, :k]
        hits = torch.gather(labels, dim=1, index=k_cut)
        label_len = torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1))
        metrics[f"Recall@{k}"] = (hits.sum(1) / label_len).mean().cpu().item()

        position = torch.arange(2, 2 + k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights.to(hits.device)).sum(1)
        idcg = torch.Tensor([weights[: min(int(n), k)].sum() for n in answer_counts])
        idcg = idcg.to(dcg.device)
        metrics[f"NDCG@{k}"] = (dcg / idcg).mean().cpu().item()

    return metrics


def _train_one_epoch(
    model: Union[DDP, DMP],
    train_loader: data_utils.DataLoader,
    device: torch.device,
    optimizer: EmbOptimType,
    epoch: int,
):
    model.train()
    if torch.cuda.is_available():
        torch.cuda.set_device(dist.get_rank())
    loss_logs = []
    cross_entropy = nn.CrossEntropyLoss(ignore_index=PAD_ID)  # todo: label_smoothing
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        batch = [x.to(device) for x in batch]
        optimizer.zero_grad()
        seqs, labels = batch

        kjt = _to_kjt(seqs, device)
        logits = model(kjt)  # B x T x V
        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()
        loss_logs.append(loss.item())

    outputs = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(outputs, sum(loss_logs) / len(loss_logs))
    if dist.get_rank() == 0:
        print(f"\nEpoch {epoch + 1}, average loss {(sum(outputs) or 0) / len(outputs)}\n")


def _validate(
    model: Union[DDP, DMP],
    val_loader: data_utils.DataLoader,
    device: torch.device,
    epoch: int,
):
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.set_device(dist.get_rank())
    keys = [f"Recall@{k}" for k in METRICS_K] + [f"NDCG@{k}" for k in METRICS_K]
    metrics_log = defaultdict(list)
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Eval {epoch+1}"):
            batch = [x.to(device) for x in batch]
            metrics = _calculate_metrics(model, batch, device)
            for key in keys:
                metrics_log[key].append(metrics[key])

    metrics_avg = {
        key: sum(values) / len(values) for key, values in metrics_log.items()
    }
    outputs = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(outputs, metrics_avg)

    metrics_dist = dict()
    for key in keys:
        # noinspection PyUnresolvedReferences
        metrics_dist[key] = np.mean([m[key] for m in outputs])

    if dist.get_rank() == 0:
        print(f"\nEpoch {epoch + 1}, metrics {metrics_dist}\n")


def train_val_test(
    model: Union[DDP, DMP],
    train_loader: data_utils.DataLoader,
    val_loader: data_utils.DataLoader,
    device: torch.device,
    optimizer: EmbOptimType,
    num_epochs: int,
    export_root: str,
):
    _validate(model, val_loader, device, -1)
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)  # DistributedSampler
        _train_one_epoch(
            model,
            train_loader,
            device,
            optimizer,
            epoch,
        )
        _validate(model, val_loader, device, epoch)
        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                export_root + f"epoch_{epoch + 1}_model.pth",
            )
            print(f"Epoch {epoch + 1} model has been saved to {export_root}")


def main():
    config = read_configs()
    train_data_path = config.data_dir / "parquet_bert4rec" / config.train_data
    eval_data_path = config.data_dir / "parquet_bert4rec" / config.eval_data
    # cache_dir = config.data_dir / "huggingface"

    rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
        optimizer = EmbOptimType.ADAM
    else:
        device = torch.device("cpu")
        backend = "gloo"
        optimizer = EmbOptimType.SGD  # fbgemm CPU version doesn't support ADAM

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    world_size = dist.get_world_size()
    train_batch_size = config.per_device_train_batch_size * world_size
    eval_batch_size = config.per_device_eval_batch_size * world_size

    train_data_size = get_data_size(train_data_path)
    eval_data_size = get_data_size(eval_data_path)
    print(f"===== train size: {train_data_size:,}, eval size: {eval_data_size:,} =====")
    print(f"===== num devices: {world_size} =====\n")

    train_data = pl.read_parquet(train_data_path).to_pandas()
    eval_data = pl.read_parquet(eval_data_path).to_pandas()
    path = Path.read_text(config.data_dir / "size_map_bert4rec.json")
    n_items = json.loads(path)["n_items"]
    # 0 for padding, item_count + 1 for mask
    vocab_size = n_items + 2
    print(f"==== vocab size: {vocab_size:,} ====")

    train_loader, eval_loader = Bert4RecDataLoader(
        train_data,
        eval_data,
        train_batch_size,
        eval_batch_size,
    ).get_pytorch_dataloaders(rank, world_size)

    model_bert4rec = Bert4Rec(
        vocab_size,
        config.max_len,
        config.embed_dim,
        config.n_heads,
        config.n_layers,
    ).to(device)

    if config.model_parallel:
        fused_params = {
            "optimizer": optimizer,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
        }
        model = DMP(
            module=model_bert4rec,
            device=device,
            sharders=[
                cast(ModuleSharder[nn.Module], EmbeddingCollectionSharder(fused_params))
            ],
        )
        dense_optimizer = KeyedOptimizerWrapper(
            dict(in_backward_optimizer_filter(model.named_parameters())),
            lambda params: optim.Adam(
                params, lr=config.learning_rate, weight_decay=config.weight_decay
            ),
        )
        optimizer = CombinedOptimizer([model.fused_optimizer, dense_optimizer])
    else:
        device_ids = [rank] if backend == "nccl" else None
        model = DDP(model_bert4rec, device_ids=device_ids)
        optimizer = optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

    train_val_test(
        model,
        train_loader,
        eval_loader,
        device,
        optimizer,
        config.n_epochs,
        export_root="bert4rec",
    )


def run_local(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["LOCAL_RANK"] = "0"
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    main()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
    # import torch.multiprocessing as mp
    # world_size = 2
    # mp.spawn(run_local, args=(world_size,), nprocs=world_size, join=True)
