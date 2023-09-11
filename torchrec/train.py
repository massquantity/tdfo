import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import cast, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import IterableDataset
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torch import distributed as torch_dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel as DMP
from torchrec.distributed.types import ModuleSharder
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from tqdm import tqdm

from data import load_streaming_data
from models import Bert4Rec
from preprocessing import EVAL_NEG_NUM, PAD_ID
from utils import get_data_size, read_configs

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
    seqs = batch["eval_seqs"].to(device)
    candidates = batch["candidate_items"].to(device)
    kjt = _to_kjt(seqs, device)
    scores = model(kjt)  # B * T * V
    scores = scores[:, -1, :]  # B * V
    scores = torch.gather(scores, dim=1, index=candidates)
    labels = torch.tensor([1] + [0] * EVAL_NEG_NUM, dtype=torch.float32)
    labels = labels.repeat(scores.size(0), 1).to(device)
    metrics = recalls_and_ndcgs_for_ks(scores, labels)
    return metrics


def recalls_and_ndcgs_for_ks(scores: torch.Tensor, labels: torch.Tensor):
    metrics = dict()
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
    train_loader: DataLoader,
    n_train_steps: int,
    device: torch.device,
    optimizer: EmbOptimType,
    epoch: int,
):
    model.train()
    if torch.cuda.is_available():
        torch.cuda.set_device(torch_dist.get_rank())
    loss_logs = []
    cross_entropy = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
    for batch in tqdm(train_loader, total=n_train_steps, desc=f"Epoch {epoch+1}"):
        seqs = batch["train_interactions"].to(device)
        labels = batch["labels"].to(device)
        kjt = _to_kjt(seqs, device)
        logits = model(kjt)  # B x T x V
        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_logs.append(loss.item())

    outputs = [None for _ in range(torch_dist.get_world_size())]
    torch_dist.all_gather_object(outputs, sum(loss_logs) / len(loss_logs))
    if torch_dist.get_rank() == 0:
        print(f"\nEpoch {epoch + 1}, average loss {(sum(outputs) or 0) / len(outputs)}\n")  # fmt: skip


def _validate(
    model: Union[DDP, DMP],
    eval_loader: DataLoader,
    n_eval_steps: int,
    device: torch.device,
    epoch: int,
):
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.set_device(torch_dist.get_rank())
    keys = [f"Recall@{k}" for k in METRICS_K] + [f"NDCG@{k}" for k in METRICS_K]
    metrics_log = defaultdict(list)
    with torch.no_grad():
        for batch in tqdm(eval_loader, total=n_eval_steps, desc=f"Eval {epoch + 1}"):
            metrics = _calculate_metrics(model, batch, device)
            for key in keys:
                metrics_log[key].append(metrics[key])

    metrics_avg = {
        key: sum(values) / len(values) for key, values in metrics_log.items()
    }
    outputs = [None for _ in range(torch_dist.get_world_size())]
    torch_dist.all_gather_object(outputs, metrics_avg)

    metrics_dist = dict()
    for key in keys:
        # noinspection PyUnresolvedReferences
        metrics_dist[key] = np.mean([m[key] for m in outputs])

    if torch_dist.get_rank() == 0:
        print(f"\nEpoch {epoch + 1}, metrics {metrics_dist}\n")


def train_val_test(
    model: Union[DDP, DMP],
    train_dataset: IterableDataset,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    n_train_steps: int,
    n_eval_steps: int,
    device: torch.device,
    optimizer: EmbOptimType,
    num_epochs: int,
    export_root: str,
):
    _validate(model, eval_loader, n_eval_steps, device, -1)
    for epoch in range(num_epochs):
        # train_loader.sampler.set_epoch(epoch)  # DistributedSampler
        train_dataset.set_epoch(epoch)
        _train_one_epoch(
            model,
            train_loader,
            n_train_steps,
            device,
            optimizer,
            epoch,
        )
        _validate(model, eval_loader, n_eval_steps, device, epoch)
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
    cache_dir = config.data_dir / "huggingface"

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

    if not torch_dist.is_initialized():
        torch_dist.init_process_group(backend=backend)

    world_size = torch_dist.get_world_size()
    train_batch_size = config.per_device_train_batch_size * world_size
    eval_batch_size = config.per_device_eval_batch_size * world_size

    train_data_size = get_data_size(train_data_path)
    eval_data_size = get_data_size(eval_data_path)
    print(f"===== train size: {train_data_size:,}, eval size: {eval_data_size:,} =====")
    print(f"===== num devices: {world_size} =====\n")

    train_dataset, train_loader, eval_loader = load_streaming_data(
        train_data_path,
        eval_data_path,
        cache_dir,
        train_batch_size,
        eval_batch_size,
        config.num_workers,
        config.seed,
    )
    n_train_steps = math.ceil(train_data_size / train_batch_size)
    n_eval_steps = math.ceil(eval_data_size / eval_batch_size)

    path = Path.read_text(config.data_dir / "size_map_bert4rec.json")
    n_items = json.loads(path)["n_items"]
    # 0 for padding, item_count + 1 for mask
    vocab_size = n_items + 2
    print(f"==== vocab size: {vocab_size:,} ====")

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
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay  # fmt: skip
        )

    train_val_test(
        model,
        train_dataset,
        train_loader,
        eval_loader,
        n_train_steps,
        n_eval_steps,
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
    torch_dist.init_process_group("gloo", rank=rank, world_size=world_size)
    main()
    torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
    # import torch.multiprocessing as mp
    # world_size = 2
    # mp.spawn(run_local, args=(world_size,), nprocs=world_size, join=True)
