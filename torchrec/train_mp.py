import os
from typing import cast, Dict, List, Union

import numpy as np
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
from preprocessing import Preprocessor
from utils import read_configs


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
    seqs, candidates, labels = batch
    kjt = _to_kjt(seqs, device)
    scores = model(kjt)  # B * T * V
    scores = scores[:, -1, :]  # B * V
    scores = torch.gather(scores, dim=1, index=candidates)
    metrics = recalls_and_ndcgs_for_ks(scores, labels)
    return metrics


def recalls_and_ndcgs_for_ks(scores: torch.Tensor, labels: torch.Tensor):
    metrics = {}
    # scores = scores
    # labels = labels
    answer_count = labels.sum(dim=1)
    labels_float = labels.float()
    _, cut = torch.sort(-scores, dim=1)
    for k in (10, 100):
        k_cut = cut[:, :k]
        hits = torch.gather(labels_float, dim=1, index=k_cut)
        label_len = torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())
        metrics["Recall@%d" % k] = (hits.sum(1) / label_len).mean().cpu().item()

        position = torch.arange(2, 2 + k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights.to(hits.device)).sum(1)
        # idcg = (answer_count[:, :k] * weights.to(answer_count.device)).sum(1)
        idcg = torch.Tensor([weights[: min(int(n), k)].sum() for n in answer_count])
        idcg = idcg.to(dcg.device)
        metrics["NDCG@%d" % k] = (dcg / idcg).mean().cpu().item()

    return metrics


def _train_one_epoch(
    model: Union[DDP, DMP],
    train_loader: data_utils.DataLoader,
    device: torch.device,
    optimizer: optim.Adam,
    epoch: int,
):
    model.train()
    if torch.cuda.is_available():
        torch.cuda.set_device(dist.get_rank())
    loss_logs = []
    train_iterator = iter(train_loader)
    cross_entropy = nn.CrossEntropyLoss(ignore_index=0)  # todo: label_smoothing
    outputs = [None for _ in range(dist.get_world_size())]
    for batch in tqdm(train_iterator, desc=f"Epoch {epoch+1}"):
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

    dist.all_gather_object(outputs, sum(loss_logs) / len(loss_logs))
    if dist.get_rank() == 0:
        print(f"Epoch {epoch + 1}, average loss { (sum(outputs) or 0) / len(outputs)}")


def _validate(
    model: Union[DDP, DMP],
    val_loader: data_utils.DataLoader,
    device: torch.device,
    epoch: int,
    is_testing: bool = False,
) -> None:
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.set_device(dist.get_rank())
    outputs = [None for _ in range(dist.get_world_size())]
    keys = ["Recall@10",  "Recall@100", "NDCG@10", "NDCG@100"]
    metrics_log: Dict[str, List[float]] = {key: [] for key in keys}

    with torch.no_grad():
        for _, batch in enumerate(val_loader):
            batch = [x.to(device) for x in batch]

            metrics = _calculate_metrics(model, batch, device)

            for key in keys:
                metrics_log[key].append(metrics[key])

    metrics_avg = {
        key: sum(values) / len(values) for key, values in metrics_log.items()
    }
    dist.all_gather_object(outputs, metrics_avg)

    def _dict_mean(dict_list: List[Dict[str, float]]) -> Dict[str, float]:
        mean_dict = {}
        for key in dict_list[0].keys():
            mean_dict[key] = np.mean([d[key] for d in dict_list], axis=0)
        return mean_dict

    if dist.get_rank() == 0:
        print(
            f"{'Epoch ' + str(epoch + 1) if not is_testing else 'Test'}, metrics {_dict_mean(outputs)}"
        )


def train_val_test(
    model: Union[DDP, DMP],
    train_loader: data_utils.DataLoader,
    val_loader: data_utils.DataLoader,
    test_loader: data_utils.DataLoader,
    device: torch.device,
    optimizer: optim.Adam,
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
            print(f"epoch {epoch + 1} model has been saved to {export_root}")
    _validate(model, test_loader, device, num_epochs, is_testing=True)


def main():
    config = read_configs()
    use_dmp = True
    rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    world_size = dist.get_world_size()
    batch_size = config.per_device_train_batch_size * world_size

    df = Preprocessor(config).get_processed_dataframes()
    # 0 for padding, item_count + 1 for mask
    vocab_size = len(df["smap"]) + 2
    print("vocab size: ", vocab_size)
    bert4recDataloader = Bert4RecDataLoader(
        df,
        batch_size,
        batch_size,
        batch_size,
    )
    (
        train_loader,
        val_loader,
        test_loader,
    ) = bert4recDataloader.get_pytorch_dataloaders(rank, world_size)

    model_bert4rec = Bert4Rec(
        vocab_size,
        config.max_len,
        config.embed_dim,
        config.n_heads,
        config.n_layers,
    ).to(device)

    if use_dmp:
        fused_params = {
            "optimizer": EmbOptimType.SGD,  # ADAM
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
        val_loader,
        test_loader,
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
    # main()
    import torch.multiprocessing as mp
    world_size = 2
    mp.spawn(run_local, args=(world_size,), nprocs=world_size, join=True)
