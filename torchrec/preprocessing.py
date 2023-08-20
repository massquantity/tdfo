import random
from collections import Counter
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils import Config


class Preprocessor:
    def __init__(self, config: Config):
        self.raw_data = pd.read_csv(
            config.data_dir / config.train_data,
            sep="::",
            names=["uid", "sid", "rating", "timestamp"],
        )
        self.max_len = config.max_len
        self.sliding_step = config.sliding_step
        self.mask_prob = config.mask_prob
        self.user_count = None
        self.item_count = None
        self.mask_token = None
        self.pad_token = 0
        self.np_rng = np.random.default_rng(config.seed)

    def get_processed_dataframes(self) -> Dict[str, Any]:
        df = self._preprocess()
        return df

    def _preprocess(self) -> Dict[str, Any]:
        df, umap, smap = self._densify_index()
        train, val, test = self._split_df(df, len(umap))
        df = {"train": train, "val": val, "test": test, "umap": umap, "smap": smap}
        self.user_count = len(df["umap"])
        self.item_count = len(df["smap"])
        self.mask_token = self.item_count + 1  # item id range: [1, self.item_count]
        final_df = self._mask_and_labels(self._generate_negative_samples(df))
        return final_df

    def _densify_index(self):
        df = self.raw_data
        umap = {u: i for i, u in enumerate(set(df["uid"]))}
        smap = {s: i + 1 for i, s in enumerate(set(df["sid"]))}
        df["uid"] = df["uid"].map(umap)
        df["sid"] = df["sid"].map(smap)
        return df, umap, smap

    @staticmethod
    def _split_df(df: pd.DataFrame, user_count: int):
        # leave-one-out in the paper
        print("Splitting")
        user_group = df.groupby("uid")
        user2items = user_group.apply(
            lambda d: list(d.sort_values(by="timestamp")["sid"])
        )
        train, val, test = {}, {}, {}
        for user in range(user_count):
            items = user2items[user]
            train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
        return train, val, test

    def _generate_negative_samples(
        self,
        df: Dict[str, Any],
    ) -> Dict[str, Any]:
        # follow the paper, no negative samples in training set
        # 100 negative samples in test set, 2 for random to save time
        test_set_sample_size = 100

        # use popularity random sampling align with paper
        popularity = Counter()
        for user in range(self.user_count):
            popularity.update(df["train"][user])
            popularity.update(df["val"][user])
            popularity.update(df["test"][user])
        items_list, freq = zip(*popularity.items())
        freq_sum = sum(freq)
        prob = [float(i) / freq_sum for i in freq]
        test_negative_samples = {}
        min_size = test_set_sample_size
        print("Sampling negative items")
        for user in range(self.user_count):
            seen = set(df["train"][user])
            seen.update(df["val"][user])
            seen.update(df["test"][user])
            samples = []
            while len(samples) < test_set_sample_size:
                sampled_ids = self.np_rng.choice(
                    items_list, test_set_sample_size * 2, replace=False, p=prob
                )
                sampled_ids = [x for x in sampled_ids if x not in seen]
                samples.extend(sampled_ids[:])
            min_size = min_size if min_size < len(samples) else len(samples)
            test_negative_samples[user] = samples
        if min_size == 0:
            raise RuntimeError(
                "we sampled 0 negative samples for a user, please increase the data size"
            )
        test_negative_samples = {
            key: value[:min_size] for key, value in test_negative_samples.items()
        }
        df["test_negative_samples"] = test_negative_samples
        return df

    def _generate_masked_train_set(self, train_data: Dict[int, List[int]]) -> pd.DataFrame:
        df = []
        for user, seqs in train_data.items():
            # sliding_step = int(0.1 * self.max_len)
            # beg_idx = list(
            #    range(
            #        len(seq) - self.max_len,
            #        0,
            #        -self.sliding_step,
            #    )
            # )
            # beg_idx.append(0)
            # seqs = [seq[i : i + self.max_len] for i in beg_idx[::-1]]
            for i in range(0, len(seqs), self.sliding_step):
                seq = seqs[i: i + self.max_len]
                tokens = []
                labels = []
                for s in seq:
                    prob = random.random()
                    if prob < self.mask_prob:
                        prob /= self.mask_prob

                        if prob < 0.8:
                            tokens.append(self.mask_token)
                        else:
                            tokens.append(random.randint(1, self.item_count))
                        labels.append(s)
                    else:
                        tokens.append(s)
                        labels.append(self.pad_token)  # grads will be ignored in `nn.CrossEntropyLoss`
                if len(tokens) < self.max_len:
                    mask_len = self.max_len - len(tokens)
                    tokens = [self.pad_token] * mask_len + tokens
                    labels = [self.pad_token] * mask_len + labels
                df.append([user, tokens, labels])
        return pd.DataFrame(df, columns=["user", "seqs", "labels"])

    def _generate_labeled_eval_set(
        self,
        train_data: Dict[int, List[int]],
        eval_data: Dict[int, List[int]],
        negative_samples: Dict[int, List[int]],
    ) -> pd.DataFrame:
        df = []
        for user, seqs in train_data.items():
            answer = eval_data[user]
            negs = negative_samples[user]
            candidates = answer + negs
            labels = [1] * len(answer) + [0] * len(negs)
            tokens = seqs
            tokens = tokens + [self.mask_token]
            tokens = tokens[-self.max_len :]
            if len(tokens) < self.max_len:
                padding_len = self.max_len - len(tokens)
                tokens = [self.pad_token] * padding_len + tokens
            df.append([user, tokens, candidates, labels])
        return pd.DataFrame(df, columns=["user", "seqs", "candidates", "labels"])

    def _mask_and_labels(self, df: Dict[str, Any]) -> Dict[str, Any]:
        masked_train_set = self._generate_masked_train_set(df["train"])
        labled_val_set = self._generate_labeled_eval_set(
            df["train"],
            df["val"],
            df["test_negative_samples"],
        )
        train_with_valid = {
            key: df["train"].get(key, []) + df["val"].get(key, [])
            for key in set(list(df["train"].keys()) + list(df["val"].keys()))
        }
        labled_test_set = self._generate_labeled_eval_set(
            train_with_valid, df["test"], df["test_negative_samples"]
        )
        masked_df = {
            "train": masked_train_set,
            "val": labled_val_set,
            "test": labled_test_set,
            "umap": df["umap"],
            "smap": df["smap"],
        }
        return masked_df


if __name__ == "__main__":
    import utils
    config = utils.read_configs()
    pp = Preprocessor(config)
    df = pp.get_processed_dataframes()
    print(df["train"].columns)
    print(df["train"]["seqs"])
    print()
    print(df["val"].columns)
    print(df["val"])
