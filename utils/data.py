from __future__ import annotations

from dataclasses import dataclass
from typing import *

import torch
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers.utils import PaddingStrategy


def build_dataset(
    tokenizer: AutoTokenizer,
    ds: Union[Dataset, str],
    label_type: Optional[
        Literal["oracle", "sampled", "interpolated", "fineinterpolated"]
    ] = None,
    diff_scaling_factor: float = 1.0,
    seed: int = 42,
    id_eval_size: int = 1024,
    num_proc: int = 16,
    max_train_size: Optional[int] = None,
    clean_cache: bool = False,
) -> Tuple[Dataset, Dataset]:
    """Perform tokenization and annotation from preference dataset.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        Any tokenizer from Huggingface transformers.
    ds : Union[Dataset, str]
        A dataset or a path to a dataset (online or local).
        The dataset should at least have the following columns:
            - chosen
            - rejected
        To add labels, also make sure the dataset has the following columns:
            - chosen_score
            - rejected_score
    label_type : Literal[oracle, sampled, interpolated, fineinterpolated], optional
        The ordinal feedback label type, by default None
        - oracle: as is
        - sampled: 2-level
        - interpolated: 3-level
        - fineinterpolated: 5-level
    diff_scaling_factor : float, optional
        the scaling factor (to be multiplied with the raw score), by default 1.0
    seed : int, optional
        random seed for reproducibility, by default 42
    id_eval_size : int, optional
        eval set size sampled from the training set, by default 1024
    num_proc : int, optional
        process number for tokenization, by default 16
    max_train_size : Optional[int], optional
        maximum size of the training set, by default None
    clean_cache : bool, optional
        clean cache files after loading the dataset, by default False

    Returns
    -------
    Tuple[Dataset, Dataset]
        train_dataset, eval_dataset
    """

    def tokenize(sample):
        """ """
        sample["positive"] = tokenizer.apply_chat_template(
            sample["chosen"], tokenize=False, add_generation_prompt=False
        )
        if isinstance(tokenizer.bos_token, str):
            sample["positive"] = sample["positive"].replace(tokenizer.bos_token, "")
        
        sample["negative"] = tokenizer.apply_chat_template(
            sample["rejected"], tokenize=False, add_generation_prompt=False
        )
        if isinstance(tokenizer.bos_token, str):
            sample["negative"] = sample["negative"].replace(tokenizer.bos_token, "")

        tokenized_pos = tokenizer(sample["positive"], truncation=True)
        tokenized_neg = tokenizer(sample["negative"], truncation=True)
        sample["input_ids_j"] = tokenized_pos["input_ids"]
        sample["attention_mask_j"] = tokenized_pos["attention_mask"]
        sample["input_ids_k"] = tokenized_neg["input_ids"]
        sample["attention_mask_k"] = tokenized_neg["attention_mask"]
        sample["score_j"] = float(sample["chosen_score"])
        sample["score_k"] = float(sample["rejected_score"])

        # ! we make sure chosen/rejected are determined by score_j and score_k
        if sample["score_j"] - sample["score_k"] < 0:
            sample["input_ids_j"] = tokenized_neg["input_ids"]
            sample["attention_mask_j"] = tokenized_neg["attention_mask"]
            sample["input_ids_k"] = tokenized_pos["input_ids"]
            sample["attention_mask_k"] = tokenized_pos["attention_mask"]
            sample["score_j"] = float(sample["rejected_score"])
            sample["score_k"] = float(sample["chosen_score"])

        # ! we scale score_j and score_k with diff_scaling_factor here
        # ! instead of keeping the raw score in the dataitem,
        # ! as we might use them for evaluation loss computation later.
        sample["score_j"] *= diff_scaling_factor
        sample["score_k"] *= diff_scaling_factor
        # add discrete label
        if label_type is None:
            pass
        elif label_type == "oracle":
            sample["label"] = torch.nn.functional.sigmoid(
                torch.tensor(sample["score_j"] - sample["score_k"])
            ).item()  # remember to convert to regular float
        elif label_type == "sampled":
            true_p = torch.nn.functional.sigmoid(
                torch.tensor(sample["score_j"] - sample["score_k"])
            )
            sampled_p = torch.bernoulli(true_p)
            sample["label"] = sampled_p.item()
        elif label_type == "interpolated":
            true_p = torch.nn.functional.sigmoid(
                torch.tensor(sample["score_j"] - sample["score_k"])
            )
            if true_p.item() < 0.5:
                sampled_p = torch.bernoulli(true_p * 2)
                sample["label"] = sampled_p.item() * 0.5
            elif true_p.item() > 0.5:
                sampled_p = torch.bernoulli(true_p * 2 - 1)
                sample["label"] = sampled_p.item() * 0.5 + 0.5
            else:
                sample["label"] = 0.5
        elif label_type == "fineinterpolated":
            true_p = torch.nn.functional.sigmoid(
                torch.tensor(sample["score_j"] - sample["score_k"])
            )
            # we have 5 interpolation points: [0, 0.2, 0.5, 0.8, 1]
            p_intervals = {
                "significantly_worse": (0, 0.2),
                "slightly_worse": (0.2, 0.5),
                "slightly_better": (0.5, 0.8),
                "significantly_better": (0.8, 1),
            }
            for k, v in p_intervals.items():
                if v[0] <= true_p.item() <= v[1]:
                    p_case = k
                    break
            lower, upper = p_intervals[p_case]
            # Bernoulli sampling with
            # (p - p_lower) / (p_upper - p_lower) -> p_upper
            # (p_upper - p) / (p_upper - p_lower) -> p_lower
            sampled_p = (
                torch.bernoulli((true_p - lower) / (upper - lower))
                * (upper - lower)
                + lower
            )
            sample["label"] = sampled_p.item()
        else:
            raise ValueError("Invalid label_type")

        return sample

    if isinstance(ds, str):
        try:
            ds = load_dataset(ds, split="train")
        except:
            ds = load_from_disk(ds)

    if clean_cache:
        ds.cleanup_cache_files()

    ds = ds.map(lambda x, i: {"__ID__": i}, with_indices=True)
    ds = ds.shuffle(seed=seed)
    ds = ds.map(tokenize, num_proc=num_proc)

    eval_dataset = ds.select(range(id_eval_size))

    if max_train_size is None:
        train_dataset = ds.select(range(id_eval_size, len(ds)))
    else:
        train_dataset = ds.select(
            range(id_eval_size, min(max_train_size + id_eval_size, len(ds)))
        )

    return train_dataset, eval_dataset


def post_filter_by_ratio(
    ds: Dataset,
    n_samples: int = 5000,
    positive_ratio: float = 1.0,
    seed: int = 42,
):
    """Filter the dataset by ratio of binary and tied samples.

    Parameters
    ----------
    ds : Dataset
        Dataset from `build_dataset` function.
    n_samples : int, optional
        the number of samples to return, by default 5000
    positive_ratio : float, optional
        the binary sample ratio, by default 1.0
    seed : int, optional
        seed for reproducibility, by default 42

    Returns
    -------
    Dataset
        filtered dataset
    """
    n_pos = int(n_samples * positive_ratio)
    n_tied = n_samples - n_pos

    ds_pos = ds.filter(lambda x: x["label"] == 1)
    ds_tied = ds.filter(lambda x: x["label"] < 1 and x["label"] > 0)

    return concatenate_datasets(
        [
            ds_pos.shuffle(seed=seed).select(range(min(n_pos, len(ds_pos)))),
            ds_tied.shuffle(seed=seed).select(range(min(n_tied, len(ds_tied)))),
        ]
    )


@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        scores = []
        merged_features = []
        scores_j = []
        scores_k = []
        ids = []
        if "label" in features[0]:
            extra_labels = []
        for feature in features:
            # by sequentially appending j and k samples,
            # we make sure positive and negative samples are paired; then we apply padding.
            merged_features.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            scores.append(feature["score_j"])
            scores_j.append(feature["score_j"])
            merged_features.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
            scores.append(feature["score_k"])
            scores_k.append(feature["score_k"])
            if "label" in feature:
                extra_labels.append(feature["label"])
            ids.append(feature["__ID__"])
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": torch.tensor(scores, dtype=torch.float),
            "labels_j": torch.tensor(scores_j, dtype=torch.float),
            "labels_k": torch.tensor(scores_k, dtype=torch.float),
            "ids": torch.tensor(ids, dtype=torch.long),
            "return_loss": True,
        }
        if "label" in features[0]:  # used for loss computation
            batch["labels_extra"] = torch.tensor(
                extra_labels, dtype=torch.float
            )

        return batch
