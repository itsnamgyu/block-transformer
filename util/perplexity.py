"""
Tools for evaluating loss (perplexity) by position. Used for PG19 experiments.
"""
import pandas as pd
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.block_transformer import BlockTransformer
from paths import PROJECT_ROOT


def load_pg19_test_dataset():
    return load_dataset(f"{PROJECT_ROOT}/pg19/pg19.py", split="test")


def get_length_statistics(dataset, key="text"):
    if len(dataset) > 100_000:
        print("Subsampling the dataset to 100,000 samples")
        dataset = dataset.select(range(100000))
    lengths = []
    for sample in dataset:
        lengths.append(len(sample[key]))
    return pd.Series(lengths).describe()


def compute_loss_by_position_over_chunks(model, dataset, tokenizer, context_length=8192, batch_size=16,
                                         device="cuda:0", print_skipped=False):
    """
    :param model:
    :param dataset:
    :param tokenizer:
    :param context_length:
    :param batch_size:
    :param device:
    :return: (average_loss_by_position, total_samples)
    """
    criterion = nn.CrossEntropyLoss(reduction="none")
    total_loss = 0
    total_samples = 0  # sample of `context_length` chunks

    for i, sample in enumerate(tqdm(dataset)):
        # tokenize and split into chunks of context_length
        inputs = tokenizer(sample["text"], return_tensors="pt")
        length = inputs["input_ids"].size(-1)
        if length < context_length:
            if print_skipped:
                print(f"Skipping sample {i} with length {length}")
                continue

        for key in inputs:
            length = inputs[key].shape[1]
            remove = length % context_length
            inputs[key] = inputs[key][:, :-remove].reshape(-1, context_length).to(device)
        inputs["labels"] = inputs["input_ids"].clone()

        # dict of 2d tensors to list of dicts of 1d tensors
        inputs = [dict(zip(inputs, values)) for values in zip(*inputs.values())]
        # inputs[i] = {"input_ids": ..., "attention_mask": ..., "labels": ....}

        # re-batch into sizes of `batch_size`
        dataloader = DataLoader(inputs, batch_size=batch_size)
        for batch in dataloader:
            bs = batch["input_ids"].size(0)
            if isinstance(model, BlockTransformer):
                for key in batch:
                    batch[key] = batch[key].view(bs, -1, model.block_length)
                batch["block_attention_mask"] = batch["attention_mask"].any(dim=-1).long()
            labels = batch.pop("labels")

            with torch.no_grad():
                output = model(**batch)
            v = output["logits"].size(-1)

            if isinstance(model, BlockTransformer):
                loss = criterion(output["logits"].reshape(-1, v), labels[:, 1:].reshape(-1)).view(bs, -1)
            else:
                loss = criterion(output["logits"][:, :-1, :].reshape(-1, v), labels[:, 1:].reshape(-1)).view(bs, -1)
            # (bs, context_length - 1)
            loss = loss.sum(dim=0)  # (context_length -1,)

            total_samples += bs
            total_loss = loss if total_loss is None else total_loss + loss

    average_loss = (total_loss / total_samples).cpu()
    return average_loss, total_samples
