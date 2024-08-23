"""
Implementation based on trl.trainer.ConstantLengthDataset
"""
import random
from typing import List, Iterator

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer


class LanguageModelingDataset(IterableDataset):
    def __init__(self, dataset: Dataset, tokenizer: PreTrainedTokenizer, block_length, max_length,
                 dataset_text_field=None, data_formatter=None, continuous=True, buffer_size=2 ** 22,
                 seed=42, global_shuffling=True, local_shuffling=True, random_pad_first_block=True,
                 pad_to_block_boundary=True, transforms: list = None):
        """
        :param dataset:
        :param tokenizer: make sure the tokenizer automatically adds eos tokens
        :param block_length: None for vanilla mode
        :param max_length: in number of tokens
        :param continuous: whether to loop over the dataset once it is finished. Note that documents from the next
        iteration may be packed together with documents from the previous iteration.
        :param buffer_size: number of characters to keep in buffer (for batch tokenization) (may overshoot)
        If the loader does not prefetch, a large buffer size (I've tried 2^25) will have higher throughput due to larger
        batch tokenization, but will hang during the batch tokenization step. Smaller values are recommended if the
        loader supports prefetching (e.g. DataLoader with num_workers > 0, including 1).

        This is deprecated as of March 2024, and the functionality may have diverged from the
        TokenizedCorpusDataset.
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_length = block_length
        self.max_length = max_length
        if data_formatter is None:
            if dataset_text_field is None:
                raise ValueError("Either dataset_text_field or data_formatting_function must be specified.")
            # self.data_formatter = lambda x: x[dataset_text_field]
            self.dataset_text_field = dataset_text_field
        else:
            self.data_formatter = data_formatter
        self.continuous = continuous
        self.buffer_size = buffer_size
        self.seed = seed
        self.global_shuffling = global_shuffling
        self.local_shuffling = local_shuffling
        self.random_pad_first_block = random_pad_first_block
        self.pad_to_block_boundary = pad_to_block_boundary
        self.transforms = transforms

        self.block_mode = self.block_length is not None

        if self.tokenizer.padding_side == "left":
            raise ValueError("The tokenizer padding side must be right.")

        if self.tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must have an eos token")

        if self.block_mode and self.tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer must have a pad token in block_mode")

        if self.block_mode and max_length % block_length != 0:
            raise ValueError(
                f"Max length ({max_length}) must be divisible by block length ({block_length})."
            )

    def __len__(self):
        # Not the actual number of packed samples
        return len(self.dataset)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        epoch = 0
        dataset = self.dataset
        if self.global_shuffling:
            dataset = self.dataset.shuffle(seed=self.seed)
        if self.local_shuffling:
            local_rng = np.random.RandomState(self.seed)
        iterator = iter(dataset)
        buffer: List[str] = []
        current_buffer_size: int = 0  # in number of characters
        token_buffer: List[int] = []
        data_remaining = True
        while data_remaining:
            # Fill text buffer
            while current_buffer_size < self.buffer_size:
                try:
                    # Get text, add EOS, add random prefix padding to first block
                    # sample = self.data_formatter(next(iterator))
                    sample = next(iterator)[self.dataset_text_field]
                    sample = sample + self.tokenizer.eos_token
                    if self.block_mode and self.random_pad_first_block:
                        pad_length = random.randint(0, self.block_length - 1)
                        sample = self.tokenizer.pad_token * pad_length + sample
                    buffer.append(sample)
                    current_buffer_size += len(sample)
                except StopIteration:
                    if self.continuous:
                        epoch += 1
                        if self.global_shuffling:
                            dataset = self.dataset.shuffle(seed=self.seed + epoch)
                        iterator = iter(dataset)
                    else:
                        data_remaining = False
                        break

            # Tokenize (move data from buffer to token_buffer)
            # TODO: optimize this (tokenization could be handled in a background thread)
            # Tokenize samples into complete blocks (only add right padding until last block boundary)
            tokenized = self.tokenizer(buffer, add_special_tokens=False)["input_ids"]

            if self.block_mode and self.pad_to_block_boundary:
                for sample in tokenized:
                    # Right pad samples until block boundary
                    pad_length = (self.block_length - len(sample) % self.block_length) % self.block_length
                    sample.extend([self.tokenizer.pad_token_id] * pad_length)
            buffer = []
            current_buffer_size = 0

            for sample in tokenized:
                token_buffer.extend(sample)

            # Stack full samples from token buffer
            n_full_samples = len(token_buffer) // self.max_length
            if n_full_samples == 0:
                continue
            full_samples = torch.LongTensor(token_buffer[:n_full_samples * self.max_length])
            full_samples = full_samples.reshape(n_full_samples, self.max_length)
            if self.local_shuffling:
                full_samples = full_samples[local_rng.permutation(range(n_full_samples))]
            token_buffer = token_buffer[n_full_samples * self.max_length:]

            for i, input_ids in enumerate(full_samples):
                if self.block_mode:
                    attention_mask = torch.where(input_ids == self.tokenizer.pad_token_id, 0, 1)
                else:
                    attention_mask = torch.ones_like(input_ids)
                final_sample = {"input_ids": input_ids, "attention_mask": attention_mask}
                if self.transforms is not None:
                    for transform in self.transforms:
                        final_sample = transform(final_sample)
                yield final_sample
