import math
from typing import Set

import torch


class SplitFixedBlocks:
    """
    Previously `ToBlocks`. Deprecated and replaced by `SplitBlocks` with `distribution="fixed"` as of 2024-03-01.
    """

    def __init__(self, block_length: int):
        self.block_length = block_length

    def __call__(self, sample: dict):
        sample["input_ids"] = sample["input_ids"].reshape(-1, self.block_length)
        sample["attention_mask"] = sample["attention_mask"].reshape(-1, self.block_length)
        if "label" in sample:
            sample["label"] = sample["label"].reshape(-1, self.block_length)
        # full-EOS blocks will have a block_attention_mask of 0
        sample["block_attention_mask"] = sample["attention_mask"].any(dim=1).long()
        return sample


class AddLabels:
    def __call__(self, sample: dict):
        attention_mask = sample["attention_mask"]
        labels = sample["input_ids"].clone()
        labels[attention_mask == 0] = -100
        sample.update({"labels": labels})
        return sample


class RemoveIndex:
    """
    Note that we include `index` in the sample to ensure reproducability of randomness in preprocessing steps,
    e.g., `SplitBlocks()` with random length distribution. Make sure to remove `index` before feeding the sample
    to the model.
    """
    def __call__(self, sample: dict):
        sample.pop("index", None)
        return sample


DISTRIBUTIONS = {}  # populated below


class SplitBlocks:
    def __init__(self, distribution: str, distribution_kwargs: dict, pad_token_id: int):
        try:
            self.distribution: BlockLengthDistribution = DISTRIBUTIONS[distribution](**distribution_kwargs)
        except KeyError:
            raise ValueError(f"Unknown block length distribution: {distribution}")
        self.pad_token_id = pad_token_id

    def __call__(self, sample: dict):
        total_length = sample["input_ids"].shape[-1]
        block_lengths = self.distribution.get_lengths(total_length, sample.get("index"))

        n_blocks = len(block_lengths)
        input_ids = sample["input_ids"]
        batched = len(input_ids.shape) > 1
        batch_size = input_ids.shape[0] if batched else 1
        padded_size = (batch_size, n_blocks, self.distribution.max)

        split = {}
        padded = {}
        split["input_ids"] = sample["input_ids"].split_with_sizes(block_lengths.tolist(), dim=-1)
        padded["input_ids"] = sample["input_ids"].new_full(padded_size, self.pad_token_id)
        split["attention_mask"] = sample["attention_mask"].split_with_sizes(block_lengths.tolist(), dim=-1)
        padded["attention_mask"] = sample["attention_mask"].new_zeros(padded_size)

        if "label" in sample:
            label = sample["label"]
            split["label"] = label.split_with_sizes(block_lengths.tolist(), dim=-1)
            padded["label"] = sample["label"].new_full(padded_size, -100)

        for i, block_length in enumerate(block_lengths):
            padded["input_ids"][:, i, :block_length] = split["input_ids"][i]
            padded["attention_mask"][:, i, :block_length] = split["attention_mask"][i]
            if "label" in sample:
                padded["label"][:, i, :block_length] = split["label"][i]

        if not batched:  # maintain the same dimensionality as the input (batched or not)
            padded = {k: v.squeeze(0) for k, v in padded.items()}

        sample.update(padded)
        # full-EOS blocks will have a block_attention_mask of 0
        sample["block_attention_mask"] = sample["attention_mask"].any(dim=1).long()

        return sample


class BlockLengthDistribution:
    def __init__(self, pmf: torch.Tensor, seed: int = 42, verbose=True):
        if pmf[0] != 0:
            raise ValueError("The `pmf[0]` should be 0 since block length 0 is not allowed.")

        self.pmf = pmf
        self.seed = seed

        self.mean = pmf.dot(torch.arange(len(pmf), dtype=torch.float64)).item()
        self.domain: Set[int] = set()
        for i, p in enumerate(pmf):
            if p != 0:
                self.domain.add(i)
        self.max = max(self.domain)

        self.generator = torch.Generator(device="cpu")

        if verbose:
            name = self.__class__.__name__
            print(f"Using random block length sampled from {name} with mean={self.mean:.2f}:")
            print(f" Probability mass ".center(80, "-"))
            for i, p in enumerate(pmf):
                if p != 0:
                    bar = "=" * int(p * 60)
                    space = " " * (60 - int(p * 60))
                    print(f"{i:>2d}: {p * 100:6.2f}% [{bar}{space}]")

            print("-" * 80)

    def get_lengths(self, total_length: int, sample_index: int = None):
        """
        :param total_length:
        :param sample_index: optional index of the sample in the dataset used for stable sampling, i.e., to ensure
        reproducibility.
        """
        # all random operations must use this generator to ensure reproducibility
        seed = self.seed + sample_index if sample_index else self.seed
        generator = self.generator.manual_seed(seed % (2 ** 32 - 1))  # wrap to 32-bit

        n_blocks = math.ceil(total_length / self.mean)
        lengths = torch.multinomial(self.pmf, n_blocks, replacement=True, generator=generator)

        tries = 0
        current_total = lengths.sum()
        # adjust lengths to strictly match total_length
        while current_total != total_length:
            # select random block and add or remove 1 if possible (within domain)
            i = torch.randint(0, n_blocks, (1,), generator=generator).item()
            if current_total < total_length and int(lengths[i] + 1) in self.domain:
                lengths[i] += 1
                current_total += 1
            elif current_total > total_length and int(lengths[i] - 1) in self.domain:
                lengths[i] -= 1
                current_total -= 1

            tries += 1
            if tries > 500:
                raise ValueError("Taking too long to adjust block lengths. Something might be wrong or the pmf "
                                 "or other parameters may be too extreme.")

        return lengths


class FixedDistribution(BlockLengthDistribution):
    """
    Dirac delta distribution for fixed block length.
    """

    def __init__(self, length: int = 4):
        self.length = length
        pmf = torch.zeros((length + 1,), dtype=torch.float64)
        pmf[length] = 1
        super().__init__(pmf)

    def get_lengths(self, total_length: int, sample_index: int = None):
        """
        Custom implementation to avoid sampling.
        """
        if total_length % self.length != 0:
            # Support for this case is possible, but not implemented yet
            raise ValueError(f"Total length {total_length} is not divisible by the fixed block length {self.length}")
        n_blocks = int(total_length // self.length)
        return torch.full((n_blocks,), self.length, dtype=torch.long)


class UniformDistribution(BlockLengthDistribution):
    def __init__(self, mean: int = 4, radius: int = None):
        if radius is None:
            radius = mean - 1  # the domain of the uniform distribution is [1, mean_block_length * 2 - 1]
        if mean - radius < 1:
            raise ValueError("The width is too large for the mean block length.")

        start = mean - radius  # inclusive
        end = mean + radius + 1  # exclusive
        width = end - start
        pmf = torch.zeros((end,), dtype=torch.float64)
        pmf[start:end] = 1 / width
        super().__init__(pmf)


DISTRIBUTIONS.update({
    "fixed": FixedDistribution,
    "uniform": UniformDistribution,
})
