import numpy as np
import torch


class TokenizedCorpus:
    def __init__(self, token_data: np.ndarray, document_lengths: np.ndarray, document_indices: np.ndarray):
        """
        :param token_data: uint16 recommended
        :param document_lengths: uint16 recommended
        """
        self.token_data = token_data
        self.document_lengths = document_lengths
        self.document_indices = document_indices
        self.total_length = document_indices[-1] + document_lengths[-1]

    def __len__(self):
        return self.document_indices.shape[0]

    def __getitem__(self, i: int):
        return self.token_data[self.document_indices[i]:self.document_indices[i] + self.document_lengths[i]].copy()


class TokenizedCorpusDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_corpus: TokenizedCorpus, length: int, eos_token: int,
                 transforms: list = None, pad_token: int = None, block_length: int = None,
                 random_pad_first_block: bool = False, pad_to_block_boundary: bool = False, seed: int = 42):
        """
        :param tokenized_corpus:
        :param length:
        :param eos_token: token to place after documents
        iteration may be packed together with documents from the previous iteration.
        :param transforms: to apply to each sample
        :param block_length:
        :param pad_token: token used for block padding (also between documents) (for block mode)
        :param random_pad_first_block:
        :param pad_to_block_boundary:
        :param seed:

        Note that block_length, random_pad_first_block, and pad_to_block_boundary are only relevant when using fixed
        block length (`cfg.block_split.distribution == "fixed"`). As of March 2024, we decided to forgo the use of
        block-specific padding, originally intended to make sure that two documents do not get packed together in the
        same block. Preliminary experiments showed that this did not have a significant impact, and is
        too complicated to implement for variable block lengths.
        """
        self.tokenized_corpus = tokenized_corpus
        self.length = length
        self.eos_token = eos_token
        self.transforms = transforms
        self.pad_token = pad_token
        self.block_length = block_length
        self.random_pad_first_block = random_pad_first_block
        self.pad_to_block_boundary = pad_to_block_boundary
        self.seed = seed

        if block_length:
            self.block_mode = True
            if block_length >= 32768:
                raise NotImplementedError("block_length above int16 max not supported.")
            if self.length % self.block_length != 0:
                raise ValueError(
                    f"Max length ({self.length}) must be divisible by block length ({self.block_length})."
                )
        else:
            self.block_mode = False  # vanilla mode

        if (self.random_pad_first_block or self.pad_to_block_boundary) and pad_token is None:
            raise ValueError("pad_token must be specified if random_pad_first_block or pad_to_block_boundary is True.")

        self._prepare_indices()

    def __len__(self):
        return self.padded_total_length // self.length

    def __getitem__(self, idx):
        input_ids = torch.full([self.length], -1, dtype=torch.long)
        attention_mask = torch.full([self.length], -1, dtype=torch.long)

        corpus_index = idx * self.length % self.padded_total_length  # current index in the padded corpus
        sample_length = 0  # current length of the sample
        document_index = np.searchsorted(self.padded_document_indices, corpus_index, side="right") - 1
        assert 0 <= document_index < self.tokenized_corpus.document_indices.shape[-1]

        # fill sample
        while sample_length < self.length:
            # current_index_in_document refers to the current location with respect to the start of the actual document
            # not the start of the padded document
            current_index_in_document = corpus_index - self.padded_document_indices[document_index] - \
                                        self.left_pad_lengths[document_index]
            sample_remaining = self.length - sample_length

            if current_index_in_document < 0:
                # insert left padding
                assert self.block_mode  # should not be in padding territory in vanilla mode
                # sample:                        [P P P P P P P P P P P P P P P P P P P P P P P ...
                # current document (padded): [P P P P D D D D D D D D D D D D D D D D D D E P P]
                # current location                ^
                # e.g., in the above example, current_index_in_document = -2
                left_pad_remaining = - current_index_in_document
                pad_length = min(left_pad_remaining, sample_remaining)
                input_ids[sample_length:sample_length + pad_length] = self.pad_token
                attention_mask[sample_length:sample_length + pad_length] = 0
                corpus_index += pad_length
                sample_length += pad_length
            elif current_index_in_document < self.tokenized_corpus.document_lengths[document_index]:
                # insert document
                # sample:                        [P P - - - - - - - - - - - - - - - - - - - - - ...
                # current document (padded): [P P P P D D D D D D D D D D D D D D D D D D E P P]
                # current index:                      ^
                document_remaining = self.tokenized_corpus.document_lengths[document_index] - current_index_in_document
                sample_remaining = self.length - sample_length
                copy_length = min(document_remaining, sample_remaining)

                copy_start_index = int(self.tokenized_corpus.document_indices[document_index] + current_index_in_document)
                data = self.tokenized_corpus.token_data[copy_start_index:copy_start_index + copy_length]
                input_ids[sample_length:sample_length + copy_length] = torch.from_numpy(np.array(data, dtype=np.int64))
                attention_mask[sample_length:sample_length + copy_length] = 1

                corpus_index += copy_length
                sample_length += copy_length
                current_index_in_document += copy_length
            elif current_index_in_document == self.tokenized_corpus.document_lengths[document_index]:
                # insert eos
                # sample:                        [P P D D D D D D D D D D D D D D D D D D - - - ...
                # current document (padded): [P P P P D D D D D D D D D D D D D D D D D D E P P]
                # current index:                                                          ^
                input_ids[sample_length] = self.eos_token
                attention_mask[sample_length] = 1
                sample_length += 1
                corpus_index += 1
                if not self.block_mode or not self.pad_to_block_boundary:
                    # no right padding, end of padded document
                    document_index += 1
            elif current_index_in_document > self.tokenized_corpus.document_lengths[document_index]:
                # insert right padding
                assert self.block_mode and self.pad_to_block_boundary  # should not be in padding territory in vanilla mode
                # sample:                        [P P D D D D D D D D D D D D D D D D D D D P P ...
                # current document (padded): [P P P P D D D D D D D D D D D D D D D D D D D P P]
                # current index:                                                            ^
                right_pad_used = current_index_in_document - self.tokenized_corpus.document_lengths[document_index]
                right_pad_remaining = self.right_pad_lengths[document_index] - right_pad_used
                pad_length = min(right_pad_remaining, sample_remaining)
                input_ids[sample_length:sample_length + pad_length] = self.pad_token
                attention_mask[sample_length:sample_length + pad_length] = 0
                corpus_index += pad_length
                sample_length += pad_length
                if pad_length == right_pad_remaining:
                    # end of padded document
                    document_index += 1

        assert (input_ids == -1).sum() == 0
        assert (attention_mask == -1).sum() == 0
        if not self.block_mode:
            assert (attention_mask == 1).all()
        sample = {
            "index": idx,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self.transforms:
            for transform in self.transforms:
                sample = transform(sample)
        return sample

    def _prepare_indices(self):        
        # Takes about 3 seconds for deduped The Pile (~134M documents)
        if self.block_mode:
            if self.random_pad_first_block:
                pad_rng = np.random.RandomState(self.seed)
                self.left_pad_lengths = pad_rng.randint(self.block_length,
                                                        size=self.tokenized_corpus.document_indices.shape, dtype=np.int16)
                # +1 is added for the eos token
                left_padded_document_lengths = self.tokenized_corpus.document_lengths + self.left_pad_lengths + 1
            else:
                self.left_pad_lengths = np.zeros(self.tokenized_corpus.document_indices.shape, dtype=np.int16)
                left_padded_document_lengths = self.tokenized_corpus.document_lengths + 1
                
            if self.pad_to_block_boundary:
                self.right_pad_lengths = self.block_length - left_padded_document_lengths % self.block_length
                self.right_pad_lengths[self.right_pad_lengths == self.block_length] = 0
                self.right_pad_lengths = self.right_pad_lengths.astype(np.int16)
            else:
                self.right_pad_lengths = np.zeros(self.tokenized_corpus.document_indices.shape, dtype=np.int16)

            self.padded_document_lengths = left_padded_document_lengths + self.right_pad_lengths
        else:
            self.left_pad_lengths = np.zeros(self.tokenized_corpus.document_indices.shape, dtype=np.int16)
            self.right_pad_lengths = np.zeros(self.tokenized_corpus.document_indices.shape, dtype=np.int16)
            self.padded_document_lengths = self.tokenized_corpus.document_lengths + 1

        cumsum = np.cumsum(np.concatenate([[0], self.padded_document_lengths]), dtype=np.int64)
        self.padded_total_length = cumsum[-1]
        # Start of each document in the padded corpus, i.e., the start index of the left padding
        # The start index of the actual document is this + self.left_pad_lengths
        self.padded_document_indices = cumsum[:-1]
