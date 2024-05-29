import os

import numpy as np
from tqdm import trange

from custom_dataset.tokenized_corpus import TokenizedCorpus

SAMPLE_START_INDEX = 0
SAMPLE_END_INDEX = 134320000
STEP = 20_000


class T5PileTokenizedCorpus(TokenizedCorpus):
    def __init__(self, t5_pile_idxmaps_path: str):
        document_lengths = self._get_document_lengths(t5_pile_idxmaps_path)
        document_indices = self._get_document_indicies(t5_pile_idxmaps_path, document_lengths)
        total_tokens = document_lengths.sum().item()

        token_data = self._get_token_data(t5_pile_idxmaps_path, document_indices, total_tokens)
        super().__init__(token_data, document_lengths, document_indices)

    def _get_token_data(self, path, document_indices, total_tokens):
        if os.path.exists(os.path.join(path, "token_data.bin")):
            print("Loading token_data.bin")
            return np.memmap(os.path.join(path, "token_data.bin"), dtype="uint16", mode="r")
        else:
            start, end = SAMPLE_START_INDEX, SAMPLE_END_INDEX
            step = STEP

            # TODO: hardcode the length of the token_data array (takes 16 min)
            token_data = np.memmap(os.path.join(path, "token_data.bin"), dtype="uint16", mode="w+",
                                   shape=(total_tokens,))

            idx = 0
            for batch_start in trange(start, end, step):
                _path = os.path.join(path, f"data_{batch_start:010d}_{batch_start + step:010d}.npy")
                data = np.load(_path)
                token_data[idx:idx + data.shape[0]] = data[:]
                idx += data.shape[0]
                assert document_indices[min(batch_start + step - 1, document_indices.shape[0] - 1)]

            return token_data

    def _get_document_lengths(self, path):
        if os.path.exists(os.path.join(path, "document_lengths.idx")):
            print("Loading document_lengths.idx")
            return np.memmap(os.path.join(path, "document_lengths.idx"), dtype="uint32", mode="r")
        else:
            start, end = SAMPLE_START_INDEX, SAMPLE_END_INDEX
            step = STEP

            lengths_lst = []
            for batch_start in trange(start, end, step):
                _path = os.path.join(path, f"lengths_{batch_start:010d}_{batch_start + step:010d}.npy")
                lengths_lst.append(np.load(_path))

            lengths_ndarray = np.concatenate(lengths_lst)

            document_lengths = np.memmap(os.path.join(path, "document_lengths.idx"), dtype="uint32", mode="w+",
                                         shape=lengths_ndarray.shape)
            document_lengths[:] = lengths_ndarray[:]
            return document_lengths

    def _get_document_indicies(self, path, document_lengths):
        if os.path.exists(os.path.join(path, "document_indices.idx")):
            print("Loading document_indices.idx")
            return np.memmap(os.path.join(path, "document_indices.idx"), dtype="uint64", mode="r")
        else:
            indices_ndarray = np.cumsum(document_lengths)
            indices_ndarray = np.concatenate([[0], indices_ndarray[:-1]])

            document_indices = np.memmap(os.path.join(path, "document_indices.idx"), dtype="uint64", mode="w+",
                                         shape=indices_ndarray.shape)
            document_indices[:] = indices_ndarray[:]
            return document_indices
