import os

import numpy as np

from util.mmap_dataset import MMapIndexedDataset
from custom_dataset.tokenized_corpus import TokenizedCorpus


class PythiaPileTokenizedCorpus(TokenizedCorpus):
    def __init__(self, pythia_pile_idxmaps_path: str):
        self.path = os.path.join(pythia_pile_idxmaps_path, "pile_0.87_deduped_text_document")
        self.dataset = MMapIndexedDataset(self.path, skip_warmup=True)

        token_data = np.memmap(self.path + ".bin", dtype="uint16", mode="r", order="C")
        document_lengths = self.dataset._index._sizes
        document_indices = self.dataset._index._pointers // 2
        super().__init__(token_data, document_lengths, document_indices)
