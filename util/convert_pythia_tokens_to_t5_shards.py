"""
Usage:
python convert_pythia_tokens_to_t5_shards.py 0 20000000
python convert_pythia_tokens_to_t5_shards.py 20000000 40000000
python convert_pythia_tokens_to_t5_shards.py 40000000 60000000
python convert_pythia_tokens_to_t5_shards.py 60000000 80000000
python convert_pythia_tokens_to_t5_shards.py 80000000 100000000
python convert_pythia_tokens_to_t5_shards.py 100000000 120000000
python convert_pythia_tokens_to_t5_shards.py 120000000 140000000

This script converts the Pythia tokenized corpus to T5 tokenized corpus in shards.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAYON_RS_NUM_CPUS"] = "12"  # number of CPUs to use for tokenizer batch encoding
import sys

import numpy as np
from tqdm import trange

from custom_dataset.pythia_pile_tokenized_corpus import PythiaPileTokenizedCorpus
from util.tokenizer import TOKENIZERS

PYTHIA_PILE_IDXMAPS_PATH = "/mnt/data2/pythia/pythia_pile_idxmaps"

corpus = PythiaPileTokenizedCorpus(PYTHIA_PILE_IDXMAPS_PATH)

tk_t5 = TOKENIZERS["t5"]
tk_pythia = TOKENIZERS["pythia"]


def get_t5_documents(corpus, start, end):
    start = int(max(0, start))
    end = int(min(end, len(corpus)))
    documents = []
    for i in range(start, end):
        documents.append(corpus[i].tolist())
    if not documents:
        return []
    strings = tk_pythia.batch_decode(documents)
    documents = tk_t5(strings, add_special_tokens=False, return_token_type_ids=False, return_attention_mask=False)[
        "input_ids"]
    arrays = []
    for d in documents:
        arrays.append(np.array(d, dtype=np.uint16))
    return arrays


def main():

    start = int(sys.argv[1])
    end = int(sys.argv[2])
    step = 20_000
    end = min(end, len(corpus))
    print(f"start={start}, end={end}, step={step}")
    for batch_start in trange(start, end, step):
        batch_end = batch_start + step
        target_path = f"/mnt/data2/t5_pile_shards/lengths_{batch_start:010d}_{batch_end:010d}.npy"
        if os.path.exists(target_path):
            continue
        arrays = get_t5_documents(corpus, batch_start, batch_end)
        lengths = np.array([array.shape[0] for array in arrays], dtype=np.int32)
        data = np.concatenate(arrays)
        os.makedirs("t5", exist_ok=True)
        np.save(f"/mnt/data2/t5_pile_shards/lengths_{batch_start:010d}_{batch_end:010d}.npy", lengths)
        np.save(f"/mnt/data2/t5_pile_shards/data_{batch_start:010d}_{batch_end:010d}.npy", data)


if __name__ == "__main__":
    main()
