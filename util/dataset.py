import os
import warnings

from datasets import load_dataset

from custom_dataset.language_modeling_dataset import LanguageModelingDataset
from custom_dataset.tokenized_corpus import TokenizedCorpus, TokenizedCorpusDataset
from util.data_preprocessing import AddLabels, SplitBlocks, RemoveIndex

HF_DATASETS = {
    # arguments for the `load_dataset` function
    "redpajama_book": {"path": "togethercomputer/RedPajama-Data-1T", "name": "book"},
    "redpajama_wikipedia": {"path": "togethercomputer/RedPajama-Data-1T", "name": "wikipedia"},
    "pile_hf": {"path": "monology/pile-uncopyrighted", "cache_dir": "/mnt/data2/huggingface/datasets"},
}

TOKENIZED_DATASETS = {
    "pythia_pile": "pythia",  # tokenizer used for pre-tokenization
    "t5_pile": "t5",
}


def load_train_dataset_from_config(cfg, tokenizer):
    if cfg.dataset in HF_DATASETS:
        dataset_type = "hf"
        if "redpajama" in cfg.dataset:
            os.environ["RED_PAJAMA_DATA_DIR"] = cfg.RED_PAJAMA_DATA_DIR
        train_dataset = load_dataset(**HF_DATASETS[cfg.dataset])
        train_dataset = train_dataset["train"]

    elif cfg.dataset in TOKENIZED_DATASETS:
        dataset_type = "token"
        # check if tokenizer used by dataset is compatible with the one specified in config
        if "tokenizer" in cfg:
            tokenizer_used = TOKENIZED_DATASETS[cfg.dataset]
            if isinstance(cfg.tokenizer, str):
                if cfg.tokenizer != tokenizer_used:
                    raise ValueError(f"Tokenizer {cfg.tokenizer} is not compatible with dataset {cfg.dataset}")
            else:
                if "embedder" in cfg.tokenizer and cfg.tokenizer.embedder != tokenizer_used:
                    message = f"Tokenizer {cfg.tokenizer.embedder} is not compatible with dataset {cfg.dataset}"
                    raise ValueError(message)
                if "token_decoder" in cfg.tokenizer and cfg.tokenizer.token_decoder != tokenizer_used:
                    message = f"Tokenizer {cfg.tokenizer.token_decoder} is not compatible with dataset {cfg.dataset}"
                    raise ValueError(message)

        # load corpus
        if cfg.dataset == "pythia_pile":
            corpus = _load_pythia_pile_corpus(cfg)
        elif cfg.dataset == "t5_pile":
            corpus = _load_t5_pile_corpus(cfg)
        else:
            raise ValueError(f"Unknown dataset: {cfg.dataset}")

    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    if cfg.block_mode:
        block_length = cfg.block_length
        random_pad_first_block = cfg.random_pad_first_block
        pad_to_block_boundary = cfg.pad_to_block_boundary

        if not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
            print("Setting pad token to eos token")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        transforms = [
            SplitBlocks(cfg.block_split.distribution, cfg.block_split.get("distribution_kwargs", {}),
                        tokenizer.pad_token_id),
            AddLabels(),
            RemoveIndex(),
        ]
    else:
        block_length = None
        random_pad_first_block = False
        pad_to_block_boundary = False
        transforms = [
            AddLabels(),
            RemoveIndex(),
        ]

    if dataset_type == "hf":
        if cfg.dataloader_num_workers > 1:
            raise ValueError("Using multiple workers with LanguageModelingDataset will lead to each worker getting a"
                             "replica of the same data. This is the default behavior of IterableDataset.")
        return LanguageModelingDataset(train_dataset, tokenizer, block_length=block_length,
                                       max_length=cfg.max_length, dataset_text_field="text",
                                       random_pad_first_block=random_pad_first_block,
                                       pad_to_block_boundary=pad_to_block_boundary, transforms=transforms)
    elif dataset_type == "token":
        if cfg.dataloader_num_workers <= 1:
            warnings.warn(f"Using cfg.dataloader_num_workers={cfg.dataloader_num_workers} with TokenizedCorpusDataset."
                          f"You may want to increase this number to speed up data loading.")
        return TokenizedCorpusDataset(corpus, length=cfg.max_length, eos_token=tokenizer.eos_token_id,
                                      transforms=transforms,
                                      pad_token=tokenizer.pad_token_id if cfg.block_mode else None,
                                      block_length=block_length, random_pad_first_block=random_pad_first_block,
                                      pad_to_block_boundary=pad_to_block_boundary)
    else:
        raise AssertionError()


def _load_pythia_pile_corpus(cfg) -> TokenizedCorpus:
    from custom_dataset.pythia_pile_tokenized_corpus import PythiaPileTokenizedCorpus
    corpus = PythiaPileTokenizedCorpus(cfg.pythia_pile_idxmaps_path)
    return corpus


def _load_t5_pile_corpus(cfg) -> TokenizedCorpus:
    from custom_dataset.t5_pile_tokenized_corpus import T5PileTokenizedCorpus
    corpus = T5PileTokenizedCorpus(cfg.t5_pile_idxmaps_path)
    return corpus
