from transformers import AutoTokenizer

from util.token_mapper import TokenMapper

TOKENIZERS = {
    "roberta": AutoTokenizer.from_pretrained("roberta-base"),
    "t5": AutoTokenizer.from_pretrained("t5-base"),
    "gpt2": AutoTokenizer.from_pretrained("gpt2"),
    "gpt-neo": AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B"),
    "pythia": AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m-deduped",
        revision="step3000",
        cache_dir="./pythia-70m-deduped/step3000",
    )
}


def load_tokenizer_and_mapper_from_block_config(cfg):
    """
    :return: tokenizer (for embedder), token_mapper (from embedder to token_decoder)
    """
    if cfg.tokenizer.embedder == cfg.tokenizer.token_decoder:
        tokenizer = TOKENIZERS[cfg.tokenizer.embedder]
        return tokenizer, None
    else:
        embedder_tokenizer = TOKENIZERS[cfg.tokenizer.embedder]
        token_decoder_tokenizer = TOKENIZERS[cfg.tokenizer.token_decoder]
        token_mapper = TokenMapper(embedder_tokenizer=embedder_tokenizer,
                                   token_decoder_tokenizer=token_decoder_tokenizer)
        return embedder_tokenizer, token_mapper


def load_tokenizer_from_vanilla_config(cfg):
    return TOKENIZERS[cfg.tokenizer]
