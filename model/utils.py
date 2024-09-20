from typing import Tuple

import torch
from omegaconf import DictConfig
from tokenizers import Tokenizer
from transformers import AutoConfig, GPTNeoXTokenizerFast
from transformers import GPTNeoForCausalLM, GPTNeoXForCausalLM

from model.block_decoder import GPTNeoBlockDecoder, GPTNeoXBlockDecoder
from model.block_decoder.base import BaseBlockDecoder
from model.block_transformer import BlockTransformer
from model.embedder import RobertaEmbedder, RobertaCLSEmbedder, T5Embedder
from model.embedder.base import BaseEmbedder
from model.embedder.lookup import LookupEmbedder, LookupConfig
from model.token_decoder import GPTNeoTokenDecoder, GPTNeoXTokenDecoder, T5TokenDecoder
from model.token_decoder.base import BaseTokenDecoder
from util.tokenizer import load_tokenizer_and_mapper_from_block_config

MODEL_CLS = {
    "gpt-neo": GPTNeoForCausalLM,
    "gpt-neo-x": GPTNeoXForCausalLM,
}

EMBEDDER_CLS = {
    "roberta": RobertaEmbedder,
    "roberta_cls": RobertaCLSEmbedder,
    "t5": T5Embedder,
    "lookup": LookupEmbedder,
}

EMBEDDER_CONFIG_CLS = {
    "lookup": LookupConfig,
}

BLOCK_DECODER_CLS = {
    "gpt-neo": GPTNeoBlockDecoder,
    "gpt-neo-x": GPTNeoXBlockDecoder,
}

TOKEN_DECODER_CLS = {
    "gpt-neo": GPTNeoTokenDecoder,
    "gpt-neo-x": GPTNeoXTokenDecoder,
    "t5": T5TokenDecoder,
}


def get_torch_dtype(cfg: DictConfig):
    if cfg.precision == "bf16":
        return torch.bfloat16
    elif cfg.precision == "fp16":
        return torch.float16
    elif cfg.precision == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Precision {cfg.precision} not supported.")


def load_vanilla_model_from_config(cfg: DictConfig):
    model_cls = MODEL_CLS[cfg.model]

    attn_implementation = cfg.get("attn_implementation", None)
    torch_dtype = get_torch_dtype(cfg)

    if cfg.use_pretrained_weights:
        print("Loading model from pretrained weights...")
        return model_cls.from_pretrained(cfg.model_name_or_path, attn_implementation=attn_implementation,
                                         torch_dtype=torch_dtype)
    else:
        print("Initializing model from scratch...")
        # For some reason, the implementation of transformers version 4.36.2 requires `attn_implementation`
        # to be passed to both functions. The library code looks very ad-hoc at the moment.
        config = AutoConfig.from_pretrained(cfg.model_name_or_path, attn_implementation=attn_implementation,
                                            torch_dtype=torch_dtype)
        if cfg.get("model_config") is not None:
            print("Using custom config for vanilla model...")
            for k, v in cfg.model_config.items():
                if not hasattr(config, k):
                    raise ValueError(f"Config key {k} not found in model config.")
                print(f"   {k}: {v}")
                setattr(config, k, v)
        return model_cls._from_config(config, attn_implementation=attn_implementation, torch_dtype=torch_dtype)


def load_embedder_from_config(cfg: DictConfig, block_decoder: BaseBlockDecoder) -> BaseEmbedder:
    embedder_cls = EMBEDDER_CLS[cfg.embedder.cls]
    embedder_kwargs = {
        "block_length": cfg.block_length,
        "n_embedding_tokens": cfg.embedder.n_embedding_tokens,
        "projection_method": cfg.embedder.get("projection_method", None),
        "projection_hidden_size": block_decoder.hidden_size,
    }
    if cfg.embedder.get("init_projection") is not None:
        embedder_kwargs["init_projection"] = cfg.embedder.init_projection
    if cfg.embedder.get("n_cls_tokens") is not None:
        embedder_kwargs["n_cls_tokens"] = cfg.embedder.n_cls_tokens
    attn_implementation = cfg.embedder.get("attn_implementation", None)
    torch_dtype = get_torch_dtype(cfg)
    if cfg.embedder.use_pretrained_weights:
        print(f"Loading embedder from pretrained weights...")
        return embedder_cls.from_pretrained(cfg.embedder.model_name_or_path, attn_implementation=attn_implementation,
                                            torch_dtype=torch_dtype, **embedder_kwargs)
    else:
        print("Initializing embedder from scratch...")
        # Embedder kwargs that are not used by config are passed to the model. This is the default behavior of
        # `PreTrainedModel.from_pretrained`.
        # Note that attn_implementation and torch_dtype need to be passed to both functions regardless.
        if cfg.embedder.model_name_or_path is None:
            try:
                config_cls = EMBEDDER_CONFIG_CLS[cfg.embedder.cls]
            except KeyError:
                raise ValueError(f"Custom embedder config class for {cfg.embedder.cls} not found. Maybe you forgot "
                                 f"to specify `cfg.embedder.model_name_or_path`?")
            config_dict = {"attn_implementation": attn_implementation,}
            config, unused_kwargs = config_cls.from_dict(config_dict, return_unused_kwargs=True, **embedder_kwargs)
        else:
            config, unused_kwargs = AutoConfig.from_pretrained(cfg.embedder.model_name_or_path,
                                                               attn_implementation=attn_implementation,
                                                               return_unused_kwargs=True, **embedder_kwargs)

        if cfg.embedder.get("config") is not None:
            print("Using custom config for embedder...")
            for k, v in cfg.embedder.config.items():
                if not hasattr(config, k):
                    raise ValueError(f"Config key {k} not found in embedder config.")
                print(f"   {k}: {v}")
                setattr(config, k, v)
        return embedder_cls._from_config(config=config, attn_implementation=attn_implementation,
                                         torch_dtype=torch_dtype, **unused_kwargs)


def load_block_decoder_from_config(cfg: DictConfig) -> BaseBlockDecoder:
    block_decoder_cls = BLOCK_DECODER_CLS[cfg.block_decoder.cls]
    block_decoder_kwargs = {
        "n_embedding_tokens": cfg.embedder.n_embedding_tokens,
        "use_block_decoding_loss": cfg.get("block_decoding_loss", dict()).get("enable", False),
        "block_decoding_loss_type": cfg.get("block_decoding_loss", dict()).get("type", "contrastive"),
    }
    attn_implementation = cfg.block_decoder.get("attn_implementation", None)
    torch_dtype = get_torch_dtype(cfg)
    if cfg.block_decoder.use_pretrained_weights:
        print("Loading block decoder from pretrained weights...")
        return block_decoder_cls.from_pretrained(cfg.block_decoder.model_name_or_path,
                                                 attn_implementation=attn_implementation,
                                                 torch_dtype=torch_dtype, **block_decoder_kwargs)
    else:
        print("Initializing block decoder from scratch...")
        config, unused_kwargs = AutoConfig.from_pretrained(cfg.block_decoder.model_name_or_path,
                                                           attn_implementation=attn_implementation,
                                                           return_unused_kwargs=True, **block_decoder_kwargs)
        if cfg.block_decoder.get("config") is not None:
            print("Using custom config for block decoder...")
            for k, v in cfg.block_decoder.config.items():
                if not hasattr(config, k):
                    raise ValueError(f"Config key {k} not found in block decoder config.")
                print(f"   {k}: {v}")
                setattr(config, k, v)
            if cfg.block_decoder.cls == "gpt-neo" and cfg.block_decoder.config.get("num_layers") is not None:
                config.attention_layers = ["global", "local"] * ((cfg.block_decoder.config.num_layers + 1) // 2)
                config.attention_layers = config.attention_layers[:cfg.block_decoder.config.num_layers]
        return block_decoder_cls._from_config(config=config, attn_implementation=attn_implementation,
                                              torch_dtype=torch_dtype, **unused_kwargs)


def load_token_decoder_from_config(cfg: DictConfig, block_decoder: BaseBlockDecoder) -> BaseTokenDecoder:
    token_decoder_cls = TOKEN_DECODER_CLS[cfg.token_decoder.cls]
    token_decoder_kwargs = {
        "block_length": cfg.block_length,
        "n_embedding_tokens": cfg.embedder.n_embedding_tokens,
        "projection_hidden_size": block_decoder.hidden_size,
        "expansion_method": cfg.token_decoder.expansion_method,
        "expansion_ratio": cfg.token_decoder.get("expansion_ratio", None),
        "decoding_strategy": cfg.token_decoder.decoding_strategy,
    }
    if cfg.token_decoder.get("init_expansion") is not None:
        token_decoder_kwargs["init_expansion"] = cfg.token_decoder.init_expansion
    attn_implementation = cfg.token_decoder.get("attn_implementation", None)
    torch_dtype = get_torch_dtype(cfg)
    if cfg.token_decoder.use_pretrained_weights:
        print("Loading token decoder from pretrained weights...")
        return token_decoder_cls.from_pretrained(cfg.token_decoder.model_name_or_path,
                                                 attn_implementation=attn_implementation,
                                                 torch_dtype=torch_dtype, **token_decoder_kwargs)
    else:
        print("Initializing token decoder from scratch...")
        config, unused_kwargs = AutoConfig.from_pretrained(cfg.token_decoder.model_name_or_path,
                                                           attn_implementation=attn_implementation,
                                                           return_unused_kwargs=True,
                                                           **token_decoder_kwargs)
        if cfg.token_decoder.get("config") is not None:
            print("Using custom config for token decoder...")
            for k, v in cfg.token_decoder.config.items():
                if not hasattr(config, k):
                    raise ValueError(f"Config key {k} not found in token decoder config.")
                print(f"   {k}: {v}")
                setattr(config, k, v)
                if cfg.token_decoder.cls == "gpt-neo" and cfg.token_decoder.config.get("num_layers") is not None:
                    config.attention_layers = ["global", "local"] * ((cfg.token_decoder.config.num_layers + 1) // 2)
                    config.attention_layers = config.attention_layers[:cfg.token_decoder.config.num_layers]

        return token_decoder_cls._from_config(config=config, attn_implementation=attn_implementation,
                                              torch_dtype=torch_dtype, **unused_kwargs)


def load_block_transformer_from_config(cfg: DictConfig) -> Tuple[BlockTransformer, Tokenizer]:
    """
    Does not account for `cfg.load_from_vanilla`.
    """
    tokenizer, token_mapper = load_tokenizer_and_mapper_from_block_config(cfg)
    block_decoder = load_block_decoder_from_config(cfg)
    embedder = load_embedder_from_config(cfg, block_decoder=block_decoder)
    token_decoder = load_token_decoder_from_config(cfg, block_decoder=block_decoder)
    model = BlockTransformer(embedder=embedder,
                             block_decoder=block_decoder,
                             token_decoder=token_decoder,
                             token_mapper=token_mapper,
                             use_token_decoding_loss=cfg.token_decoding_loss.enable,
                             use_block_decoding_loss=cfg.block_decoding_loss.enable,
                             block_decoding_loss_weight=cfg.block_decoding_loss.weight,
                             decoding_strategy=cfg.token_decoder.decoding_strategy, )

    if isinstance(tokenizer, GPTNeoXTokenizerFast):
        # This is done to differentiate eos token and pad token. if not, then we erroneously get the
        # "A decoder-only architecture is being used, but right-padding was detected!" warning because
        # token decoding starts with a placeholder eos
        # Note that token id 1 is not used by the model
        token_decoder.generation_config.pad_token_id = 1

    return model, tokenizer


def load_block_from_vanilla_checkpoint(cfg: DictConfig, block, vanilla):
    print("Loading block from vanilla model...")

    # embedder
    block.embedder.embeddings.load_state_dict(vanilla.gpt_neox.embed_in.state_dict())
    if cfg.load_from_vanilla.get("initialize_mean_embedder_projection"):
        # NOTE - only use for lookup embedder with projection_layer and same hidden size as block decoder
        print("Initializing mean embedder projection...")
        projection_layer = block.embedder.projection_layer
        mean_weight = 1 / cfg.block_length
        weight_shape = projection_layer.weight.shape

        diag_matrix = torch.eye(weight_shape[0]) * mean_weight
        diag_matrix = diag_matrix.unsqueeze(-1).repeat(1, 1, weight_shape[-1])

        # Assign the diagonal matrices to the weights
        with torch.no_grad():
            projection_layer.weight.copy_(diag_matrix)

        # Fill bias with zeros
        with torch.no_grad():
            projection_layer.bias.zero_()

    # check number of layers
    vanilla_layers = len(vanilla.gpt_neox.layers)
    block_layers = len(block.block_decoder.gpt_neox.layers)
    token_layers = len(block.token_decoder.gpt_neox.layers)
    print(f"Vanilla layers:       {vanilla_layers}")
    print(f"Block decoder layers: {block_layers}")
    print(f"Token decoder layers: {token_layers}")
    if cfg.load_from_vanilla.method == "skip":
        if block_layers * 2 != vanilla_layers:
            raise ValueError(f"Block decoder has {block_layers} layers, but vanilla model has {vanilla_layers} layers.")
        if token_layers * 2 != vanilla_layers:
            raise ValueError(f"Token decoder has {token_layers} layers, but vanilla model has {vanilla_layers} layers.")
    elif cfg.load_from_vanilla.method == "partition":
        if block_layers + token_layers != vanilla_layers:
            raise ValueError(f"Block decoder has {block_layers} layers and token decoder has {token_layers} layers, "
                             f"but vanilla model has {vanilla_layers} layers.")
    elif cfg.load_from_vanilla.method == "duplicate":
        if block_layers != vanilla_layers:
            raise ValueError(f"Block decoder has {block_layers} layers, but vanilla model has {vanilla_layers} layers.")
        if token_layers != block_layers:
            raise ValueError(f"Token decoder has {token_layers} layers, but block decoder has {block_layers} layers.")
    else:
        raise ValueError(f"Invalid method {cfg.load_from_vanilla.method}")

    # block decoder
    if cfg.load_from_vanilla.method == "skip":
        for i in range(int(block_layers)):
            block.block_decoder.gpt_neox.layers[i].load_state_dict(vanilla.gpt_neox.layers[2 * i].state_dict())
    elif cfg.load_from_vanilla.method == "partition":
        for i in range(int(block_layers)):
            block.block_decoder.gpt_neox.layers[i].load_state_dict(vanilla.gpt_neox.layers[i].state_dict())
    elif cfg.load_from_vanilla.method == "duplicate":
        for i in range(block_layers):
            block.block_decoder.gpt_neox.layers[i].load_state_dict(vanilla.gpt_neox.layers[i].state_dict())
    else:
        raise ValueError(f"Invalid method {cfg.load_from_vanilla.method}")

    # token decoder
    if cfg.load_from_vanilla.get("compute_token_decoder_embeddings"):
        print("Computing token decoder embeddings with block decoder...")
        inputs_embeds = block.embedder.embeddings.weight.data.unsqueeze(1)
        inputs_embeds = inputs_embeds.cuda()
        block.cuda()
        with torch.no_grad():
            output = block.block_decoder.gpt_neox(inputs_embeds=inputs_embeds, return_dict=False)
        hidden_states = output[0].cpu()
        block.cpu()
        print(f"Inputs embeds shape: {inputs_embeds.shape}")
        print(f"Hidden states shape: {hidden_states.shape}")
        print(f"Inputs embeds mean: {inputs_embeds.mean()}")
        print(f"Hidden states mean: {hidden_states.mean()}")
        print(f"Inputs embeds std: {inputs_embeds.std()}")
        print(f"Hidden states std: {hidden_states.std()}")
        assert len(hidden_states.shape) == 3
        assert hidden_states.shape == inputs_embeds.shape
        with torch.no_grad():
            block.token_decoder.gpt_neox.embed_in.weight.copy_(hidden_states.squeeze(1))
    else:
        block.token_decoder.gpt_neox.embed_in.load_state_dict(vanilla.gpt_neox.embed_in.state_dict())
    if cfg.load_from_vanilla.method == "skip":
        for i in range(token_layers):
            block.token_decoder.gpt_neox.layers[i].load_state_dict(vanilla.gpt_neox.layers[2 * i].state_dict())
    elif cfg.load_from_vanilla.method == "partition":
        for i in range(token_layers):
            block.token_decoder.gpt_neox.layers[i].load_state_dict(vanilla.gpt_neox.layers[i + block_layers].state_dict())
    elif cfg.load_from_vanilla.method == "duplicate":
        for i in range(token_layers):
            block.token_decoder.gpt_neox.layers[i].load_state_dict(vanilla.gpt_neox.layers[i].state_dict())
    else:
        raise ValueError(f"Invalid method {cfg.load_from_vanilla.method}")

    # initialize token decoder expansion_layer to identity
    if cfg.load_from_vanilla.get("initialize_identity_expansion_layer"):
        # NOTE - only use for prefix token decoder with expansion_layer
        expansion_layer = block.token_decoder.expansion_layer
        expansion_ratio = block.token_decoder.expansion_ratio
        diag_matrix = torch.eye(expansion_layer.weight.shape[1]) * 1
        diag_matrix = diag_matrix.unsqueeze(-1).repeat(expansion_ratio, 1, 1)
        # Assign the diagonal matrices to the weights
        with torch.no_grad():
            expansion_layer.weight.copy_(diag_matrix)
        # Fill bias with zeros
        with torch.no_grad():
            expansion_layer.bias.zero_()

    # misc
    block.token_decoder.gpt_neox.final_layer_norm.load_state_dict(vanilla.gpt_neox.final_layer_norm.state_dict())
    block.token_decoder.embed_out.load_state_dict(vanilla.embed_out.state_dict())

    return block
