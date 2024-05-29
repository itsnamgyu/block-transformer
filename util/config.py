import os
import warnings

import torch
from omegaconf import DictConfig, open_dict, OmegaConf

from paths import PROJECT_ROOT


def load_config(config_path: str) -> DictConfig:
    cfg = OmegaConf.load(config_path)
    preprocess_config(cfg)
    return cfg


def preprocess_config(cfg: DictConfig, check_mode: str = None):
    print(" Preprocess Config ".center(80, "-"))

    # Determine block_split and block_mode
    with open_dict(cfg):
        # Prior to March 2024, we had set `block_length` to specify fixed-length block split. Now, we use `block_split`.
        if cfg.get("block_length") is not None and cfg.get("block_split") is None:
            print(f"Setting `block_split` to fixed distribution with length {cfg.block_length}")
            cfg.block_split = {
                "distribution": "fixed",
                "distribution_kwargs": {"length": cfg.block_length}
            }
        cfg.block_mode = cfg.get("block_split") is not None

    if check_mode == "block" and not cfg.block_mode:
        raise ValueError("Config does not include block_length but block mode is expected. Maybe you meant to call"
                         "`pretrain_vanilla_transformer.py`?")
    if check_mode == "vanilla" and cfg.block_mode:
        raise ValueError("Config includes block_length but vanilla mode is expected. Maybe you meant to call"
                         "`pretrain_block_transformer.py`?")

    if cfg.precision not in ["fp32", "fp16", "bf16"]:
        raise NotImplementedError(f"Precision {cfg.precision} is not implemented yet")
    elif cfg.precision == "fp16":
        warnings.warn("Are you sure you want to use fp16? We use bf16 by default.")

    # Automatically determine batch size and gradient accumulation steps
    n_gpus = torch.cuda.device_count()
    n_gpus = n_gpus or 1
    if cfg.get("total_batch_size") is not None:
        print("Automatically determining batch size based on `total_batch_size`")
        # Automatically set per_device_train_batch_size or
        # (if per_device_train_batch_size is already set) gradient_accumulation_steps
        if cfg.get("gradient_accumulation_steps") is not None:
            raise ValueError("Cannot specify both total_batch_size and gradient_accumulation_steps")
        if cfg.get("per_device_train_batch_size") is not None:
            cfg.gradient_accumulation_steps = round(cfg.total_batch_size / (cfg.per_device_train_batch_size * n_gpus))
            print(f"total_batch_size              : {cfg.total_batch_size} (given)")
            print(f"torch.cuda.device_count()     : {n_gpus}")
            print(f"per_device_train_batch_size   : {cfg.per_device_train_batch_size} (given)")
            print(f"gradient_accumulation_steps   : {cfg.gradient_accumulation_steps} (computed)")
            actual_total = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * n_gpus
            print(f"actual total batch size       : {actual_total}")
        else:
            cfg.per_device_train_batch_size = round(cfg.total_batch_size / n_gpus)
            print(f"total_batch_size              : {cfg.total_batch_size} (given)")
            print(f"torch.cuda.device_count()     : {n_gpus}")
            print(f"per_device_train_batch_size   : {cfg.per_device_train_batch_size} (computed)")
            print(f"actual total batch size       : {cfg.per_device_train_batch_size * n_gpus}")

    if "wandb_run_name" in cfg and cfg.get("wandb_run_name") is None:
        print(f"Setting wandb_run_name    : {cfg.name}")
        cfg.wandb_run_name = cfg.name

    if "output_dir" in cfg and cfg.get("output_dir") is None:
        print(f"Setting output_dir        : {cfg.name}")
        cfg.output_dir = cfg.name

    # Check stop_steps and save_steps
    if cfg.get("stop_steps") is not None and cfg.get("save_steps") is not None:
        if cfg.stop_steps % cfg.save_steps != 0:
            warning = f"stop_steps ({cfg.stop_steps}) is not divisible by save_steps ({cfg.save_steps})"
            warnings.warn(warning)

    if cfg.get("deepspeed") is not None:
        # Prepend PROJECT_ROOT if not absolute using os.path.isabs
        if not os.path.isabs(cfg.deepspeed):
            cfg.deepspeed = os.path.join(PROJECT_ROOT, cfg.deepspeed)
        print(f"Using deepspeed config    : {cfg.deepspeed}")

    if cfg.block_mode:
        # Autofill block model configs
        with open_dict(cfg):
            for model_key in ["embedder", "token_decoder", "block_decoder"]:
                model = cfg[model_key]
                if "config" in model and (model.cls in ["roberta", "roberta_cls"] or model.cls == "gpt-neo-x"):
                    if model.config.get("num_attention_heads") is None:
                        if model.config.hidden_size <= 256:
                            head_dimension = 32
                        elif model.config.hidden_size <= 1536:
                            head_dimension = 64
                        else:
                            head_dimension = 128
                        print(f"[{model_key}] Setting num_attention_heads to hidden_size // {head_dimension}")
                        if model.config.hidden_size % head_dimension != 0:
                            raise ValueError("hidden_size must be divisible by {head_dimension}")
                        model.config.num_attention_heads = model.config.hidden_size // head_dimension
                    if model.config.get("intermediate_size") is None:
                        print(f"[{model_key}] Setting intermediate_size to hidden_size * 4")
                        model.config.intermediate_size = model.config.hidden_size * 4

        # Check decoding strategy
        if cfg.token_decoder.get("decoding_strategy") == "cross_attention":
            assert cfg.token_decoder.cls == "t5", "cross_attention is only supported for T5TokenDecoder"

        with open_dict(cfg):
            if cfg.get("random_pad_first_block") is None:
                cfg.random_pad_first_block = False
            if cfg.get("pad_to_block_boundary") is None:
                cfg.pad_to_block_boundary = False
        if cfg.block_split.distribution != "fixed":
            if cfg.random_pad_first_block or cfg.pad_to_block_boundary:
                raise ValueError("`random_pad_first_block` and `pad_to_block_boundary` are only supported for fixed "
                                 "block length distribution")

        # Warn deprecated configs
        if cfg.embedder.get("projection_hidden_size") is not None:
            warnings.warn("cfg.embedder.projection_hidden_size is deprecated. We now use BaseBlockDecoder.hidden_size "
                          "instead")

    print("-" * 80)
