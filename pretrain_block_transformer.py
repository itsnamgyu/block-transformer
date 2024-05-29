import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from accelerate import load_checkpoint_and_dispatch
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments, Trainer, GPTNeoXForCausalLM

from model.block_transformer import BlockTransformer
from model.utils import load_embedder_from_config, load_token_decoder_from_config, load_block_decoder_from_config, \
    get_torch_dtype
from paths import *
from util.callback import LossLoggingCallback, BatchSizeRampupCallback, FixedStoppingCallback, ZeroshotEvalCallback
from util.config import preprocess_config
from util.dataset import load_train_dataset_from_config
from util.tokenizer import load_tokenizer_and_mapper_from_block_config


@hydra.main(config_path="conf/trainer", config_name="pretrain_block_transformer")
def main(cfg: DictConfig):
    preprocess_config(cfg, check_mode="block")

    os.environ["WANDB_ENTITY"] = cfg.wandb_entity
    os.environ["WANDB_PROJECT"] = cfg.wandb_project
    if cfg.get("wandb_watch") is not None:
        os.environ["WANDB_WATCH"] = cfg.get("wandb_watch")
    os.environ["WANDB_RESUME"] = "allow"
    if cfg.get("wandb_run_id") is not None:
        os.environ["WANDB_RUN_ID"] = cfg.wandb_run_id

    print("Loading tokenizers...")
    tokenizer, token_mapper = load_tokenizer_and_mapper_from_block_config(cfg)

    print("Loading dataset...")
    train_dataset = load_train_dataset_from_config(cfg, tokenizer)

    print("Loading models...")
    block_decoder = load_block_decoder_from_config(cfg)
    embedder = load_embedder_from_config(cfg, block_decoder)
    token_decoder = load_token_decoder_from_config(cfg, block_decoder)
    model = BlockTransformer(embedder=embedder, block_decoder=block_decoder, token_decoder=token_decoder,
                             token_mapper=token_mapper,
                             use_token_decoding_loss=cfg.token_decoding_loss.enable,
                             use_block_decoding_loss=cfg.block_decoding_loss.enable,
                             block_decoding_loss_weight=cfg.block_decoding_loss.weight,
                             use_auto_encoding_loss=cfg.auto_encoding_loss.enable,
                             auto_encoding_loss_weight=cfg.auto_encoding_loss.weight,
                             decoding_strategy=cfg.token_decoder.decoding_strategy, )

    if cfg.get("load_from_vanilla") and cfg.load_from_vanilla.get("enable") and not cfg.get("resume_from_checkpoint"):
        print("-" * 80)
        print("Loading weights from vanilla transformer...")
        print("-" * 80)
        from model.utils import load_vanilla_model_from_config, load_block_from_vanilla_checkpoint

        # Load vanilla model
        if cfg.load_from_vanilla.get("config_path"):
            path = os.path.join(PROJECT_ROOT, cfg.load_from_vanilla.config_path)
            vanilla_cfg = OmegaConf.load(path)
            preprocess_config(vanilla_cfg)
            vanilla = load_vanilla_model_from_config(vanilla_cfg)
            print("Loading vanilla weights from checkpoint:")
            print(cfg.load_from_vanilla.checkpoint_path)
            load_checkpoint_and_dispatch(vanilla, cfg.load_from_vanilla.checkpoint_path)
            print("done")
        elif cfg.load_from_vanilla.get("model_name_or_path"):
            path = cfg.load_from_vanilla.model_name_or_path
            if "pythia" not in path:
                raise NotImplementedError("Only Pythia models are supported for now")
            print(f"Loading HuggingFace pretrained vanilla model '{path}'...", end="")
            vanilla = GPTNeoXForCausalLM.from_pretrained(cfg.load_from_vanilla.model_name_or_path,
                                                         torch_dtype=get_torch_dtype(cfg))
            print(" done")
        else:
            raise ValueError("Either `config_path` or `model_name_or_path` must be provided in `load_from_vanilla`.")

        # Load weights
        model = load_block_from_vanilla_checkpoint(cfg, model, vanilla)
        print("-" * 80)
        print("Finished loading weights from vanilla transformer...")
        print("-" * 80)

        del vanilla

    print("Loading trainer...")
    if cfg.freeze:
        # TODO: hardcode the names of the embedder parameters
        for p in model.embedder.encoder.parameters():
            p.requires_grad = False
    else:
        if cfg.adaptive_lr_auto_encoder:
            raise NotImplementedError("Adaptive LR is not implemented yet")

    report_to = []
    if cfg.wandb:
        report_to.append("wandb")

    train_args = TrainingArguments(
        learning_rate=cfg.learning_rate,
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        weight_decay=cfg.weight_decay,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        output_dir=os.path.join(SAVE_DIR, cfg.output_dir),
        overwrite_output_dir=True,
        max_steps=cfg.num_train_steps,
        warmup_steps=cfg.num_warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        dataloader_num_workers=cfg.dataloader_num_workers,
        # multiple workers get replicas of the same data for IterableDatasets
        fp16=cfg.precision == "fp16",
        bf16=cfg.precision == "bf16",
        report_to=report_to,
        run_name=cfg.wandb_run_name,
        deepspeed=cfg.deepspeed,
    )

    callbacks = [
        LossLoggingCallback(cfg=cfg),
    ]
    if "wall_time_measurement" in cfg and cfg.wall_time_measurement.get("enable"):
        from util.callback import WallTimeMeasurementCallback
        kwargs = cfg.wall_time_measurement.get("kwargs", {})
        callbacks.append(WallTimeMeasurementCallback(**kwargs))
    if cfg.stop_steps is not None:
        callbacks.append(FixedStoppingCallback(cfg.stop_steps))
    if cfg.get("zero_shot_eval") and cfg.zero_shot_eval.enable:
        callbacks.append(ZeroshotEvalCallback(cfg, tokenizer))
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        callbacks=callbacks,
    )
    if cfg.batch_size_rampup_steps:
        trainer.add_callback(BatchSizeRampupCallback(trainer, cfg.batch_size_rampup_steps))

    train_result = trainer.train(
        resume_from_checkpoint=cfg.resume_from_checkpoint
    )

    emb_path = os.path.join(SAVE_DIR, cfg.output_dir, "embedder")
    os.makedirs(emb_path, exist_ok=True)
    trainer.model.embedder.save_pretrained(f"{emb_path}")

    dec_path = os.path.join(SAVE_DIR, cfg.output_dir, "token_decoder")
    os.makedirs(dec_path, exist_ok=True)
    trainer.model.token_decoder.save_pretrained(f"{dec_path}")

    dec_path = os.path.join(SAVE_DIR, cfg.output_dir, "block_decoder")
    os.makedirs(dec_path, exist_ok=True)
    trainer.model.block_decoder.save_pretrained(f"{dec_path}")

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
