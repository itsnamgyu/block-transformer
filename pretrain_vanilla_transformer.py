import os

import hydra
from omegaconf import DictConfig
from transformers import TrainingArguments, Trainer

from model.utils import load_vanilla_model_from_config
from paths import SAVE_DIR
from util.callback import BatchSizeRampupCallback, FixedStoppingCallback
from util.config import preprocess_config
from util.dataset import load_train_dataset_from_config
from util.tokenizer import load_tokenizer_from_vanilla_config


@hydra.main(config_path="conf/trainer", config_name="pretrain_transformer")
def main(cfg: DictConfig):
    preprocess_config(cfg, check_mode="vanilla")

    os.environ["WANDB_ENTITY"] = cfg.wandb_entity
    os.environ["WANDB_PROJECT"] = cfg.wandb_project
    if cfg.get("wandb_watch") is not None:
        os.environ["WANDB_WATCH"] = cfg.get("wandb_watch")
    os.environ["WANDB_RESUME"] = "allow"
    if cfg.get("wandb_run_id") is not None:
        os.environ["WANDB_RUN_ID"] = cfg.wandb_run_id

    print("Loading tokenizers...")
    tokenizer = load_tokenizer_from_vanilla_config(cfg)

    print("Loading dataset...")
    train_dataset = load_train_dataset_from_config(cfg, tokenizer)

    print("Loading models...")
    model = load_vanilla_model_from_config(cfg)

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
        max_steps=cfg.num_train_steps,
        warmup_steps=cfg.num_warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        dataloader_num_workers=cfg.dataloader_num_workers,
        # multiple workers get replicas of the same data for IterableDatasets
        bf16=cfg.precision == "bf16",
        fp16=cfg.precision == "fp16",
        overwrite_output_dir=True,
        report_to=report_to,
        run_name=cfg.wandb_run_name,
        deepspeed=cfg.deepspeed,
    )

    callbacks = []
    if "wall_time_measurement" in cfg and cfg.wall_time_measurement.get("enable"):
        from util.callback import WallTimeMeasurementCallback
        kwargs = cfg.wall_time_measurement.get("kwargs", {})
        callbacks.append(WallTimeMeasurementCallback(**kwargs))
    if cfg.stop_steps is not None:
        callbacks.append(FixedStoppingCallback(cfg.stop_steps))
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

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
