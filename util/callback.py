import os
import time
import warnings
from typing import List, Tuple
from collections import defaultdict
from omegaconf import OmegaConf, open_dict

import numpy as np
import torch
import wandb
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, Trainer
from transformers.trainer_pt_utils import distributed_concat

import lm_eval
from lm_eval import evaluator, utils
from lm_eval.tasks import initialize_tasks
from lm_eval.api.registry import ALL_TASKS
from model.block_transformer import BlockTransformer


class LossLoggingCallback(TrainerCallback):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log only if it's the right step according to log_steps
        if state.global_step % args.logging_steps == 0:
            model: BlockTransformer = kwargs["model"]
            self.args = args

            # To prevent OOM during all_gather
            torch.cuda.empty_cache()

            block_decoding_loss = model.block_decoding_loss
            token_decoding_loss = model.token_decoding_loss
            auto_encoding_loss = model.auto_encoding_loss
            loss_by_position = None
            mean_length_loss = None
            if hasattr(model.token_decoder, "loss_by_position"):
                # get mean over all steps since last logging step to stabilize curve
                loss_by_position = model.token_decoder.loss_by_position / model.token_decoder.loss_by_position_count
                loss_by_position = loss_by_position.clone()
                model.token_decoder.loss_by_position.zero_()
                model.token_decoder.loss_by_position_count = 0
                if model.token_decoder.block_length is not None:
                    # Only take the mean of the first `block_length` positions
                    # Used to compare variable-length and fixed-length models
                    mean_length_loss = loss_by_position[:model.token_decoder.block_length].mean()

            if block_decoding_loss is not None:
                mean_block_decoding_loss = self.get_mean_across_devices(block_decoding_loss, args.local_rank).item()
            else:
                mean_block_decoding_loss = 0.0

            if token_decoding_loss is not None:
                mean_token_decoding_loss = self.get_mean_across_devices(token_decoding_loss, args.local_rank).item()
            else:
                mean_token_decoding_loss = 0.0

            if auto_encoding_loss is not None:
                mean_auto_encoding_loss = self.get_mean_across_devices(auto_encoding_loss, args.local_rank).item()
            else:
                mean_auto_encoding_loss = 0.0

            if loss_by_position is not None:
                mean_loss_by_position = self.get_mean_across_devices(loss_by_position, args.local_rank, True).tolist()
            else:
                mean_loss_by_position = None

            if mean_length_loss is not None:
                mean_length_loss = self.get_mean_across_devices(mean_length_loss, args.local_rank).item()
            else:
                mean_length_loss = None

            # Only the main process logs the average
            if state.is_world_process_zero:
                print({
                    "block_decoding_loss": mean_block_decoding_loss,
                    "token_decoding_loss": mean_token_decoding_loss,
                    "auto_encoding_loss": mean_auto_encoding_loss,
                    "loss_by_position": mean_loss_by_position,
                    "mean_length_loss": mean_length_loss,
                })

                if self.cfg.get("wandb"):
                    wandb_log = {
                        "train/block_decoding_loss": mean_block_decoding_loss,
                        "train/token_decoding_loss": mean_token_decoding_loss,
                        "train/auto_encoding_loss": mean_auto_encoding_loss,
                        "train/mean_length_loss": mean_length_loss,
                    }
                    if mean_loss_by_position:
                        for i, loss in enumerate(mean_loss_by_position):
                            wandb_log[f"train/loss_by_position/{i + 1}"] = loss
                    num_input_tokens_seen = self.cfg.total_batch_size * self.cfg.max_length * state.global_step
                    wandb_log["num_input_tokens_seen"] = num_input_tokens_seen
                    wandb.log(wandb_log)

    def get_mean_across_devices(self, tensor, local_rank=-1, keepdim=False):
        if local_rank == -1:
            mean_tensor = tensor
        else:
            # Gather losses from all processes using distributed_concat
            if tensor.dim() == 0:
                tensor = torch.Tensor([tensor]).to(tensor.device)
            shape = tensor.shape
            tensors = distributed_concat(tensor)
            devices = tensors.shape[0] // shape[0]
            tensors = tensors.reshape((devices,) + shape)

            # Compute the average loss across all processes
            mean_tensor = tensors.mean() if not keepdim else tensors.mean(dim=0)

        return mean_tensor


class FixedStoppingCallback(TrainerCallback):
    """
    This callback is used when you want to set a certain `num_train_steps` for the learning rate scheduler
    (e.g., `get_linear_schedule_with_warmup`) but you want to stop training before that number of steps is reached.
    """

    def __init__(self, stop_steps: int):
        super().__init__()
        self.stop_steps = stop_steps

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step == self.stop_steps:  # this will stop training at `stop_steps + 1` just to be safe
            print(f"Stopping training at stop_steps={self.stop_steps}")
            control.should_training_stop = True


class BatchSizeRampupCallback(TrainerCallback):
    """
    This callback is a hack to allow us to ramp up the batch size during training. We only consider ramping up from
    half-batch to full-batch. This is because we can only change the `gradient_accumulation_steps` parameter in the
    HuggingFace Trainer (and deepspeed engine), and this may be very small (e.g., 2)

    Note that this does not affect the total number of training tokens because the total number of backward passes
    (num_train_steps x gradient_accumulation_steps) remains the same. It is predetermined in the HuggingFace
    Trainer training loop. However, it will affect the total number of training tokens at each global step, i.e., the
    number of optimization steps.
    """

    def __init__(self, trainer: Trainer, rampup_steps: int):
        super().__init__()
        if rampup_steps < 0:
            raise ValueError(f"rampup_steps must be >= 0, got {rampup_steps}")
        self.rampup_steps = rampup_steps
        self.trainer = trainer  # hack to get access to the trainer
        self.original_gas = trainer.args.gradient_accumulation_steps

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.original_gas > 1:
            if self.original_gas % 2 == 0:
                args.gradient_accumulation_steps //= 2
            else:
                warnings.warn(
                    "gradient_accumulation_steps is not divisible by 2, rounding up to the nearest multiple of 2")
                args.gradient_accumulation_steps = (self.original_gas + 1) // 2
        else:
            raise ValueError("gradient_accumulation_steps must be > 1 for batch size rampup")

        original_batch_size = self.original_gas * args.per_device_train_batch_size * args.world_size
        train_batch_size = args.gradient_accumulation_steps * args.per_device_train_batch_size * args.world_size
        print("Halving batch size prior to rampup...")
        print(f"Changing gradient_accumulation_steps from {self.original_gas} to {args.gradient_accumulation_steps}")
        print(f"Changing total train_batch_size from {original_batch_size} to {train_batch_size}")
        self.trainer.deepspeed.set_train_batch_size(train_batch_size)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step + 1 == self.rampup_steps:
            args.gradient_accumulation_steps = self.original_gas
            train_batch_size = args.gradient_accumulation_steps * args.per_device_train_batch_size * args.world_size
            print("Ramping up batch size...")
            print(f"Changing gradient_accumulation_steps to {args.gradient_accumulation_steps}")
            print(f"Changing total train_batch_size to {train_batch_size}")
            self.trainer.deepspeed.set_train_batch_size(train_batch_size)


class WallTimeMeasurementCallback(TrainerCallback):
    """
    This callback is used to measure the wall time of the training process. This uses `torch.cuda.Event().record()` and
    `torch.cuda.synchronize()` to measure the wall time of the training process as `train/wall_time`. We also measure
    the wall time using `time.time()` as `train/cpu_wall_time`. Wall time is determined by the time between the
    `on_step_end` of the previous step and the current step. Note that both metrics measure the combined wall time of
    the CPU and GPU computation, but the cuda-event based wall time should be more accurate, accounting for
    asynchronous operations.

    See https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch/.

    - All times are measured in milliseconds.
    - Time is only measured for the main process (i.e., `state.is_world_process_zero`). Measured wall-time will
        include the time spent on communication between processes. We explicitly do not synchronize (`all_gather`)
        devices for time measurements, as this would add significant overhead.
    """

    def __init__(self, warmup_steps: int = 20, print_stats_interval: int = 20):
        """
        :param warmup_steps: The number of steps to wait before starting to measure wall time. This is because the
            first few steps may be slower due to initialization and caching.
        :param print_stats_interval: The interval at which to print the mean and standard deviation of the
            `train/wall_time` and `train/cpu_wall_time` metrics. This is useful for monitoring the stability of the
            training process.
        """
        super().__init__()
        self.warmup_steps = warmup_steps
        self.events_by_step: List[Tuple[int, torch.cuda.Event]] = []
        self.wall_times: List[float] = []
        self.cpu_wall_times: List[float] = []
        self.last_cpu_time = None
        self.print_stats_interval = print_stats_interval

    def _print_stats(self, wall_time_history, label):
        times = wall_time_history[-self.print_stats_interval:]
        next_step = self.warmup_steps + len(wall_time_history) + 1
        mean, std = sum(times) / len(times), np.std(times)
        s1, s2 = next_step - self.print_stats_interval, next_step
        print(f"{label} between steps [{s1:>03d}, {s2:>03d}): mean={mean:.2f}ms, std={std:.2f}ms")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero and state.global_step >= self.warmup_steps - 1:
            # -1 because we need to record the start of the 0th step, i.e., end of the -1th step
            # synchronously measure CPU wall time
            cpu_time = time.time()
            if self.last_cpu_time is not None:
                cpu_wall_time = (cpu_time - self.last_cpu_time) * 1000
                self.cpu_wall_times.append(cpu_wall_time)
                if len(self.cpu_wall_times) % self.print_stats_interval == 0:
                    self._print_stats(self.cpu_wall_times, "`time`-based wall time")
                wandb.log({"train/cpu_wall_time": cpu_wall_time, "train/global_step": state.global_step})
            self.last_cpu_time = cpu_time

            # asynchronously measure GPU wall time by polling the events
            event = torch.cuda.Event(enable_timing=True)
            self.events_by_step.append((state.global_step, event))
            event.record()
            completed = 0
            for (s, e1), (_, e2) in zip(self.events_by_step[:-2], self.events_by_step[1:-1]):
                if e2.query():
                    wall_time = e1.elapsed_time(e2)
                    completed += 1
                    self.wall_times.append(wall_time)
                    if len(self.wall_times) % self.print_stats_interval == 0:
                        self._print_stats(self.wall_times, "`cuda.Event`-based wall time")
                    wandb.log({"train/wall_time": wall_time, "train/global_step": state.global_step})
                else:
                    break
            self.events_by_step = self.events_by_step[completed:]


class ZeroshotEvalCallback(TrainerCallback):
    """
    This callback is used to evaluate the model on the zero-shot task. This is done by running lm_eval at the end of
    each epoch. The results are logged to wandb.
    """

    def __init__(self, cfg, tokenizer) -> None:
        super().__init__()

        self.cfg = cfg
        self.tokenizer = tokenizer

    def get_tasks(self, eval_cfg):
        if not ALL_TASKS:
            initialize_tasks('INFO')
        if eval_cfg.tasks is None:
            task_names = ALL_TASKS
        else:
            if os.path.isdir(eval_cfg.tasks):
                import glob

                task_names = []
                yaml_path = os.path.join(eval_cfg.tasks, "*.yaml")
                for yaml_file in glob.glob(yaml_path):
                    config = utils.load_yaml_config(yaml_file)
                    task_names.append(config)
            else:
                tasks_list = eval_cfg.tasks.split(",")
                task_names = utils.pattern_match(tasks_list, ALL_TASKS)
                for task in [task for task in tasks_list if task not in task_names]:
                    if os.path.isfile(task):
                        config = utils.load_yaml_config(task)
                        task_names.append(config)
                task_missing = [
                    task
                    for task in tasks_list
                    if task not in task_names and "*" not in task
                ]  # we don't want errors if a wildcard ("*") task name was used

                if task_missing:
                    missing = ", ".join(task_missing)
                    raise ValueError(f"Tasks not found: {missing}.")

        return task_names

    def get_model(self, args, **kwargs):
        model = kwargs["model"]
        if self.cfg.get("model") is not None:
            print("Evaluate zero-shot task with model that pretrained with pretrain_vanilla_transformer.py")
            mode = "hf"
        else:
            print("Evaluate zero-shot task with model that pretrained with pretrain_block_transformer.py")
            mode = "block"

        eval_cfg = self.cfg.get("zero_shot_eval")
        if eval_cfg.get("device") is None:
            OmegaConf.set_struct(eval_cfg, True)
            with open_dict(eval_cfg):
                eval_cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            warnings.warn(f"If you want to specify the device, use eval_cfg.device")
        if eval_cfg.get("eval_batch_size") is None:
            OmegaConf.set_struct(eval_cfg, True)
            with open_dict(eval_cfg):
                eval_cfg.eval_batch_size = args.gradient_accumulation_steps * args.per_device_train_batch_size * args.world_size
            warnings.warn(f"If you want to specify the eval_batch_size, use eval_cfg.eval_batch_size")
        
        if self.cfg.zero_shot_eval.get("eval_no_pad", False):
            mode += "_no_pad"

        model = lm_eval.api.registry.get_model(mode)(
            pretrained=model,
            tokenizer=self.tokenizer,
            device=eval_cfg.device,
            batch_size=eval_cfg.eval_batch_size,
        )
        return model, eval_cfg

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero and state.global_step % self.cfg.zero_shot_eval.eval_steps == 0:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            model, eval_cfg = self.get_model(args, **kwargs)
            task_names = self.get_tasks(eval_cfg)

            results = evaluator.simple_evaluate(
                model=model,
                tasks=task_names,
                tokenizer=self.tokenizer,
                batch_size=eval_cfg.eval_batch_size,
                device=eval_cfg.device,
                # num_fewshot=eval_cfg.num_fewshot,
                # max_batch_size=eval_cfg.max_batch_size,
                # use_cache=eval_cfg.use_cache,
                # limit=eval_cfg.limit,
                # decontamination_ngrams_path=eval_cfg.decontamination_ngrams_path,
                # check_integrity=eval_cfg.check_integrity,
                # write_out=eval_cfg.write_out,
                # log_samples=eval_cfg.log_samples,
                # gen_kwargs=eval_cfg.gen_kwargs,
            )

            if results is not None:
                values = []
                for k, dic in results["results"].items():
                    version = results["versions"][k]
                    n = str(results["n-shot"][k])

                    if "alias" in dic:
                        k = dic.pop("alias")

                    for (mf), v in dic.items():
                        m, _, f = mf.partition(",")
                        if m.endswith("_stderr"):
                            continue

                        if m + "_stderr" + "," + f in dic:
                            se = dic[m + "_stderr" + "," + f]
                            if se != "N/A":
                                se = "%.4f" % se
                            values.append([k, version, f, n, m, "%.4f" % v, "±", se])
                        else:
                            values.append([k, version, f, n, m, "%.4f" % v, "", ""])
                        k = ""
                        version = ""

                if values:
                    print("Zero-shot evaluation results:")
                    print("-" * 90)
                    for v in values:
                        print(" | ".join([f"{_v:^14}" if i in [0, 4, 5, 7] else f"{_v:4}" for i, _v in enumerate(v)]))
                    print("-" * 90)

            if self.cfg.get("wandb"):
                wandb_log = defaultdict(float)
                for v in values:
                    if v[0]:  # else: use the previous task name
                        name = v[0]
                    metric = v[4]
                    mean_value = float(v[5])

                    wandb_log[f"eval/{name}_{metric}"] = mean_value
                wandb.log(wandb_log)

            torch.cuda.empty_cache()
            end.record()
            torch.cuda.synchronize()
            print(f"Zero-shot evaluation took {start.elapsed_time(end):.2f}ms")
