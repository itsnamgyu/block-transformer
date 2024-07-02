"""
Usage
```
python measure_generation_time.py --config_name=block_main_b4_85
python measure_generation_time.py --config_name=block_main_b4_85 ++benchmark_batch_sizes=[1,2,4,8,16,32,64]
python measure_generation_time.py --config_name=block_main_b4_85 ++benchmark_prefill_length=1 ++benchmark_decode_length=2048
```
"""
import math
import os
import sys
import time
from typing import Optional

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, open_dict
from torch.cuda import OutOfMemoryError
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import trange
from transformers import GPTNeoXTokenizerFast, PreTrainedModel

from model.block_transformer import BlockTransformer
from model.utils import load_vanilla_model_from_config, load_embedder_from_config, load_block_decoder_from_config, \
    load_token_decoder_from_config
from paths import SAVE_DIR, PROJECT_ROOT
from util.config import preprocess_config
from util.tokenizer import load_tokenizer_and_mapper_from_block_config

DEVICE = "cuda"
DEFAULT_PREFILL_LENGTH = 1
DEFAULT_DECODE_LENGTH = 2048


def get_number_of_repetitions(max_length):
    if max_length < 256:
        return 10
    else:
        return 5


def get_max_length(block_length: Optional[int], prefill_length: int, decode_length: int):
    if block_length:
        prefill_blocks = math.ceil(prefill_length / block_length)
        return prefill_blocks * block_length + decode_length
    else:
        return prefill_length + decode_length


def prepare_vanilla_model(cfg: DictConfig, prefill_length, decode_length):
    with open_dict(cfg):
        if "model_config" not in cfg:
            cfg.model_config = {}
        cfg.model_config["max_position_embeddings"] = get_max_length(None, prefill_length, decode_length) - 1
    model = load_vanilla_model_from_config(cfg)
    model.generation_config.eos_token_id = None
    return model


def prepare_block_model(cfg: DictConfig, prefill_length, decode_length):
    tokenizer, token_mapper = load_tokenizer_and_mapper_from_block_config(cfg)

    with open_dict(cfg):
        if "config" not in cfg.block_decoder:
            cfg.block_decoder.config = {}
        max_length = get_max_length(cfg.block_length, prefill_length, decode_length)
        max_blocks = math.ceil(max_length / cfg.block_length)
        cfg.block_decoder.config["max_position_embeddings"] = max_blocks - 1

    block_decoder = load_block_decoder_from_config(cfg)
    embedder = load_embedder_from_config(cfg, block_decoder)
    token_decoder = load_token_decoder_from_config(cfg, block_decoder)
    model = BlockTransformer(embedder=embedder, block_decoder=block_decoder, token_decoder=token_decoder,
                             token_mapper=token_mapper,
                             use_token_decoding_loss=cfg.token_decoding_loss.enable,
                             use_block_decoding_loss=cfg.block_decoding_loss.enable,
                             block_decoding_loss_weight=cfg.block_decoding_loss.weight,
                             decoding_strategy=cfg.token_decoder.decoding_strategy, )

    if isinstance(tokenizer, GPTNeoXTokenizerFast):
        # pad token exists in vocab but not in gpt-neox tokenizer nor gpt-neox model config
        # this is done to differentiate eos token and pad token. if not, then we erroneously get the
        # "A decoder-only architecture is being used, but right-padding was detected!" warning because
        # token decoding starts with eos
        token_decoder.config.pad_token_id = 1

    token_decoder.config.eos_token_id = 0
    token_decoder.generation_config.eos_token_id = None

    return model


def generate_with_vanilla_model_and_measure_time(model: PreTrainedModel, prefill_length, decode_length, batch_size,
                                                 log_path=None) -> float:
    """
    :param model:
    :param prefill_length:
    :param decode_length:
    :param batch_size:
    :return: milliseconds
    """
    assert prefill_length > 0
    assert decode_length > 0
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, prefill_length))
    max_length = get_max_length(None, prefill_length, decode_length)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.inference_mode():
        if log_path is None:
            output = model.generate(input_ids.to(DEVICE), max_length=max_length)
        else:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
                         profile_memory=True,
                         on_trace_ready=torch.profiler.tensorboard_trace_handler(log_path), with_stack=True) as prof:
                with record_function("model_inference"):
                    output = model.generate(input_ids.to(DEVICE), max_length=max_length)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    end.record()

    assert output.shape[1] == max_length
    torch.cuda.synchronize()

    return start.elapsed_time(end)


def generate_with_block_model_and_measure_time(model: BlockTransformer, prefill_length, decode_length, batch_size,
                                               log_path=None) -> float:
    """
    :param model:
    :param prefill_length:
    :param decode_length:
    :param batch_size:
    :return: milliseconds
    """
    max_length = get_max_length(model.block_length, prefill_length, decode_length)
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, prefill_length))
    inputs = model.preprocess_inputs_for_generation(input_ids.to(DEVICE))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    if log_path is None:
        output = model.generate(**inputs, max_length=max_length, benchmark=False)
    else:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False, profile_memory=True, with_stack=False) as prof:
            with record_function("model_inference"):
                output = model.generate(**inputs, max_length=max_length, benchmark=False)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        prof.export_chrome_trace(log_path)
    end.record()

    assert output.shape[1] * output.shape[2] == math.ceil(max_length / model.block_length) * model.block_length
    torch.cuda.synchronize()

    return start.elapsed_time(end)


def prepare_model(cfg: DictConfig, prefill_length, decode_length):
    if cfg.block_mode:
        model = prepare_block_model(cfg, prefill_length, decode_length)
    else:
        model = prepare_vanilla_model(cfg, prefill_length, decode_length)
    return model


def generate(model, prefill_length, decode_length, batch_size, log_path=None):
    if isinstance(model, BlockTransformer):
        return generate_with_block_model_and_measure_time(model, prefill_length, decode_length, batch_size, log_path)
    else:
        return generate_with_vanilla_model_and_measure_time(model, prefill_length, decode_length, batch_size, log_path)


def measure_generation(model, prefill_length, decode_length, batch_size, log_path=None):
    print(f" ( prefill={prefill_length} decode={decode_length} / bs={batch_size} ) ".center(80, "-"))
    torch.cuda.empty_cache()

    # Warmup
    try:
        print("Warming up... ", end="")
        sys.stdout.flush()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = generate(model, prefill_length, decode_length, batch_size)
        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)
        print(f"done in {time:.2f}ms")

        # Measure
        times = []
        torch.cuda.reset_peak_memory_stats(DEVICE)
        if isinstance(model, BlockTransformer):
            repeat = get_number_of_repetitions(get_max_length(model.block_length, prefill_length, decode_length))
        else:
            repeat = get_number_of_repetitions(get_max_length(None, prefill_length, decode_length))
        for _ in trange(repeat, desc=f"bs={batch_size}"):
            times.append(generate(model, prefill_length, decode_length, batch_size, log_path))
        max_memory_allocated = torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024 / 1024  # in GiB
        max_memory_reserved = torch.cuda.max_memory_reserved(DEVICE) / 1024 / 1024 / 1024  # in GiB
        times = np.array(times)
        mean = np.mean(times).item()

        # Print stats
        tps = mean / batch_size
        tpt = mean / decode_length / batch_size
        print(f"mean:                 {mean:8.2f}ms / {tps:8.2f}ms per sample / {tpt:8.2f}ms per token")
        print(f"max memory_allocated: {max_memory_allocated:8.2f}GB")
        print(f"max memory_reserved:  {max_memory_reserved:8.2f}GB")

        stats = {
            "prefill_length": prefill_length,
            "decode_length": decode_length,
            "batch_size": batch_size,
            "oom": False,
            "mean": mean,
            "mean_per_sample": mean / batch_size,
            "mean_per_token": mean / decode_length / batch_size,
            "max_memory_allocated": max_memory_allocated,
            "max_memory_reserved": max_memory_reserved,
        }

        if len(times) > 1:
            std = np.std(times).item()
            print(f"std:                   {std:8.2f}ms")
            stats["std"] = std

        print()

        return stats
    except RuntimeError as e:
        if isinstance(e, OutOfMemoryError) or "out of memory" in str(e):
            print()
            print(f"Out of memory using batch size {batch_size} for prefill={prefill_length} decode={decode_length}")
            torch.cuda.empty_cache()
            return {
                "batch_size": batch_size,
                "prefill_length": prefill_length,
                "decode_length": decode_length,
                "oom": True,
            }
        else:
            raise e


def predict_oom(data, batch_size, prefill_length, decode_length):
    """
    Predict OOM based on previous measurements (very conservatively)
    """
    if not data:
        return False
    df = pd.DataFrame(data)
    c1 = (df.batch_size <= batch_size)
    c2 = (df.prefill_length == prefill_length)
    c3 = (df.decode_length == decode_length)
    if df[c1 & c2 & c3].oom.any():
        return True
    else:
        return False


def get_max_batch_size_row(data, prefill_length, decode_length, oom=False):
    if not data:
        return None
    df = pd.DataFrame(data)
    df = df[df.prefill_length == prefill_length]
    df = df[df.decode_length == decode_length]
    df = df[df.oom == oom]
    if df.empty:
        return None
    return df.iloc[df.batch_size.argmax()]


def measurement_exists(data, batch_size, prefill_length, decode_length):
    if not data:
        return False
    df = pd.DataFrame(data)
    c1 = df.batch_size == batch_size
    c2 = df.prefill_length == prefill_length
    c3 = df.decode_length == decode_length
    return (c1 & c2 & c3).any()


def predict_memory_per_sample(data, prefill_length, decode_length):
    df = pd.DataFrame(data)
    df = df[df.prefill_length == prefill_length]
    df = df[df.decode_length == decode_length]
    if (~df.oom).sum() < 2:
        print("Not enough data to predict memory per sample.")
        return None

    df.sort_values("batch_size", inplace=True)
    print(" Previous measurements ".center(60, "-"))
    print(df.loc[:, ["batch_size", "prefill_length", "decode_length", "max_memory_allocated", "mean_per_sample"]])
    df = df[~df.oom]
    b1, b2 = df.iloc[0].batch_size, df.iloc[-1].batch_size
    m1, m2 = df.iloc[0].max_memory_allocated, df.iloc[-1].max_memory_allocated
    print("-" * 60)

    memory_per_sample = (m2 - m1) / (b2 - b1)
    print(f"Predicted memory per sample: {memory_per_sample} GiB")

    if memory_per_sample <= 0:
        if b1 == 1 and b2 == 2:
            # this can happen when the memory usage is too small, assume 0.1 GiB
            memory_per_sample = 0.1
        else:
            raise ValueError("Predicted memory per sample is negative. This is unexpected.")

    return memory_per_sample


def find_next_batch_size(data, prefill_length, decode_length, available_memory):
    """
    Find middle ground between (1) smallest batch size that did not OOM and (2) predicted max batch size that fills
    100% of available VRAM or smallest batch size that OOMed

    Used to find next candidate for batch size binary search

    Return None if end of searchf
    """
    print("Finding next batch size for binary search...")
    df = pd.DataFrame(data)
    df = df[df.prefill_length == prefill_length]
    df = df[df.decode_length == decode_length]

    memory_per_sample = predict_memory_per_sample(data, prefill_length, decode_length)
    if memory_per_sample is None:
        print("End of search")
        return None

    # find largest batch size that did not OOM
    em_df = df[~df.oom].sort_values("batch_size")  # enough memory
    if em_df.empty:
        raise ValueError("No successful measurements found.")
    b1 = em_df.iloc[-1].batch_size
    m1 = em_df.iloc[-1].max_memory_allocated

    oom_df = df[df.oom]
    if oom_df.empty:
        # predict max batch size (filling exactly 100% of available memory)
        remaining = available_memory - m1
        remaining_samples = int(remaining / memory_per_sample)
        b2 = b1 + remaining_samples
    else:
        # find smallest batch size that OOMed
        b2 = oom_df.batch_size.min()

    middle = (b1 + b2) // 2

    # roughly round to an appropriate multiple of a power of 2
    if 2048 <= middle:
        middle = round(middle / 256) * 256
    if 512 <= middle < 2048:
        middle = round(middle / 64) * 64
    if 128 <= middle < 512:
        middle = round(middle / 16) * 16
    if 32 <= middle < 128:
        middle = round(middle / 4) * 4

    if middle == b2 or middle <= b1:
        print("End of search")
        return None
    else:
        print(f"Found next batch size: {middle}")
        return middle


@hydra.main(config_path="conf/trainer", config_name="pretrain_transformer")
def main(cfg: DictConfig):
    global_start_time = time.time()

    preprocess_config(cfg)

    # for profiling
    if cfg.get("profiling", False):
        if cfg.get("log_path") is None:
            with open_dict(cfg):
                cfg.log_path = os.path.join(PROJECT_ROOT, "results", "profiler")
        if not os.path.isdir(cfg.log_path):
            os.makedirs(cfg.log_path, exist_ok=True)
    else:
        with open_dict(cfg):
            cfg.log_path = None

    print()
    print(" Benchmark configurations ".center(80, "-"))
    if "benchmark_batch_sizes" in cfg:
        print(f"Batch sizes:    {cfg.benchmark_batch_sizes}")
    else:
        print(f"Batch sizes:    auto (default)")
    if "benchmark_prefill_length" in cfg:
        print(f"Prefill length: {cfg.benchmark_prefill_length}")
    else:
        print(f"Prefill length: {DEFAULT_PREFILL_LENGTH} (default)")
    if "benchmark_decode_length" in cfg:
        print(f"Decode length:  {cfg.benchmark_decode_length}")
    else:
        print(f"Decode length:  {DEFAULT_DECODE_LENGTH} (default)")

    batch_sizes = cfg.get("benchmark_batch_sizes", None)
    prefill_length = cfg.get("benchmark_prefill_length", DEFAULT_PREFILL_LENGTH)
    decode_length = cfg.get("benchmark_decode_length", DEFAULT_DECODE_LENGTH)

    total_memory = torch.cuda.get_device_properties(DEVICE).total_memory / 1024 / 1024 / 1024  # in GiB

    output_path = os.path.join(SAVE_DIR, cfg.output_dir, "generation_time.csv")
    print(" Output path ".center(80, "-"))
    print(output_path)

    if os.path.exists(output_path):
        print("Reading existing measurement data")
        df = pd.read_csv(output_path, index_col=0)
        data = df.to_dict(orient="records")
        for record in data:
            if "oom" not in record:
                # Old measurements did not have OOM flag
                # Set to False as only successful measurements had been saved
                record["oom"] = False
            if "prefill_length" not in record:
                # Old measurements did not have "prefill_length" and "decode_length"
                # They only had "length" which was the "decode_length", with "prefill_length" = 1
                record["prefill_length"] = 1
                record["decode_length"] = record["length"]
                del record["length"]
    else:
        data = []
    print("-" * 80)
    print()

    print(" Preparing model ".center(80, "-"))
    model = prepare_model(cfg, prefill_length, decode_length)
    model.to(DEVICE)
    print("-" * 80)
    print()

    print(" Running generation measurements ".center(80, "-"))

    if batch_sizes is None:
        # auto-find batch sizes (binary search)
        for batch_size in [1, 2]:
            if not measurement_exists(data, batch_size, prefill_length, decode_length):
                stats = measure_generation(model, prefill_length, decode_length, batch_size)
                data.append(stats)

        max_memory = get_max_batch_size_row(data, prefill_length, decode_length, oom=False).max_memory_allocated
        while total_memory - max_memory >= 2:
            next_batch_size = find_next_batch_size(data, prefill_length, decode_length, total_memory)
            if next_batch_size is None:
                break
            if predict_oom(data, next_batch_size, prefill_length, decode_length):
                raise AssertionError("Something is wrong with the measurements")

            if cfg.log_path is not None:
                log_path = os.path.join(cfg.log_path, f"{cfg.name}_{next_batch_size}_{prefill_length}_{decode_length}.log")
            else:
                log_path = None
            stats = measure_generation(model, prefill_length, decode_length, next_batch_size, log_path=log_path)
            data.append(stats)

            max_memory = get_max_batch_size_row(data, prefill_length, decode_length, oom=False).max_memory_allocated
            print("-" * 80)
    else:
        # loop over predefined batch sizes
        for batch_size in batch_sizes:
            if measurement_exists(data, batch_size, prefill_length, decode_length):
                print(f"Skipping batch size {batch_size} as it is already measured")
                continue

            if predict_oom(data, batch_size, prefill_length, decode_length):
                print(f"Skipping batch size {batch_size} as it is predicted to OOM")
                continue

            if cfg.log_path is not None:
                log_path = os.path.join(cfg.log_path, f"{cfg.name}_{batch_size}_{prefill_length}_{decode_length}.log")
            else:
                log_path = None
            stats = measure_generation(model, prefill_length, decode_length, batch_size, log_path=log_path)
            data.append(stats)
            print("-" * 80)

    df = pd.DataFrame(data)
    df = df.sort_values(["prefill_length", "decode_length", "batch_size"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)

    print(" Generation time data saved to ".center(80, "-"))
    print(output_path)
    print("-" * 80)
    print(f"Total time: {time.time() - global_start_time:.2f}s")


if __name__ == "__main__":
    main()
