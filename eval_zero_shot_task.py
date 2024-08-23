import glob
import json
import logging
import os
import re
import sys
import wandb
from pathlib import Path
from collections import defaultdict

import numpy as np
import hydra
from omegaconf import DictConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from safetensors import safe_open
from omegaconf import OmegaConf, open_dict

import lm_eval
from lm_eval import evaluator, utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.tasks import include_path, initialize_tasks
from lm_eval.utils import make_table
from model.block_transformer import BlockTransformer
from model.utils import load_vanilla_model_from_config, load_embedder_from_config, load_block_decoder_from_config, load_token_decoder_from_config
from util.tokenizer import load_tokenizer_from_vanilla_config, load_tokenizer_and_mapper_from_block_config
from util.config import preprocess_config
from paths import PROJECT_ROOT, SAVE_DIR


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


@hydra.main(config_path="conf/eval", config_name="eval_multiple_ckpt")
def main(cfg: DictConfig):
    if cfg.get("output_path") is None:
        cfg.output_path = cfg.name
    
    if cfg.get("wandb"): 
        os.environ["WANDB_ENTITY"] = cfg.wandb_entity
        os.environ["WANDB_PROJECT"] = cfg.wandb_project
        if cfg.get("wandb_watch") is not None:
            os.environ["WANDB_WATCH"] = cfg.get("wandb_watch")
            
    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{cfg.verbosity}"))
    eval_logger.info(f"Verbosity set to {cfg.verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    initialize_tasks(cfg.verbosity)
    
    if cfg.get("eval_multiple_ckpt"):
        eval_multiple_ckpt(cfg, eval_logger)
    else:
        eval_zero_shot_task(cfg, eval_logger)
    
    
def eval_zero_shot_task(cfg: DictConfig, eval_logger=None):    
    if cfg.model in ["hf-auto", "hf", "huggingface"]:
        print("Preparing evaluation of vanilla transformer")
        preprocess_config(cfg, check_mode="vanilla")
        assert cfg.model_args, "Must specify --model_args"

        tokenizer = load_tokenizer_from_vanilla_config(cfg)

        if "safetensors" in cfg.model_args:
            print("Loading vanilla transformer checkpoint")
            OmegaConf.set_struct(cfg, True)
            cfg.model, cfg.model_name_or_path = cfg.model_name_or_path.split(",")
            with open_dict(cfg):
                cfg.use_pretrained_weights = False
            with init_empty_weights():
                model = load_vanilla_model_from_config(cfg)
            
            model_args = cfg.model_args.split("=")
            if len(model_args) == 1:
                ckpt = model_args[0]
            else:
                assert model_args[0] == "pretrained", "model_args should be 'pretrained=ckpt'."
                ckpt = model_args[1]
            load_checkpoint_and_dispatch(model, ckpt, device_map="auto")
            cfg.model, cfg.model_args = "hf", None            
            model = lm_eval.api.registry.get_model(cfg.model)(
                pretrained=model,
                tokenizer=tokenizer,
                device=cfg.device,
                batch_size=cfg.batch_size,
            )
        else:
            print("Using huggingface model")
            model = cfg.model
            
    elif cfg.model in ["block", "block_transformer"]:
        print("Preparing evaluation of block transformer")
        preprocess_config(cfg, check_mode="block")
        
        model_args = cfg.model_args.split("=")
        if len(model_args) == 1:
            ckpt = model_args[0]
        else:
            assert model_args[0] == "pretrained", "model_args should be 'pretrained=ckpt'."
            ckpt = model_args[1]
        
        # Inspect checkpoint
        try:
            _ = safe_open(ckpt, framework="pt", device="cpu")
        except:
            print("Failed to load checkpoint")
            return

        print("Loading block transformer checkpoint")
        tokenizer, token_mapper = load_tokenizer_and_mapper_from_block_config(cfg)
        with init_empty_weights():
            block_decoder = load_block_decoder_from_config(cfg)
            embedder = load_embedder_from_config(cfg, block_decoder)
            token_decoder = load_token_decoder_from_config(cfg, block_decoder)
            model = BlockTransformer(embedder=embedder, block_decoder=block_decoder, token_decoder=token_decoder,
                                     token_mapper=token_mapper,
                                     decoding_strategy=cfg.token_decoder.decoding_strategy,)
        load_checkpoint_and_dispatch(model, ckpt, device_map="auto")
        
        if cfg.get("eval_no_pad", False):
            cfg.model += "_no_pad"
        
        model = lm_eval.api.registry.get_model(cfg.model)(
            pretrained=model,
            tokenizer=tokenizer,
            token_mapper=token_mapper,
            device=cfg.device,
            batch_size=cfg.batch_size,
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model}")

    if cfg.limit:
        eval_logger.warning(
            " --limit SHOULD ONLY BE USED FOR TESTING."
            "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )
    if cfg.include_path is not None:
        eval_logger.info(f"Including path: {cfg.include_path}")
        include_path(cfg.include_path)

    if cfg.tasks is None:
        task_names = ALL_TASKS
    elif cfg.tasks == "list":
        eval_logger.info(
            "Available Tasks:\n - {}".format("\n - ".join(sorted(ALL_TASKS)))
        )
        sys.exit()
    else:
        if os.path.isdir(cfg.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(cfg.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            tasks_list = cfg.tasks.split(",")
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
                eval_logger.error(
                    f"Tasks were not found: {missing}\n"
                    f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lm-eval --tasks list` for list of available tasks, or '--verbosity DEBUG' to troubleshoot task registration issues."
                )

    if cfg.output_path:
        cfg.output_path = os.path.join(SAVE_DIR, cfg.output_path)
        
        path = Path(cfg.output_path)
        # check if file or 'dir/results.json' exists
        if path.is_file():
            eval_logger.warning(
                f"File already exists at {path}. Results will be overwritten."
            )
            output_path_file = path
            path = path.parent
        elif Path(cfg.output_path).joinpath("results.json").is_file():
            eval_logger.warning(
                f"File already exists at {path}. Results will be overwritten."
            )
            output_path_file = path.joinpath("results.json")
        elif path.suffix in (".json", ".jsonl"):
            output_path_file = path
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path.parent
        else:
            path.mkdir(parents=True, exist_ok=True)
            output_path_file = path.joinpath("results.json")
    elif cfg.log_samples and not cfg.output_path:
        assert cfg.output_path, "Specify --output_path"

    eval_logger.info(f"Selected Tasks: {task_names}")

    results = evaluator.simple_evaluate(
        model=model,
        model_args=cfg.model_args,        
        tasks=task_names,
        num_fewshot=cfg.num_fewshot,
        batch_size=cfg.batch_size,
        max_batch_size=cfg.max_batch_size,
        device=cfg.device,
        use_cache=cfg.use_cache,
        limit=cfg.limit,
        decontamination_ngrams_path=cfg.decontamination_ngrams_path,
        check_integrity=cfg.check_integrity,
        write_out=cfg.write_out,
        log_samples=cfg.log_samples,
        gen_kwargs=cfg.gen_kwargs,
        tokenizer=tokenizer,
    )

    if results is not None:
        if cfg.log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(
            results, indent=2, default=_handle_non_serializable, ensure_ascii=False
        )
        if cfg.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        if cfg.output_path:
            print(f"Saving results to {output_path_file}")
            output_path_file.open("w").write(dumped)

            if cfg.log_samples:
                for task_name, config in results["configs"].items():
                    output_name = "{}_{}".format(
                        re.sub("/|=", "__", cfg.model_args), task_name
                    )
                    filename = path.joinpath(f"{output_name}.jsonl")
                    samples_dumped = json.dumps(
                        samples[task_name],
                        indent=2,
                        default=_handle_non_serializable,
                        ensure_ascii=False,
                    )
                    filename.open("w").write(samples_dumped)

        print(
            f"{cfg.model} ({cfg.model_args}), gen_kwargs: ({cfg.gen_kwargs}), limit: {cfg.limit}, num_fewshot: {cfg.num_fewshot}, "
            f"batch_size: {cfg.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))
            
    if cfg.get("eval_multiple_ckpt") and cfg.get("wandb"):
        assert results is not None
        _results = defaultdict(dict)        
        for data in results["results"].keys():
            _results[data] = {}
            for mf, v in results["results"][data].items():
                if mf == "alias":
                    continue
                
                m, _, f = mf.partition(",")
                _results[data][f"{m}"] = v
        return _results


def _restore_config(cfg, train_config, model, output_path, model_args, step):
    cfg.output_path = f"{output_path}/{model_args}/{step}.json"    
    cfg.model = model                
    cfg.tokenizer = train_config.tokenizer
    
    if cfg.model == "hf":
        cfg.model_name_or_path = f"{train_config.model},{train_config.model_name_or_path}"
        if "attn_implementation" in train_config:
            cfg.attn_implementation = train_config.attn_implementation
        if "model_config" in train_config:
            cfg.model_config = train_config.model_config
    elif cfg.model == "block":
        cfg.block_length = train_config.block_length
        cfg.block_split = train_config.block_split
        cfg.embedder = train_config.embedder
        cfg.token_decoder = train_config.token_decoder
        cfg.block_decoder = train_config.block_decoder
    else:
        raise ValueError(f"Unknown model: {cfg.model}")
    
    return cfg
    
    
def eval_multiple_ckpt(cfg: DictConfig, eval_logger=None):        
    config_dict = cfg.configs
    output_path = cfg.output_path
    for model, config_list in config_dict.items():
        for config_name in config_list:
            print("=" * 80)
            print("Evaluating config:", config_name)
            print("=" * 80)
            # check if there is corresponding config file
            fpath = os.path.join(PROJECT_ROOT, "conf", "trainer", f"{config_name}.yaml")
            if not os.path.isfile(fpath):
                print(f"Config file {config_name} does not exist.")
                continue
            
            # read train_config yaml file
            with open(fpath, "r") as f:
                train_config = OmegaConf.load(f)
            
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                ckpt_fpath = f"{cfg.ckpt_path}/{config_name}"
                ckpt_dirs = glob.glob(os.path.join(ckpt_fpath, "checkpoint-*"))
                step_ckpt_dirs = [(int(ckpt_dir.split("-")[-1]), ckpt_dir) for ckpt_dir in ckpt_dirs]
                step_ckpt_dirs.sort()
                last_ckpt = step_ckpt_dirs[-1]
                # filter by step interval
                interval = cfg.get("ckpt_step_interval", 1)
                print(f"Selecting checkpoints at intervals of {interval} steps")
                step_ckpt_dirs = [ckpt_dir for ckpt_dir in step_ckpt_dirs if ckpt_dir[0] % interval == 0]
                # include last checkpoint
                if last_ckpt not in step_ckpt_dirs:
                    print(f"Including checkpoint at final step {last_ckpt[0]}:")
                    print(last_ckpt[1])
                    step_ckpt_dirs.append(last_ckpt)
                ckpt_dirs.sort(key=lambda x: int(x.split("-")[-1]))
                ckpt_dirs = [ckpt_dir for step, ckpt_dir in step_ckpt_dirs]
                if cfg.eval_last_ckpt:  # evaluate only the last checkpoint
                    results = {config_name: defaultdict(dict)}
                    cfg = _restore_config(cfg, train_config, model, output_path, config_name, step=ckpt_dirs[-1])
                    safetensors_path = os.path.join(ckpt_fpath, ckpt_dirs[-1], "model.safetensors")
                    cfg.model_args = f"pretrained={safetensors_path}"

                    print("Evaluating last checkpoint")
                    print("=" * 80)
                    print(safetensors_path)
                    print("=" * 80)

                    _results = eval_zero_shot_task(cfg, eval_logger)
                    results[config_name][int(ckpt_dirs[-1].split("-")[-1])] = _results
                else:
                    results = {config_name: defaultdict(dict)}

                    print("Evaluating all checkpoints")
                    print("=" * 80)
                    for ckpt_dir in ckpt_dirs:
                        if cfg.get("min_steps"):
                            if int(ckpt_dir.split("-")[-1]) < cfg.min_steps:
                                continue
                        safetensors_path = os.path.join(ckpt_fpath, ckpt_dir, "model.safetensors")
                        print(safetensors_path)
                    print("=" * 80)

                    for ckpt_dir in ckpt_dirs:
                        if cfg.get("min_steps"):
                            if int(ckpt_dir.split("-")[-1]) < cfg.min_steps:
                                continue
                        
                        cfg = _restore_config(cfg, train_config, model, output_path, config_name, step=ckpt_dir)
                        safetensors_path = os.path.join(ckpt_fpath, ckpt_dir, "model.safetensors")
                        cfg.model_args = f"pretrained={safetensors_path}"
                        
                        _results = eval_zero_shot_task(cfg, eval_logger)
                        results[config_name][int(ckpt_dir.split("-")[-1])] = _results
                     
            if cfg.get("wandb"):
                for run_name, result in results.items():
                    wandb.init(name=run_name)
                    for step, res in result.items():
                        for data, mv in res.items():
                            for metric, value in mv.items():
                                wandb.log({f"{data}/{metric}": value}, step=int(step))
                    wandb.finish()


if __name__ == "__main__":
    main()