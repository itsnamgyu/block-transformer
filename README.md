# block-transformer

Official code for "Block Transformer: Global-to-Local Language Modeling for Fast Inference"

## Getting Started

Install requirements and prepare the Pile dataset as described below.

- Vanilla (HuggingFace) model training: `pretrain_vanilla_transformer.py`
    ```bash
    deepspeed --include localhost:0,1,2,3 --no_local_rank --master_port 29540 pretrain_vanilla_transformer.py --config-name vanilla_31 pythia_pile_idxmaps_path=/path/to/pythia_pile_idxmaps
    ```

- Block transformer training: `pretrain_vanilla_transformer.py`
  ```bash
    deepspeed --include localhost:0,1,2,3 --no_local_rank --master_port 29540 pretrain_block_transformer.py --config-name block_main_b4_5 pythia_pile_idxmaps_path=/path/to/pythia_pile_idxmaps
    ```

- Using the `torch.distributed` launcher
    ```bash
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port=29540
    ```
    - Note that this still uses deepspeed optimization. To run without deepspeed optimization,
      append `--deepspeed=null`.

- Zero-shot evaluation: `eval_zero_shot_task.py`
    ```bash
    CUDA_VISIBLE_DEVICES=0 python eval_zero_shot_task.py --config-name=240425_eval_multiple_ckpt configs.hf=["vanilla_31"] batch_size=64
    CUDA_VISIBLE_DEVICES=0 python eval_zero_shot_task.py --config-name=240425_eval_multiple_ckpt configs.block=["block_main_b4_5"] batch_size=64
    ```
  
- Inference throughput wall-time measurement: `measure_generation_time.py`
    ```bash
    CUDA_VISIBLE_DEVICES=0 python measure_generation_time.py --config_name=block_main_b4_5 ++benchmark_prefix_length=2048 ++benchmark_decode_length=128
    CUDA_VISIBLE_DEVICES=0 python measure_generation_time.py --config_name=block_main_b4_5 ++benchmark_prefix_length=128 ++benchmark_decode_length=2048
    ```
  - Works for both HF and block models.
  - By default, batch size is auto-tuned via binary search to maximize VRAM utilization.To set a specific batch size,
    use `++batch_size=64`.

## Requirements

Refer to `requirements.txt`.

### Installing FlashAttention

Requires `CUDA>=11.6` and `PyTorch>=1.12` with GPU support.
See https://github.com/Dao-AILab/flash-attention#installation-and-features.

```
pip install packaging ninja
ninja --version; echo $?  # make sure that 0 is printed. else, reinstall ninja
pip install flash-attn --no-build-isolation
```

Building wheels takes a few minutes (we've seen 10 minutes+).

FlashAttention support for GPTNeoX was added in Dec 7, 2023 and released v4.36.0.
https://github.com/huggingface/transformers/pull/26463
Update transformers to the latest version if you are using an older version.

```
pip install transformers --upgrade
```

## Data Preparation

### The Pile (Pythia version)

Refer to `https://github.com/EleutherAI/pythia/`. The resulting files are a Megatron-LM compatible dataset of
The Pile (in memory-mapped Numpy format), pre-shuffled document-wise and pre-tokenized, without any added special
tokens. The dataset can be accessed via https://github.com/EleutherAI/pythia/blob/main/utils/mmap_dataset.py.

```
git clone https://github.com/EleutherAI/pythia/  # about 500MB
cd pythia

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/EleutherAI/pythia_deduped_pile_idxmaps
cd pythia_deduped_pile_idxmaps
git config lfs.activitytimeout 3600
# sudo apt-get update; sudo apt-get install git-lfs -y
git lfs pull

cd ..

# Optionally, to ensure against corrupt files
python utils/checksum_shards.py

# Unshard data
python utils/unshard_memmap.py --input_file ./pythia_deduped_pile_idxmaps/pile_0.87_deduped_text_document-00000-of-00082.bin --num_shards 83 --output_dir ./pythia_pile_idxmaps/

# Copy over idx data
cp pythia_deduped_pile_idxmaps/pile_0.87_deduped_text_document.idx pythia_pile_idxmaps

# Checksum for final file
echo "Expected checksum: 0cd548efd15974d5cca78f9baddbd59220ca675535dcfc0c350087c79f504693"
sha256sum pythia_pile_idxmaps/pile_0.87_deduped_text_document.bin
```
