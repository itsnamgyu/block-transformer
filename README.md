# <img src="https://github.com/raymin0223/fast_robust_early_exit/assets/50742281/c57b9908-72b7-4b80-bbd2-65e7dfec22e6" width="30" height="30"> Block Transformer: Global-to-Local Language Modeling for Fast Inference (NeurIPS 2024)

<a href="https://arxiv.org/abs/2406.02657"><img src="https://img.shields.io/badge/Paper-arXiv:2406.02657-Green"></a>
<a href=#bibtex><img src="https://img.shields.io/badge/Paper-BibTex-yellow"></a>

**Namgyu Ho<sup>1,2&dagger;\*</sup> &nbsp; Sangmin Bae<sup>1\*</sup> &nbsp; Taehyeon Kim<sup>1</sup> &nbsp; Hyunjik Jo<sup>2</sup> &nbsp; Yireun Kim<sup>2</sup> &nbsp; Tal Schuster<sup>3</sup> &nbsp; Adam Fisch<sup>3</sup>    
James Thorne<sup>1&ddagger;</sup> &nbsp; Se-Young Yun<sup>1&ddagger;</sup>**   
<sup>**1**</sup>KAIST AI &nbsp; <sup>**2**</sup>LG AI Research &nbsp; <sup>**3**</sup>Google DeepMind &nbsp;    
&dagger;Work done during an internship at LG AI Research. &nbsp; \*Equal contribution. &nbsp; &ddagger;Corresponding authors.


<p align="left">
<img width="700" src="https://github.com/itsnamgyu/block-transformer/assets/50742281/b6670d09-ee5a-4dc0-a582-f215e1248af1">
</p>

- We propose **Block Transformer** architecture which adopts hierarchical global-to-local language modeling to autoregressive transformers to mitigate inference bottlenecks of self-attention.
- Block Transformer models global dependencies through _self-attention between coarse blocks_ at lower layers (in block decoder), and _decodes fine-grained tokens within each local block_ at upper layers (in token decoder).
- We leverage inference-time benefits of both global and local modules, achieving **10-20x gains in throughput** compared to vanilla transformers with equivalent perplexity.

## ⚡️ Real-World Decoding Speed Comparison

[https://www.youtube.com/watch?v=c0D7EvffYnU](https://youtu.be/9k9n0RkPBCI?feature=shared)

<div align="left">
      <a href="https://youtu.be/9k9n0RkPBCI?feature=shared">
         <img src="https://img.youtube.com/vi/9k9n0RkPBCI/0.jpg" style="width:25%;">
      </a>
</div>

# 🚀 Getting Started

To try out our pretrained Block Transformer models, install requirements and download our pretrained checkpoints (see sections below).

Note, make sure to run the following command before running any code to support absolute imports.

```
python setup.py develop
```

### Inference with Custom Prompts

Use our demo notebook at `./notebooks/inference_demo.ipynb`.

### Batch Inference Speed Demo

```
CUDA_VISIBLE_DEVICES=0 python inference_demo.py --model=block_main_b4_1.2b --batch_size=128
```

# 💎 Pretrained Checkpoints

We share all checkpoints of our main models, pretrained on tens of thousands of A100 hours. With ❤️ from LG AI
Research.

- [Dropbox](https://www.dropbox.com/scl/fo/7l7ga148gcc39ykdcqndd/AFw4Fk1XbaABFc-IuAzoyQc?rlkey=zn6cqua65wj835kkifbolr4dj&st=qcjfy53c&dl=0)
- [Google Drive](https://drive.google.com/drive/folders/1NjdYIBEsgnEWe4TgGvyJC2emOVc8i4Bw?usp=share_link)

To use our code as-is, unzip the checkpoints into the `./results` directory, as shown below.

```
block-transformer/
|-- results/
  |-- block_main_b4_1.2b/
    |-- checkpoint-570000/
      |-- model.safetensors
  |-- ...
```

# 💻 Requirements

Refer to `requirements.txt`.

Note, make sure to run the following command before running any code to support absolute imports.

```
python setup.py develop
```

### Transformers version

Our subclasses of GPTNeoX models for Block Transformer have been tested under

```
transformers==4.39.3
accelerate==0.33.0
```

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

# 📚 Pretraining

- Vanilla (HuggingFace) model training: `pretrain_vanilla_transformer.py`
    ```bash
    deepspeed --include localhost:0,1,2,3 --no_local_rank --master_port 29540 pretrain_vanilla_transformer.py --config-name vanilla_31 pythia_pile_idxmaps_path=/path/to/pythia_pile_idxmaps
    ```

- Block transformer training: `pretrain_block_transformer.py`
  ```bash
    deepspeed --include localhost:0,1,2,3 --no_local_rank --master_port 29540 pretrain_block_transformer.py --config-name block_main_b4_5 pythia_pile_idxmaps_path=/path/to/pythia_pile_idxmaps
    ```

- Using the `torch.distributed` launcher
    ```bash
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port=29540
    ```
    - Note that this still uses deepspeed optimization. To run without deepspeed optimization,
      append `--deepspeed=null`.

# 🔬 Evaluation

- Zero-shot evaluation: `eval_zero_shot_task.py`
    ```bash
    CUDA_VISIBLE_DEVICES=0 python eval_zero_shot_task.py --config-name=eval_multiple_ckpt configs.hf=["vanilla_31"] batch_size=64
    CUDA_VISIBLE_DEVICES=0 python eval_zero_shot_task.py --config-name=eval_multiple_ckpt configs.block=["block_main_b4_5"] batch_size=64
    ```
  
- Inference throughput wall-time measurement: `measure_generation_time.py`
    ```bash
    CUDA_VISIBLE_DEVICES=0 python measure_generation_time.py --config-name=block_main_b4_5 ++benchmark_prefill_length=2048 ++benchmark_decode_length=128
    CUDA_VISIBLE_DEVICES=0 python measure_generation_time.py --config-name=block_main_b4_5 ++benchmark_prefill_length=128 ++benchmark_decode_length=2048
    ```
  - Works for both HF and block models.
  - By default, batch size is auto-tuned via binary search to maximize VRAM utilization.To set a specific batch size,
    use `++batch_size=64`.

# 📑 Pretraining Data Preparation

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

## :star2: BibTeX

```
@article{ho2024block,
  title={Block Transformer: Global-to-Local Language Modeling for Fast Inference},
  author={Ho, Namgyu and Bae, Sangmin and Kim, Taehyeon and Jo, Hyunjik and Kim, Yireun and Schuster, Tal and Fisch, Adam and Thorne, James and Yun, Se-Young},
  journal={arXiv preprint arXiv:2406.02657},
  year={2024}
}
```
