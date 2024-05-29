from typing import Optional, Union

import torch

from lm_eval import utils
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import stop_sequences_criteria


@register_model("block", "block_transformer")
class BlockTransformerWrapper(HFLM):
    def __init__(
        self,
        pretrained,
        tokenizer,
        **kwargs,
    ) -> None:
        """
        Mamba (via the `mamba_ssm` package) supports the following args:
        ```
        d_model: int,
        n_layer: int,
        vocab_size: int,
        initializer_cfg=None,
        pad_vocab_size_multiple: int = 1,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        ```

        See https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L175 for more info.
        The above can all be passed via `--model_args` or to this __init__() directly
        but we recommend placing many of these within the config.json file uploaded alongside your
        Mamba model to the HF Hub instead.
        All other HuggingFace from_pretrained() kwargs
        such as those related to
        `parallelize=True`, PEFT, autoGPTQ,
        or any sub-configurations of these advanced args,
        are unsupported by the `mamba_ssm` package.

        The HFLM arguments

        `backend`, `revision`, `subfolder`, `tokenizer`, `truncation`, `max_length`,
        `device`, `dtype`, `batch_size`, `max_batch_size`, `trust_remote_code`, `use_fast_tokenizer`

        Are all supported by Mamba where they do not conflict
        with Mamba-specific restrictions such as causal LMs only.
        """

        if "backend" in kwargs:
            # mamba currently only supports causal models
            assert kwargs["backend"] == "causal"
                
        assert not isinstance(pretrained, str), "pretrained must be BlockTransformer object, not str"
        pretrained.device = kwargs.get("device", "cuda:0")
        pretrained.to(pretrained.device)
        
        super().__init__(
            pretrained=pretrained,
            tokenizer=tokenizer,
            # set appropriate defaults for tokenizer, max length, etc
            backend=kwargs.get("backend", "causal"),
            max_length=kwargs.get("max_length", 2048),
            **kwargs,
        )

    def _model_call(self, inps, attn_mask=None, labels=None):
        """
        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :param attn_mask: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :param labels: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        if self.tokenizer.pad_token_id is not None:
            pad_token_id = self.tokenizer.pad_token_id
        else:
            pad_token_id = self.tokenizer.eos_token_id
        block_length = self.model.token_decoder.block_length
        batch_size, seq_len = inps.shape
        
        # TODO: when length of context is shorter than block_length - 1, we should start | P P P A | for those tasks
        # otherwise, we can start | A B C D | for the first block
        
        # in case of | P P P A |
        eos_index = inps.eq(pad_token_id).float().argmax(dim=-1) 
        
        left_pad_len = block_length - 1  # first block should be [P P P a] for continuation
        left_pad_tokens = torch.ones(batch_size, left_pad_len, dtype=torch.long).to(inps.device) * pad_token_id    
        right_pad_len = block_length - (seq_len + left_pad_len) % block_length
        right_pad_tokens = torch.ones(batch_size, right_pad_len, dtype=torch.long).to(inps.device) * pad_token_id
        
        inps = torch.cat([left_pad_tokens, inps, right_pad_tokens], dim=1)
        assert inps.shape[1] % block_length == 0
        
        # in case of | A B C D |
        # right_pad_len = block_length - seq_len % block_length
        # right_pad_tokens = torch.ones(batch_size, right_pad_len, dtype=torch.long).to(inps.device) * pad_token_id
        
        # inps = torch.cat([inps, right_pad_tokens], dim=1)
        # assert inps.shape[1] % block_length == 0
        
        # # find the first pad_token_id in each sample of the tensor
        # eos_index = inps.eq(pad_token_id).float().argmax(dim=-1)   
        
        inps = inps.reshape(batch_size, -1, block_length)  # -1 denotes n_blocks
        assert attn_mask is None
        attn_mask = torch.where(inps == pad_token_id, 0, 1)
        
        # in case of | P P P A |
        for i, eos_i in enumerate(eos_index):
            if eos_i == 0: # this sample should be a sample with the max length in the batch
                attn_mask[i, -1, -right_pad_len] = 1
            else:  
                _eos_i = left_pad_len + eos_i
                block_i, pos_i = divmod(_eos_i.item(), block_length)
                attn_mask[i, block_i, pos_i] = 1
        block_attn_mask = attn_mask.any(dim=-1).long()     
        
        # in case of | A B C D |
        # for i, eos_i in enumerate(eos_index):
        #     if inps[0, 0, 0] == pad_token_id:  # starts with EOS token
        #         attn_mask[i, -1, -right_pad_len] = 1
        #     else:
        #         block_i, pos_i = divmod(eos_i.item(), block_length)
        #         attn_mask[i, block_i, pos_i] = 1
        # block_attn_mask = attn_mask.any(dim=-1).long() 
        
        with torch.no_grad():
            logits = self.model(
                input_ids=inps, 
                attention_mask=attn_mask, 
                block_attention_mask=block_attn_mask, 
                skip_padding_blocks=False
            )["logits"]
        logits = logits.reshape(batch_size, -1, self.model.config.vocab_size)
        
        # align to the required sequence length
        # in case of | P P P A |
        logits = logits[:, :seq_len, :]
        
        # in case of | A B C D |
        # logits = logits[:, :seq_len - (block_length - 1), :]
        
        return logits