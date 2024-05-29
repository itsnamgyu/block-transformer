from typing import Optional

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from model.embedder.base import BaseEmbedder


class LookupConfig(PretrainedConfig):
    model_type = "lookup"

    def __init__(
        self,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        vocab_size=None,
        hidden_size=None,
        initializer_range=0.02,  # same as RoBERTa and GPTNeoX from transformers==4.36.2
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id,
                         **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range


class LookupEmbedder(PreTrainedModel, BaseEmbedder):
    config_class = LookupConfig

    def __init__(self, config: LookupConfig, block_length=8, n_embedding_tokens=8, projection_method=None,
                 projection_hidden_size=None, init_projection=True):
        if config.vocab_size is None:
            raise ValueError("config.vocab_size must be specified")

        if projection_method is None:
            projection_method = "concat"

        if projection_hidden_size is None:
            raise ValueError("projection_hidden_size must be specified")

        input_tokens_per_block = block_length // n_embedding_tokens
        hidden_size = projection_hidden_size // input_tokens_per_block
        if config.hidden_size is None:
            print(f"Setting hidden_size of embedder to {hidden_size}")
            config.hidden_size = hidden_size
        else:
            if projection_method == "concat" and config.hidden_size != hidden_size:
                raise ValueError(f"config.hidden_size must be equal to projection_hidden_size // "
                                 f"(block_length // n_embedding_tokens), but got {config.hidden_size} and "
                                 f"{hidden_size}")

        super().__init__(config)
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        BaseEmbedder.__init__(self, config, block_length, n_embedding_tokens, projection_method, projection_hidden_size,
                              init_projection)

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            # Same as RoBERTa and GPTNeoX from transformers==4.36.2
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids must be specified")
        input_ids = input_ids.view(-1, input_ids.shape[-1])  # (batch_size, block_length)
        if input_ids.shape[-1] != self.block_length:
            raise ValueError(f"Sequence length must be {self.block_length}, but got {input_ids.shape[-1]}")

        hidden_state = self.embeddings(input_ids)
        assert hidden_state.shape == (input_ids.shape[0], input_ids.shape[1], self.config.hidden_size)
        hidden_state = self.project_hidden_state(hidden_state)

        return hidden_state
