from typing import Optional

import copy
import torch
from torch import nn
from transformers.models.t5.modeling_t5 import T5EncoderModel, T5Stack

from model.embedder.base import BaseEmbedder


class T5Embedder(T5EncoderModel, BaseEmbedder):
    def __init__(self, config, block_length=8, n_embedding_tokens=8, projection_method=None,
                 projection_hidden_size=None, init_projection=True):
        """
        Adapted from RobertaModel.__init__.

        Changes:
        - Add arguments related to last_hidden_state pooling

        - n_embedding_tokens: number of block embedding tokens that comprise a whole block embedding
        - projection_method:
          - "projection_layer": project `block_length` token embeddings to `n_embedding_tokens` embeddings with
            `projection_hidden_size`. `block_length` must be divisible by `n_embedding_tokens`.
          - None: concat all embeddings
        """
        super(T5EncoderModel, self).__init__(config)
        self.config = config

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        if projection_method is None:
            projection_method = "projection_layer"
            print(f'[T5Embedder] projection_method is None, defaulting to "projection_layer"')

        if projection_method == "concat":
            input_tokens_per_block = block_length // n_embedding_tokens
            hidden_size = projection_hidden_size // input_tokens_per_block
            if config.d_model is None:
                print(f"Setting hidden_size of embedder to {hidden_size}")
                config.d_model = hidden_size
            else:
                if config.d_model != hidden_size:
                    raise ValueError(f"config.d_model must be equal to projection_hidden_size // "
                                    f"(block_length // n_embedding_tokens), but got {config.d_model} and "
                                    f"{hidden_size}")

        BaseEmbedder.__init__(self, config, block_length, n_embedding_tokens, projection_method, projection_hidden_size,
                              init_projection)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # head_mask: Optional[torch.FloatTensor] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids must be specified")
        input_ids = input_ids.view(-1, input_ids.shape[-1])  # (batch_size, block_length)
        if input_ids.shape[-1] != self.block_length:
            raise ValueError(f"Sequence length must be {self.block_length}, but got {input_ids.shape[-1]}")

        output = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_state = output.last_hidden_state
        hidden_state = self.project_hidden_state(hidden_state)

        return hidden_state
