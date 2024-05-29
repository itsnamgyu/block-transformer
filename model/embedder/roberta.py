from typing import Optional

import torch
from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings, RobertaEncoder

from model.embedder.base import BaseEmbedder


class RobertaEmbedder(RobertaModel, BaseEmbedder):
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
        super(RobertaModel, self).__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.pooler = None

        if projection_method is None:
            projection_method = "projection_layer"
            print(f'[RobertaEmbedder] projection_method is None, defaulting to "projection_layer"')

        if projection_method == "concat":
            input_tokens_per_block = block_length // n_embedding_tokens
            hidden_size = projection_hidden_size // input_tokens_per_block
            if config.hidden_size is None:
                print(f"Setting hidden_size of embedder to {hidden_size}")
                config.hidden_size = hidden_size
            else:
                if config.hidden_size != hidden_size:
                    raise ValueError(f"config.hidden_size must be equal to projection_hidden_size // "
                                    f"(block_length // n_embedding_tokens), but got {config.hidden_size} and "
                                    f"{hidden_size}")

        BaseEmbedder.__init__(self, config, block_length, n_embedding_tokens, projection_method, projection_hidden_size,
                              init_projection)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
        # token_type_ids: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.Tensor] = None,
        # head_mask: Optional[torch.Tensor] = None,
        # encoder_hidden_states: Optional[torch.Tensor] = None,
        # encoder_attention_mask: Optional[torch.Tensor] = None,
        # past_key_values: Optional[List[torch.FloatTensor]] = None,
        # use_cache: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,
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
