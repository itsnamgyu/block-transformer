from typing import Optional, Tuple

import torch
from torch import nn
from transformers import GPTNeoForCausalLM, GPTNeoModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from model.token_decoder.base import BaseTokenDecoder


class GPTNeoTokenDecoder(BaseTokenDecoder, GPTNeoForCausalLM):
    def __init__(self, config, block_length, n_embedding_tokens=None, projection_hidden_size=None,
                 expansion_method=None, expansion_ratio: int = 1, init_expansion=True, decoding_strategy=None):
        """
        Adapted from GPTNeoForCausalLM.__init__.
        """
        super(GPTNeoForCausalLM, self).__init__(config)
        self.config = config  # added

        self.transformer = GPTNeoModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        BaseTokenDecoder.__init__(self, config, block_length, n_embedding_tokens, projection_hidden_size,
                                  expansion_method, expansion_ratio, init_expansion, decoding_strategy)  # added

        # Initialize weights and apply final processing
        self.post_init()

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        return self.transformer.wte(input_ids)

    def model_forward(self, inputs_embeds: torch.FloatTensor, attention_mask: torch.LongTensor,
                      past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
                      expanded_block_embeddings: torch.FloatTensor,
                      **kwargs) -> BaseModelOutputWithPastAndCrossAttentions:
        return self.transformer(
            input_ids=None,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            # head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            # use_cache=use_cache,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        return self.lm_head(hidden_states)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """
        Check for and pass `expanded_block_embeddings` argument to model input
        """
        if "block_embeddings" in kwargs:
            raise KeyError("you should pass pre-expanded `expanded_block_embeddings` to generate() instead of "
                           "`block_embeddings` to prevent redundant computation. Refer to forward() for more details")
        if "expanded_block_embeddings" not in kwargs:
            raise KeyError("expanded_block_embeddings must be specified for token_decoder generation")

        expanded_block_embeddings = kwargs.pop("expanded_block_embeddings")
        ret = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, **kwargs)
        ret["expanded_block_embeddings"] = expanded_block_embeddings

        return ret
