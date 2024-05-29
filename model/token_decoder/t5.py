import copy
import warnings
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import T5Stack, get_device_map, assert_device_map

from model.token_decoder.base import BaseTokenDecoder


class T5TokenDecoder(BaseTokenDecoder, T5ForConditionalGeneration):
    def __init__(self, config, block_length, n_embedding_tokens, projection_hidden_size=None,
                 expansion_method=None, expansion_ratio: int = 1, init_expansion=True, decoding_strategy="cross_attention"):
        """
        Adapted from T5ForConditionalGeneration.__init__.
        """
        super(T5ForConditionalGeneration, self).__init__(config)
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        self.encoder = None  # removed

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        if decoding_strategy == "cross_attention":
            print("T5TokenDecoder is using only cross-attention for decoding")
        elif decoding_strategy == "summation":
            print("T5TokenDecoder is using both summation and cross-attention for decoding")
        else:
            raise ValueError("decoding_strategy should be either 'cross_attention' or 'summation'")        

        BaseTokenDecoder.__init__(self, config, block_length, n_embedding_tokens, projection_hidden_size,
                                  expansion_method, expansion_ratio, init_expansion, decoding_strategy)  # added

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        return self.decoder.embed_tokens(input_ids)

    def model_forward(self, inputs_embeds: torch.FloatTensor, attention_mask: torch.LongTensor,
                      past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
                      expanded_block_embeddings: torch.FloatTensor,
                      **kwargs) -> BaseModelOutputWithPastAndCrossAttentions:
        return self.decoder(
            input_ids=None,
            attention_mask=attention_mask,
            # encoder_hidden_states=None,
            # encoder_attention_mask=None,
            inputs_embeds=inputs_embeds,
            # head_mask=None,
            # cross_attn_head_mask=None,
            past_key_values=past_key_values,
            encoder_hidden_states=expanded_block_embeddings,
            # use_cache=use_cache,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
            **kwargs)

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

    # The following methods are overridden from T5ForConditionalGeneration to remove encoder references.

    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5ForConditionalGeneration.parallelize` is deprecated and will be removed in v5 of Transformers, you"
            " should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also"
            " provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'encoder.block.0': 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.decoder.deparallelize()
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)
