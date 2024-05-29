from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import GPTNeoXForCausalLM, GPTNeoXModel, GPTNeoXLayer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, GPTNeoXFlashAttention2

from model.block_decoder.base import BaseBlockDecoder


class BlockGPTNeoXAttention(GPTNeoXAttention):
    def __init__(self, config, n_embedding_tokens):
        super().__init__(config)
        # modified (bias):
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool))[
            n_embedding_tokens - 1::n_embedding_tokens].view(
                1, 1, -1, max_positions
            ).repeat_interleave(n_embedding_tokens, dim=2),
            persistent=False,
        )


class BlockGPTNeoXFlashAttention2(GPTNeoXFlashAttention2):
    def __init__(self, config, n_embedding_tokens):
        super().__init__(config)
        assert n_embedding_tokens == 1, "n_embedding_tokens must be 1 for BlockGPTNeoXFlashAttention2"


GPT_NEOX_ATTENTION_CLASSES = {
    "eager": BlockGPTNeoXAttention,
    "flash_attention_2": BlockGPTNeoXFlashAttention2,
}


class BlockGPTNeoXLayer(GPTNeoXLayer):
    def __init__(self, config, n_embedding_tokens):
        super().__init__(config)
        attention_class = GPT_NEOX_ATTENTION_CLASSES[config._attn_implementation]
        self.attention = attention_class(config, n_embedding_tokens=n_embedding_tokens)  # modified


class BlockGPTNeoXModel(GPTNeoXModel):
    def __init__(self, config, n_embedding_tokens):
        super().__init__(config)
        # modified
        self.layers = nn.ModuleList(
            [BlockGPTNeoXLayer(config, n_embedding_tokens=n_embedding_tokens) for _ in range(config.num_hidden_layers)])
        self.post_init()
        # self.embed_in = None  # removed


class GPTNeoXBlockDecoder(GPTNeoXForCausalLM, BaseBlockDecoder):
    def __init__(self, config, n_embedding_tokens=8, use_block_decoding_loss=False, block_decoding_loss_type="contrastive"):
        """
        Adapted from GPTNeoXForCausalLM.__init__.
        """
        super(GPTNeoXForCausalLM, self).__init__(config)
        self.gpt_neox = BlockGPTNeoXModel(config, n_embedding_tokens=n_embedding_tokens)  # modified
        self.embed_out = None  # removed

        # Added for GPTNeoXBlockDecoder
        BaseBlockDecoder.__init__(self, config.hidden_size, n_embedding_tokens, use_block_decoding_loss, block_decoding_loss_type)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        block_attention_mask: Optional[torch.Tensor] = None,  # replaces attention_mask
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Adapted from GPTNeoXForCausalLM.forward(). Reuses code from GPTNeoBlockDecoder.forward().

        Changes:
        - Adapt block_attention_mask
        - Use shifted inputs_embeds instead of labels (ids)
        - Use block-wise loss (e.g., MSE) instead of cross-entropy
        """
        if input_ids is not None:
            raise ValueError("block decoders do not use input_ids")
        if labels is not None:
            raise ValueError("block decoders do not use labels")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        attention_mask = block_attention_mask.repeat_interleave(self.n_embedding_tokens, dim=1)
        outputs = self.gpt_neox(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        loss = self.compute_loss(hidden_states, attention_mask, inputs_embeds)  # added

        if not return_dict:
            output = (hidden_states,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )
