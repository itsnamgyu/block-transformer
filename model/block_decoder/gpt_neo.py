from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import GPTNeoForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, CausalLMOutputWithPast
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoSelfAttention, GPTNeoAttention, GPTNeoBlock, GPTNeoModel

from model.block_decoder.base import BaseBlockDecoder


class BlockGPTNeoSelfAttention(GPTNeoSelfAttention):
    def __init__(self, config, attention_type, n_embedding_tokens=8):
        super().__init__(config, attention_type)

        # modified (bias):
        max_positions = config.max_position_embeddings
        bias = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool))[
               n_embedding_tokens - 1::n_embedding_tokens].view(1, 1, -1, max_positions)
        bias = bias.repeat_interleave(n_embedding_tokens, dim=2)
        if attention_type == "local":
            bias = torch.bitwise_xor(bias, torch.tril(bias, -config.window_size))
        self.register_buffer("bias", bias, persistent=False)


class BlockGPTNeoAttention(GPTNeoAttention):
    def __init__(self, config, layer_id=0, n_embedding_tokens=8):
        super().__init__(config, layer_id=layer_id)
        # modified
        self.attention = BlockGPTNeoSelfAttention(config, self.attention_type, n_embedding_tokens=n_embedding_tokens)


class BlockGPTNeoBlock(GPTNeoBlock):
    def __init__(self, config, layer_id, n_embedding_tokens=8):
        """
        Adapted from GPTNeoBlock.__init__.
        """
        super().__init__(config, layer_id)
        self.attn = BlockGPTNeoAttention(config, layer_id, n_embedding_tokens=n_embedding_tokens)  # modified


class BlockGPTNeoModel(GPTNeoModel):
    def __init__(self, config, n_embedding_tokens=8):
        """
        Adapted from GPTNeoModel.__init__.
        """
        super().__init__(config)
        # modified
        self.h = nn.ModuleList([BlockGPTNeoBlock(config, layer_id=i, n_embedding_tokens=n_embedding_tokens) for i in
                                range(config.num_layers)])
        self.post_init()
        # self.wte = None  # removed


class GPTNeoBlockDecoder(GPTNeoForCausalLM, BaseBlockDecoder):
    def __init__(self, config, n_embedding_tokens=8, use_block_decoding_loss=False, block_decoding_loss_type="contrastive"):
        """
        Adapted from GPTNeoForCausalLM.__init__.
        """
        super(GPTNeoForCausalLM, self).__init__(config)
        self.transformer = BlockGPTNeoModel(config, n_embedding_tokens=n_embedding_tokens)  # modified
        self.lm_head = None  # removed

        # Added for GPTNeoBlockDecoder
        BaseBlockDecoder.__init__(self, config.hidden_size, n_embedding_tokens, use_block_decoding_loss, block_decoding_loss_type)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        block_attention_mask: Optional[torch.Tensor] = None,  # replaces attention_mask
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        """
        Adapted from GPTNeoForCausalLM.forward().

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
        transformer_outputs = self.transformer(
            input_ids=None,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        loss = self.compute_loss(hidden_states, attention_mask, inputs_embeds)  # added

        if not return_dict:
            output = (hidden_states,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=transformer_outputs.attentions,
        )
