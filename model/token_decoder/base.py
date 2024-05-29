import warnings
from abc import abstractmethod, ABCMeta
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithPast


class BaseTokenDecoder(metaclass=ABCMeta):
    """
    Abstract class for token decoder models
    """
    loss_by_position: torch.Tensor  # for use in logging callback

    def __init__(self, config, block_length, n_embedding_tokens=None, projection_hidden_size=None,
                 expansion_method=None, expansion_ratio: int = None, init_expansion=True, decoding_strategy=None):
        """
        Token decoders accept block embeddings as input and replaces the first N=`n_embedding_tokens` input tokens with
        projections of the block embeddings. The first N input tokens (to be replaced) must be BOS.

        - projection_hidden_size: hidden size of each block embedding token (hidden size of block decoder)
        - hidden_size: hidden size of token decoder
        - expansion_ratio: expand n_embedding_tokens by this factor
        - expansion_method:
            - "expansion_layer": use Conv1d to change hidden_size
            - None: keep n_embedding_tokens and hidden_size the same
        """
        self.config = config
        self.block_length = block_length
        self.n_embedding_tokens = n_embedding_tokens
        self.expansion_ratio = expansion_ratio
        if self.expansion_ratio is None:
            if decoding_strategy in ["summation", "cross_attention"]:
                print(f"Setting token decoder expansion_ratio to block_length (default for {decoding_strategy} "
                      f"strategy)")
                self.expansion_ratio = block_length
            elif decoding_strategy == "prefix":
                # You should probably have a big enough expansion ratio so that there is no bottleneck in passing
                # information from the block embeddings to the token decoder, i.e., so that
                # hidden_size x expansion_ratio >= block embedding size
                # This depends on the block embedding size and the hidden size of the token decoder
                raise ValueError(f"`expansion_ratio` must be specified for decoding_strategy='prefix'")
            else:
                raise ValueError(f"Invalid decoding_strategy {decoding_strategy}")  # possibly a NotImplementedError
        self.n_expanded_emb = self.n_embedding_tokens * self.expansion_ratio
        self.hidden_size = config.hidden_size
        self.projection_hidden_size = projection_hidden_size
        self.expansion_method = expansion_method
        self.decoding_strategy = decoding_strategy

        if self.config.bos_token_id is None:
            self.config.bos_token_id = self.config.eos_token_id

        if self.decoding_strategy == "summation":
            assert self.n_expanded_emb == self.block_length, \
                f"n_expanded_emb ({self.n_expanded_emb}) must be equal to block_length ({self.block_length}) " \
                f"for decoding_strategy='summation'"

        if self.expansion_method == "expansion_layer":
            self.expansion_layer = nn.Conv1d(in_channels=self.projection_hidden_size,
                                             out_channels=self.hidden_size * self.expansion_ratio,
                                             kernel_size=1, stride=1, padding=0)
            if init_expansion:
                self._init_expansion_layer()
            else:
                warnings.warn("[BaseTokenDecoder] `init_expansion=False`")
        else:
            self.expansion_layer = None

        # for generation
        if config.pad_token_id is None:
            config.pad_token_id = config.eos_token_id
        # to turn off warning errors, modify `~/anaconda3/envs/[env_name]/lib/python3.9/site-packages/transformers/generation/utils.py`

    @abstractmethod
    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """
        :param input_ids: [batch_size, length]
        :return: torch.FloatTensor[batch_size, length, hidden_size]

        Note, 1 <= `length` <= `block_length`
        """
        raise NotImplementedError("get_embedding_layer() must be implemented in subclass")

    @abstractmethod
    def model_forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        labels: Optional[torch.LongTensor],
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
        expanded_block_embeddings: torch.FloatTensor,
        **kwargs,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        """
        :param expanded_block_embeddings:
        :param input_embeds: [batch_size, length, hidden_size]
        :param attention_mask: [batch_size, length]
        :param labels: [batch_size, length]
        :return: torch.FloatTensor[batch_size, length, vocab_size]

        Note, 1 <= `length` <= `block_length`
        """
        raise NotImplementedError("model_forward() must be implemented in subclass")

    @abstractmethod
    def compute_logits(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        """
        :param hidden_states: [batch_size, length, hidden_size]
        :return: torch.FloatTensor[batch_size, length, vocab_size]

        Note, 1 <= `length` <= `block_length`
        """
        raise NotImplementedError("compute_logits() must be implemented in subclass")

    def expand_block_embeddings(self, block_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Expand block embeddings to match the number of tokens in the input sequence

        - block_embeddings: (batch_size, n_embedding_tokens, projection_hidden_size)
        - return: (batch_size, block_length, hidden_size)
        """
        if self.expansion_method == "expansion_layer":
            block_embeddings = block_embeddings.transpose(1, 2)
            # -> (batch_size, projection_hidden_size, n_embedding_tokens)
            block_embeddings = self.expansion_layer(block_embeddings)
            # -> (batch_size, hidden_size * expansion_ratio, n_embedding_tokens)
            block_embeddings = block_embeddings.transpose(1, 2)
            # -> (batch_size, n_embedding_tokens, hidden_size * expansion_ratio)
            block_embeddings = block_embeddings.reshape(block_embeddings.shape[0], -1, self.hidden_size)
            # -> (batch_size, n_embedding_tokens * expansion_ratio, hidden_size)
        elif self.expansion_method is None:
            block_embeddings = block_embeddings.repeat_interleave(self.expansion_ratio, dim=1)
        else:
            raise ValueError(f"expansion_method must be one of ['expansion_layer', None], got {self.expansion_method}")

        return block_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        block_embeddings: torch.Tensor = None,  # added
        # position_ids: Optional[torch.LongTensor] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None,
        # head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        # use_cache: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        expanded_block_embeddings: Optional[torch.FloatTensor] = None,  # added
        discard_redundant_tokens: bool = False,
        log_loss_by_position=True,
        **kwargs
    ):
        """
        Perform token_decoder forward computation. This function internally handles all the input adaptation for
        different decoding strategies. You can treat this function similar to the `forward()` function of the
        T5 decoder, with inputs in the form of [BOS, A, B, C, ...] and outputs in the form of [A^, B^, C^, D^, ...].

        `past_key_values` are also handled appropriately for generation. Note that `past_key_values` follow the
        internal *adapted* input format for each decoding strategy, which may have different prefix lengths.

        Parameters:
            input_ids:
                [[BOS   A     B     C     D    ], ...]  # given block_length=4
            labels:
                [[-100  A     B     C     D    ], ...]
            attention_mask:
                [[1     a     b     c     d    ], ...]   # the first value must be set to 1 (see summation strategy)
            expanded_block_embeddings: [batch_size, n_expanded_dim, hidden_size]
                Pre-expanded block embeddings to be used for generation. Without pre-expansion, the block_embedding
                needs to be re-expanded at each decoding step for some strategies (e.g., summation), leading to
                redundant computation.
            past_key_values:
                Cached key values from previous forward() call. This is used for generation. When past_key_values is
                set, the input_ids must only contain new tokens (this is optional in the original HF implementation,
                but we made it mandatory here).
            discard_redundant_tokens:
                For loss and forward logit computation, we discard all penultimate input ids (and the last input id).
                This involves (1) removing the final column of input_ids and attention_mask entirely, and (2) mask out
                all padding tokens and final non-padding tokens from logit computation. Masked out output logits
                are filled with zeros.
                E.g., given input_ids = [BOS, A, B, PAD, PAD] we only need logits [A^ B^]. Therefore, outputs from B,
                PAD, PAD can be excluded from logit computation.
                This should not be used for generation, because we need the final logits.
            log_loss_by_position:

        Note that, because bos_token_id is not typically defined for decoder-only models, we assign eos_token_id to
        bos_token_id (if None) in `BaseTokenDecoder.__init__`. This is fine because there is no other use for EOS in
        token decoders.

        ## Usage

        1. For loss computation
        ```
        input_ids = torch.LongTensor([[BOS, A, B, C, D], ...])
        attention_mask = torch.FloatTensor([[1, a, b, c, d], ...])
        labels = torch.LongTensor([[-100, A, B, C, D], ...])
        ...
        model(input_ids=input_ids, attention_mask=attention_mask, block_embeddings=block_embeddings, labels=labels,
              discard_last_input_id=True)
        ```

        2. For generation
        ```
        input_ids: torch.LongTensor([[BOS], ...])
        attention_mask: torch.FloatTensor([[1], ...])
        expanded_block_embeddings = model.expand_block_embeddings(block_embeddings)
        model.generate(input_ids=input_ids, attention_mask=attention_mask,
                       expanded_block_embeddings=expanded_block_embeddings)
        ```

        ## Implementation Notes

        Internally, the `input_ids`, `labels`, and `attention_mask` will be adapted according to the
        `decoding strategy`. Refer to the `_adapt_forward_inputs_for_<STRATEGY>` methods for more details.

        Note that past_key_values originates from the output of the previous forward() call, which internally called
        `model_forward() using the *adapted* inputs. Therefore, past_key_values[0][0].shape[2] is the length
        of the previous *adapted* input_ids which we name `past_adapted_length`.
        """
        # check inputs and reshape inputs such that dim 0 is batch_size
        # check input_ids
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        bs, original_length = input_ids.shape
        if past_key_values is None and (input_ids[:, 0] != self.config.bos_token_id).any():
            raise ValueError(f"input_ids must start with BOS token {self.config.bos_token_id}, got "
                             f"{input_ids[:, 0]}. Check the docstrings")
        if self.block_length is not None and input_ids.shape[1] > self.block_length + 1:
            warnings.warn(f"input_ids.shape[1] should be <= block_length + 1, got {input_ids.shape[1]}. "
                          f"You are attempting to infer beyond `block_length` on which the model was trained.")
        # check attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            if (attention_mask[:, 0] != 1).any():
                raise ValueError("attention_mask[:, 0] must be 1")
        # check labels
        if labels is not None:
            labels = labels.view(-1, labels.shape[-1])
            if (labels[:, 0] != -100).any():
                raise ValueError(f"labels must start with -100, got {labels[:, 0]}. Check the docstrings")
        # check past_key_values
        if past_key_values is not None:
            n_heads = self.config.num_attention_heads
            d_head = self.config.hidden_size // n_heads
            past_adapted_length = past_key_values[0][0].shape[2]
            assert past_key_values[0][0].shape == (bs, n_heads, past_adapted_length, d_head)
        else:
            past_adapted_length = 0
        if past_key_values is not None and labels is not None:
            raise NotImplementedError("We don't support past_key_values and labels at the same time.")
        # check block_embeddings, expanded_block_embeddings
        if sum([block_embeddings is None, expanded_block_embeddings is None]) != 1:
            raise ValueError("block_embeddings xor expanded_block_embeddings must be specified")
        if block_embeddings is None:
            if expanded_block_embeddings.shape != (bs, self.n_expanded_emb, self.hidden_size):
                raise ValueError(f"expanded_block_embeddings must be of shape "
                                 f"(batch_size, n_expanded_emb, hidden_size) = "
                                 f"({bs}, {self.n_expanded_emb}, {self.hidden_size}), "
                                 f"got {expanded_block_embeddings.shape}")
        else:
            if past_key_values is not None:
                # this is not critical, but we want to avoid redundant computation
                raise ValueError("you should pass the pre-expanded `expanded_block_embeddings` instead of "
                                 "`block_embeddings` to prevent redundant computation during generation. "
                                 "past_key_values is typically used for generation.")
            expanded_block_embeddings = self.expand_block_embeddings(block_embeddings)

        if discard_redundant_tokens:
            content_mask = input_ids != self.config.pad_token_id
            if attention_mask is not None:
                # eos tokens (pad tokens where attention_mask == 1) should be included as content
                content_mask = content_mask | (attention_mask == 1)
            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]
            redundant_token_mask = ~content_mask[:, 1:]

        # adapt input_ids and attention_mask according to the decoding strategy.
        # this will involve modifying the prefix and/or adding in block embeddings.
        # note that the prefix length should not be modified when past_key_values is used.
        # input_ids are converted to inputs_embeds at this stage.
        adaptation_function = {
            "prefix": self._adapt_forward_inputs_for_prefix,
            "summation": self._adapt_forward_inputs_for_summation,
            "cross_attention": self._adapt_forward_inputs_for_cross_attention,
        }[self.decoding_strategy]
        inputs_embeds, attention_mask = adaptation_function(
            input_ids, attention_mask, expanded_block_embeddings, past_adapted_length)

        # the following implementation of forward and loss computation is based on `GPTNeoXForCausalLM.forward()`
        # (but highly optimized). The logic should be identical for other models.

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model_forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                     past_key_values=past_key_values,
                                     expanded_block_embeddings=expanded_block_embeddings, **kwargs)
        hidden_states = outputs[0]
        from model.token_decoder import T5TokenDecoder
        if isinstance(self, T5TokenDecoder) and self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            hidden_states = hidden_states * (self.model_dim ** -0.5)

        # determine redundant output prefix length as a result of the `adaptation_function` above
        # (we want to reverse the adaptation for output)
        # e.g., if adapted_prefix_length == 3, then
        # input_embeds:         [_  _  _  A  B  C  ] <- original input was [_  A  B  C  ]
        # outputs.lm_logits:    [_  _  A^ B^ C^ D^ ] -> output should be   [A^ B^ C^ D^ ]
        # loss_per_position:    [_  _  A^ B^ C^ D^ ] -> output should be   [A^ B^ C^ D^ ]
        # thus, we should discard the redundant output prefix of length `adapted_prefix_length` - 1
        # note that extra prefixes may be required during the forward pass to propagate information from
        # the block embeddings (for some strategies, such as prefix), but their output states are not required for loss
        # computation or final output.
        if past_adapted_length == 0:
            # first decoding step or loss computation
            adapted_prefix_length = {
                "prefix": self.n_expanded_emb,
                "summation": 1,
                "cross_attention": 1,
            }[self.decoding_strategy]
            redundant_output_prefix_length = adapted_prefix_length - 1
            hidden_states = hidden_states[:, redundant_output_prefix_length:, :]

        # shift labels
        if labels is not None:
            labels = labels[:, 1:]

        # discard redundant tokens from hidden_states and labels.
        # hidden_states and labels are flattened at this stage
        if discard_redundant_tokens:
            assert bs == hidden_states.shape[0]
            output_length = hidden_states.shape[1]
            hidden_states = hidden_states[~redundant_token_mask]
            if labels is not None:
                labels = labels[~redundant_token_mask]
                assert labels.shape[0] == hidden_states.shape[0]
        else:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        assert len(hidden_states.shape) == 2
        assert hidden_states.shape[1] == self.hidden_size
        lm_logits = self.compute_logits(hidden_states.contiguous())

        lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction="none")
            lm_losses = loss_fct(lm_logits.contiguous(), labels.contiguous().view(-1))
            lm_loss = lm_losses.mean()

            # reshape original loss to log loss by position
            if log_loss_by_position:
                with torch.no_grad():
                    # We accumulate in float64 because the default dtype (bfloat16) has very inaccurate accumulation.
                    # Note that we use bf16 training with deepspeed in our experiments (as of March 2024).
                    if discard_redundant_tokens:
                        lm_loss_reshaped = lm_losses.new_full((bs, output_length), float('nan'), dtype=torch.float64)
                        lm_loss_reshaped[~redundant_token_mask] = lm_losses.detach().double()
                    else:
                        lm_loss_reshaped = lm_losses.view(bs, -1).detach().double()
                    if hasattr(self, "loss_by_position"):
                        self.loss_by_position += lm_loss_reshaped.nanmean(dim=0)
                        self.loss_by_position_count += 1
                    else:
                        self.loss_by_position = lm_loss_reshaped.nanmean(dim=0)
                        self.loss_by_position_count = 1

        if labels is None:
            if discard_redundant_tokens:
                # reshape original logits to return to caller
                reshaped_lm_logits = lm_logits.new_zeros((bs, output_length, lm_logits.shape[-1],))
                reshaped_lm_logits[~redundant_token_mask] = lm_logits.detach()
                lm_logits = reshaped_lm_logits
            else:
                lm_logits = lm_logits.view(bs, -1, lm_logits.shape[-1]).detach()
        else:
            # there is likely no use case to return logits when labels is not None
            # discard lm_logits to save VRAM
            lm_logits = None

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        block_embeddings: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Parameters:
            input_ids:
                [[EOS   (A     B     C     )], ...]
            block_embeddings: torch.Tensor[(batch_size), n_embedding_tokens, projection_hidden_size]
                Required
            attention_mask:
                [[1     (a     b     c     )], ...]  # note, the 1 at the beginning is not used

        Refer to forward() for more details.
        """
        if block_embeddings is None:
            raise ValueError("block_embeddings must be specified for generation with token decoder")
        block_embeddings = block_embeddings.view(-1, block_embeddings.shape[-2], block_embeddings.shape[-1])
        bs = block_embeddings.shape[0]
        if input_ids is None:
            input_ids = block_embeddings.new_full((bs, 1), self.config.eos_token_id, dtype=torch.long)
            if attention_mask is not None:
                # why?
                raise ValueError("attention_mask must be None when input_ids is None")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # make tensors 2d [batch_size, length]
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        if max_new_tokens:
            max_length = input_ids.shape[1] + max_new_tokens
        if max_length is None:
            max_length = self.block_length + 1
        else:
            if max_length > self.block_length + 1:
                warnings.warn(f"max_length or input length + max_new_tokens  should be <= block_length + 1, got "
                              f"{max_length}. You are attempting to generate more tokens than the block_length on "
                              f"which the model was trained")
        if (input_ids[:, 0] != self.config.eos_token_id).any():
            raise ValueError(f"input_ids must start with EOS token {self.config.eos_token_id}, got "
                             f"{input_ids[:, 0]}. Check the docstrings")

        expanded_block_embeddings = self.expand_block_embeddings(block_embeddings)
        return super().generate(input_ids=input_ids, attention_mask=attention_mask,
                                expanded_block_embeddings=expanded_block_embeddings,
                                discard_redundant_tokens=False,
                                max_length=max_length, **kwargs)

    def _init_expansion_layer(self):
        factor = 1.0  # TODO
        self.expansion_layer.weight.data.normal_(mean=0.0, std=factor * ((self.projection_hidden_size) ** -0.5))
        if hasattr(self.expansion_layer, "bias") and self.expansion_layer.bias is not None:
            self.expansion_layer.bias.data.zero_()

    def _adapt_forward_inputs_for_prefix(self, input_ids, attention_mask, expanded_block_embeddings,
                                         past_adapted_length):
        """
        Modify the prefix to match `n_expanded_emb`, using the block embeddings as the input embedding prefix.

        Parameters:
            input_ids: input_ids (after discarding last token for optimal loss computation, when labels are given)
                [[BOS   A     B     C     ...  ],]
            attention_mask:
                [[1     a     b     c     ...  ],]
            past_adapted_length: length of the *adapted* inputs_embeds in the previous forward call, i.e.,
                past_key_values[0][0].shape[2].

        Returns: (inputs_embeds, attention_mask, labels)
            inputs_embeds: [
                [BE1   BE2   A     B     C     ...  ],
            ],
            labels: [
                [-100  -100  A     B     C     D    ...  ],
            ]
            attention_mask: [
                [1     1     a     b     c     ...  ],
            ]
        where BE == expanded_block_embeddings with BE.shape[1] == n_expanded_emb, let n_expanded_emb=2.

        When past_key_values is not None (past_adapted_lengths != 0), input_ids must only contain new tokens.
        Then, we do not worry about the prefix of `input_ids`.
        """
        # remove given prefix eos
        if past_adapted_length == 0:
            input_ids = input_ids[:, 1:]
        attention_mask = attention_mask[:, 1:]

        # add prefix (block embeddings)
        inputs_embeds = self.embed_input_ids(input_ids)
        if past_adapted_length == 0:
            inputs_embeds = torch.cat([expanded_block_embeddings, inputs_embeds], dim=1)
        attention_mask_prefix = torch.ones((input_ids.shape[0], self.n_expanded_emb), dtype=torch.long,
                                           device=input_ids.device)
        attention_mask = torch.cat([attention_mask_prefix, attention_mask], dim=1)

        return inputs_embeds, attention_mask

    def _adapt_forward_inputs_for_summation(self, input_ids, attention_mask, expanded_block_embeddings,
                                            past_adapted_length):
        """
        Maintain the original prefix and add block embeddings to the input embeddings, aligned by position.

        Parameters:
            input_ids: input_ids (after discarding last token for optimal loss computation, when labels are given)
                [[BOS   A     B     C     ],]
            attention_mask:
                [[1     a     b     c     ],]
            past_adapted_length: length of the *adapted* inputs_embeds in the previous forward call, i.e.,
                past_key_values[0][0].shape[2].

        Returns: (inputs_embeds, attention_mask, labels)
            input_ids: [
                [BOS   A     B     C     ] + \
                [BE1   BE2   BE3   BE4   ],
            ]
        where BE == expanded_block_embeddings with BE.shape[1] == n_expanded_emb, let n_expanded_emb=4.

        When past_key_values is not None (past_adapted_lengths != 0), input_ids must only contain new tokens.
        Then, the `expanded_block_embeddings` are partially added to the input embeddings based on the position
        of the new tokens, which is determined by the length of the past_key_values (`past_adapted_length`).
        """
        if input_ids.shape[1] > self.block_length:
            raise ValueError(f"input_ids.shape[1] must be <= block_length, got {input_ids.shape[1]}"
                             f"You can only generate up to `block_length` tokens using this strategy")
        inputs_embeds = self.embed_input_ids(input_ids)
        # add prefix block embeddings while considering the length of past_key_values if applicable
        start, end = past_adapted_length, past_adapted_length + inputs_embeds.shape[1]
        inputs_embeds = inputs_embeds + expanded_block_embeddings[:, start:end, :]

        return inputs_embeds, attention_mask

    def _adapt_forward_inputs_for_cross_attention(self, input_ids, attention_mask, expanded_block_embeddings,
                                                  past_adapted_length):
        """
        Maintain the original prefix and embed inputs (that's it!)

        Parameters:
            input_ids: input_ids (after discarding last token for optimal loss computation, when labels are given)
                [[BOS   A     B     C     ...],]
            attention_mask:
                [[1     a     b     c     D    ...],]

        Returns: (inputs_embeds, attention_mask, labels)
        """
        inputs_embeds = self.embed_input_ids(input_ids)
        return inputs_embeds, attention_mask
