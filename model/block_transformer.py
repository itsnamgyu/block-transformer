from logging import warning
from typing import Optional, Union

import torch
from torch import nn
from transformers import PreTrainedModel

from model.block_decoder.base import BaseBlockDecoder
from model.embedder.base import BaseEmbedder
from model.token_decoder.base import BaseTokenDecoder
from util.token_mapper import TokenMapper


class BlockTransformer(nn.Module):
    def __init__(
        self,
        embedder: Union[BaseEmbedder, PreTrainedModel],
        block_decoder: Union[BaseBlockDecoder, PreTrainedModel],
        token_decoder: Union[BaseTokenDecoder, PreTrainedModel],
        token_mapper: TokenMapper = None,
        use_token_decoding_loss=False,
        use_block_decoding_loss=False,
        block_decoding_loss_weight=1.0,
        use_auto_encoding_loss=False,
        auto_encoding_loss_weight=1.0,
        decoding_strategy=None
    ):
        super().__init__()
        self.embedder = embedder
        self.token_decoder = token_decoder
        self.block_decoder = block_decoder
        self.token_mapper = token_mapper

        self.block_length = self.token_decoder.block_length
        self.n_embedding_tokens = self.embedder.n_embedding_tokens
        self.projection_hidden_size = self.embedder.projection_hidden_size

        self.use_token_decoding_loss = use_token_decoding_loss
        if self.use_token_decoding_loss and self.token_decoder is None:
            raise ValueError("token_decoder must be specified when use_token_decoding_loss is True")
        self.use_block_decoding_loss = use_block_decoding_loss
        self.block_decoding_loss_weight = block_decoding_loss_weight
        self.use_auto_encoding_loss = use_auto_encoding_loss
        self.auto_encoding_loss_weight = auto_encoding_loss_weight
        self.decoding_strategy = decoding_strategy

        # for Evaluation
        self.config = self.token_decoder.config

    def forward(
        self,
        input_ids: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        block_attention_mask: torch.LongTensor,
        labels: Optional[torch.FloatTensor] = None,
        skip_padding_blocks: bool = True,
    ):
        """
        Parameters:
            input_ids: input_ids (for embedder)
                [
                    [ [a, b, c, d], [e, f, g, h], [E, P, P, P], [P, P, P, P] ]  # ends with full EOS block
                    [ [a, b, c, d], [e, f, g, h], [i, j, E, P], [P, P, P, P] ]  # ends with partial EOS block
                ]
              - Note that E is the EOS token and P is the padding token, but these may be the same in some cases.
              - Tokens should be for the embedding model (if token_mapper is not None, then
                token_mapper.embedder_to_token_decoder is applied)
            attention_mask:
                [
                    [ [1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0] ]  # ends with full EOS block
                    [ [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0], [0, 0, 0, 0] ]  # ends with partial EOS block
                ]
            block_attention_mask:
                [
                    [ 1, 1, 1, 0 ],
                    [ 1, 1, 1, 0 ],
                ]
            labels: labels (=input_ids) for embedder (required when use_token_decoding_loss is True)
            skip_padding_blocks: if True, skip padding (full EOS) blocks
        """
        # reshape inputs so that dim 0 is batch_size
        input_ids = input_ids.view((-1,) + input_ids.shape[-2:])
        attention_mask = attention_mask.view((-1,) + attention_mask.shape[-2:])
        block_attention_mask = block_attention_mask.view(-1, block_attention_mask.shape[-1])
        batch_size, n_blocks, block_length = input_ids.shape
        if labels is not None:
            labels = labels.reshape((-1,) + labels.shape[-2:])  # make 3d
            if input_ids.shape != labels.shape:
                raise ValueError("input_ids and labels must have the same shape")
        if block_attention_mask.shape != (batch_size, n_blocks):
            raise ValueError("block_attention_mask must have shape (batch_size, n_blocks)"
                             f"expected: {(batch_size, n_blocks)}, "
                             f"got: {block_attention_mask.shape}")

        ########################################
        # EMBEDDING
        ########################################

        input_embeds = self.embedder(
            input_ids=input_ids.view(-1, block_length),
            attention_mask=attention_mask.view(-1, block_length),
        )  # -> (batch_size, n_blocks, n_embedding_tokens, hidden_size)

        ########################################
        # BLOCK DECODING
        ########################################

        input_embeds = input_embeds.view(batch_size, n_blocks * self.n_embedding_tokens, input_embeds.shape[-1])
        block_decoder_output = self.block_decoder(inputs_embeds=input_embeds, block_attention_mask=block_attention_mask,
                                                  return_dict=True)

        loss = None
        block_decoding_loss = None
        if self.use_block_decoding_loss and labels is not None:
            block_decoding_loss = block_decoder_output.loss
            block_decoding_loss = self.block_decoding_loss_weight * block_decoding_loss
            loss = block_decoding_loss if loss is None else loss + block_decoding_loss

        ########################################
        # TOKEN DECODING
        ########################################

        # shift label blocks left
        input_ids = input_ids[:, 1:, :].contiguous()
        input_ids = input_ids.view(-1, block_length)
        attention_mask = attention_mask[:, 1:, :].contiguous()
        attention_mask = attention_mask.view(-1, block_length)
        if labels is not None:
            labels = labels[:, 1:, :].contiguous()
            labels = labels.view(-1, block_length)
        block_attention_mask = block_attention_mask[:, 1:].contiguous()
        block_attention_mask = block_attention_mask.view(-1)

        # map embedder tokens to token decoder tokens
        if self.token_mapper is not None:
            input_ids = self.token_mapper.embedder_to_token_decoder(input_ids)
            labels = self.token_mapper.embedder_to_token_decoder(labels) if labels is not None else None

        assert input_ids.shape[0] == batch_size * (n_blocks - 1)

        # remove last block
        block_embeddings = block_decoder_output.hidden_states[:, :-self.n_embedding_tokens, :].contiguous()
        # -> [batch_size, (n_blocks - 1) * n_embedding_tokens, projection_hidden_size]
        block_embeddings = block_embeddings.reshape(-1, self.n_embedding_tokens, self.projection_hidden_size)
        # -> [batch_size * (n_blocks - 1), n_embedding_tokens, projection_hidden_size]
        # == [local_batch_size, n_embedding_tokens, projection_hidden_size]

        # for auto_encoding_loss
        if self.use_auto_encoding_loss:
            input_embeds = input_embeds[:, self.n_embedding_tokens:, :].contiguous()
            input_embeds = input_embeds.view(-1, self.n_embedding_tokens, self.projection_hidden_size)

        if skip_padding_blocks:
            # block_attention_mask == 1 for non padding (non full EOS) blocks
            non_padding_blocks = block_attention_mask.bool()
            input_ids = input_ids[non_padding_blocks]
            labels = labels[non_padding_blocks] if labels is not None else None
            attention_mask = attention_mask[non_padding_blocks]
            block_embeddings = block_embeddings[non_padding_blocks]
            if self.use_auto_encoding_loss:
                input_embeds = input_embeds[non_padding_blocks]

        local_batch_size = input_ids.shape[0]

        # add prefixes to token_decoder inputs
        input_ids_prefix = input_ids.new_full((local_batch_size, 1), self.token_decoder.config.bos_token_id)
        input_ids = torch.cat([input_ids_prefix, input_ids], dim=1)
        attention_mask_prefix = attention_mask.new_full((local_batch_size, 1), 1)
        attention_mask = torch.cat([attention_mask_prefix, attention_mask], dim=1)
        del input_ids_prefix, attention_mask_prefix
        if labels is not None:
            labels_prefix = labels.new_full((local_batch_size, 1), -100) if labels is not None else None
            labels = torch.cat([labels_prefix, labels], dim=1)
            del labels_prefix

        token_decoder_output = self.token_decoder(input_ids=input_ids,
                                                  attention_mask=attention_mask,
                                                  block_embeddings=block_embeddings,
                                                  discard_redundant_tokens=True,
                                                  labels=labels if self.use_token_decoding_loss else None)

        token_decoding_loss = None
        if self.use_token_decoding_loss:
            token_decoding_loss = token_decoder_output.loss
            loss = token_decoding_loss if loss is None else loss + token_decoding_loss

        auto_encoding_loss = None
        if self.use_auto_encoding_loss and labels is not None:
            auto_encoding_loss = self.token_decoder(input_ids=input_ids,
                                                    attention_mask=attention_mask,
                                                    block_embeddings=input_embeds,
                                                    labels=labels,
                                                    discard_redundant_tokens=True,
                                                    log_loss_per_position=False).loss
            if auto_encoding_loss is not None:
                # when evaluation, auto_encoding_loss is None because no labels are given
                auto_encoding_loss = self.auto_encoding_loss_weight * auto_encoding_loss
                loss = auto_encoding_loss if loss is None else loss + auto_encoding_loss

        logits_shape = (batch_size * (n_blocks - 1), block_length, self.token_decoder.config.vocab_size)
        if labels is None:
            if skip_padding_blocks:
                logits = token_decoder_output.logits.new_full(
                    logits_shape, fill_value=self.token_decoder.config.eos_token_id)
                assert non_padding_blocks.shape == (batch_size * (n_blocks - 1),)
                logits[non_padding_blocks] = token_decoder_output.logits.detach()
            else:
                logits = token_decoder_output.logits.view(logits_shape)
            logits = logits.view(batch_size, n_blocks - 1, block_length, self.token_decoder.config.vocab_size)
        else:
            logits = None  # refer to token decoder

        # save losses in self for logging
        self.loss = loss.detach().clone() if loss is not None else None
        self.token_decoding_loss = token_decoding_loss.detach().clone() if token_decoding_loss is not None else None
        self.block_decoding_loss = block_decoding_loss.detach().clone() if block_decoding_loss is not None else None
        self.auto_encoding_loss = auto_encoding_loss.detach().clone() if auto_encoding_loss is not None else None

        return {
            "logits": logits,
            "loss": loss,
            "block_decoding_loss": block_decoding_loss,
            "token_decoding_loss": token_decoding_loss,
            "auto_encoding_loss": auto_encoding_loss,
        }

    def preprocess_inputs_for_generation(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor):
        """
        :return: {
            "input_ids": torch.Tensor[batch_size, n_blocks, block_length],
            "attention_mask": torch.Tensor[batch_size, n_blocks, block_length],
            "block_attention_mask": torch.Tensor[batch_size, n_blocks],
        }
        """
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        bs, length = input_ids.shape

        block_boundary_remainder = input_ids.shape[-1] % self.block_length
        pad_length = 0
        if block_boundary_remainder != 0:
            pad_length = self.block_length - block_boundary_remainder
            # left pad with pad_token_id with full_like
            input_ids = torch.nn.functional.pad(input_ids, (pad_length, 0), mode="constant",
                                                value=self.embedder.config.pad_token_id)
            attention_mask = torch.nn.functional.pad(attention_mask, (pad_length, 0), mode="constant", value=0)

        input_ids = input_ids.view(bs, -1, self.block_length)
        attention_mask = attention_mask.view(bs, -1, self.block_length)
        block_attention_mask = attention_mask.any(dim=-1).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "block_attention_mask": block_attention_mask,
            "initial_block_padding": pad_length,
        }

    @staticmethod
    def revert_block_input_ids_to_vanilla(input_ids, added_initial_block_padding, last_block_unfilled_length):
        """
        Converts block format to vanilla format
        """
        input_ids = input_ids.view(-1, input_ids.shape[-2] * input_ids.shape[-1])
        input_ids = input_ids[:, added_initial_block_padding:]
        if last_block_unfilled_length > 0:
            input_ids = input_ids[:, :-last_block_unfilled_length]
        return input_ids

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        block_attention_mask: Optional[torch.LongTensor] = None,
        max_length: int = 100,
        benchmark: bool = False,
        streamer=None,
    ):
        """
        Inputs may be given in vanilla format or block format.
        `block_attention_mask` is required if and only if the inputs are in block format.

        Parameters:
            input_ids:
            attention_mask:
            block_attention_mask:
                If specified, will assume that all inputs are organized in blocks.
            max_length: maximum number of *tokens* to generate (not blocks)
                Includes prompt length, excludes initial block padding added within this function
            benchmark: if True, print timing information

        Returns:
            input_ids:
                torch.Tensor[batch_size, n_blocks, block_length] if block format
                torch.Tensor[batch_size, n_tokens] if vanilla format
        """
        added_initial_block_padding = 0  # padding tokens added *within* this function
        initial_shapes = {
            "input_ids": input_ids.shape,
            "attention_mask": attention_mask.shape if attention_mask is not None else None,
            "block_attention_mask": block_attention_mask.shape if block_attention_mask is not None else None,
        }
        vanilla_mode = block_attention_mask is None
        if vanilla_mode:  # vanilla format
            d = self.preprocess_inputs_for_generation(input_ids, attention_mask)
            input_ids = d["input_ids"]
            attention_mask = d["attention_mask"]
            block_attention_mask = d["block_attention_mask"]
            added_initial_block_padding = d["initial_block_padding"]
        else:  # block format
            if input_ids.shape[-1] != self.block_length:
                message = "input_ids.shape[-1] must be equal to self.block_length when block_attention_mask is given"
                raise ValueError(message)
            if attention_mask is None:
                raise ValueError("attention_mask must be specified when block_attention_mask is specified")

        n_blocks, block_length = input_ids.shape[-2:]
        input_ids = input_ids.view(-1, n_blocks, block_length)
        attention_mask = attention_mask.view(-1, n_blocks, block_length)
        block_attention_mask = block_attention_mask.view(-1, n_blocks)
        batch_size = input_ids.shape[0]

        if input_ids.shape != attention_mask.shape:
            raise ValueError(f"input_ids and attention_mask must have equivalent shapes. Given: {initial_shapes}")
        if input_ids.shape[:2] != block_attention_mask.shape[-2:]:
            raise ValueError(f"input_ids and block_attention_mask must have equivalent shapes. Given: {initial_shapes}")

        if input_ids.shape[1] * input_ids.shape[2] - added_initial_block_padding > max_length:
            warning.warn("max_length is less than the length of the input_ids. Returning original input_ids.")
            if vanilla_mode:
                input_ids = self.revert_block_input_ids_to_vanilla(input_ids, added_initial_block_padding,
                                                                   last_block_remaining_length=0)
            return input_ids

        if benchmark:
            print(f"Benchmarking generation... (benchmark={benchmark})")
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            token_decoder_starts = []
            token_decoder_ends = []
            block_decoder_starts = []
            block_decoder_ends = []

        input_block_embeds = self.embed_blocks(input_ids, attention_mask)
        # -> [batch_size, n_blocks, n_embedding_tokens, projection_hidden_size]
        input_block_embeds = input_block_embeds.view(batch_size, -1, input_block_embeds.shape[-1])
        # -> [batch_size, n_blocks * n_embedding_tokens, projection_hidden_size]

        unfinished_sequences = input_ids.new_ones([batch_size], dtype=torch.long)
        block_decoder_kwargs = dict()
        block_decoder_kwargs["past_key_values"] = None
        block_decoder_kwargs["block_attention_mask"] = block_attention_mask

        if self.n_embedding_tokens != 1:
            raise NotImplementedError("generation currently only implemented for n_embedding_tokens == 1")

        # Implementation based on `transformers.generation.utils.GenerationMixin.greedy_search`
        while True:
            if benchmark:
                block_decoder_starts.append(torch.cuda.Event(enable_timing=True))
                block_decoder_starts[-1].record()

            block_decoder_output = self.block_decoder(inputs_embeds=input_block_embeds, return_dict=True,
                                                      **block_decoder_kwargs)

            if benchmark:
                block_decoder_ends.append(torch.cuda.Event(enable_timing=True))
                block_decoder_ends[-1].record()
            next_block_embeds = block_decoder_output.hidden_states[:, -1, :]  # [bs, hidden_size]
            decode_mask = unfinished_sequences.bool()

            # decode next block. do not compute finished sequences. set finished sequences to pad_token_id
            next_tokens = input_ids.new_ones([batch_size, self.block_length],
                                             dtype=torch.long) * self.token_decoder.config.pad_token_id
            # support partial decoding of final block when max_length is not a multiple of block_length
            current_length = input_ids.shape[1] * input_ids.shape[2] - added_initial_block_padding
            next_token_count = min(max_length - current_length, self.block_length)
            block_embeddings = next_block_embeds[decode_mask, :].contiguous()  # [bs, hidden_size]
            block_embeddings = block_embeddings.view(-1, self.n_embedding_tokens, self.projection_hidden_size)
            if benchmark:
                token_decoder_starts.append(torch.cuda.Event(enable_timing=True))
                token_decoder_starts[-1].record()
            # note that first token is reserved for placeholder EOS token in token decoder
            generated = self.token_decoder.generate(block_embeddings=block_embeddings, max_length=next_token_count + 1,
                                                    streamer=streamer)

            next_tokens[decode_mask, :generated.shape[-1] - 1] = generated[:, 1:]
            if benchmark:
                token_decoder_ends.append(torch.cuda.Event(enable_timing=True))
                token_decoder_ends[-1].record()
            if self.token_mapper is not None:
                embedder_next_tokens = self.token_mapper.token_decoder_to_embedder(next_tokens)
            else:
                embedder_next_tokens = next_tokens

            # update inputs
            input_ids = torch.cat([input_ids, embedder_next_tokens.unsqueeze(1)], dim=1)

            # update unfinished_sequences
            if self.token_decoder.generation_config.eos_token_id is not None:
                unfinished_sequences *= (next_tokens.ne(self.token_decoder.generation_config.eos_token_id)).all(dim=-1)

            # check end condition
            if input_ids.shape[1] * input_ids.shape[2] >= max_length or unfinished_sequences.sum() == 0:
                break

            # re-embed next block. set finished sequences to zero embeddings
            next_block_embeds_shape = (batch_size, self.n_embedding_tokens, self.projection_hidden_size)
            input_block_embeds = input_block_embeds.new_zeros(next_block_embeds_shape)
            input_block_embeds[decode_mask, :] = self.embedder(embedder_next_tokens[decode_mask, :])

            # update block_decoder_kwargs
            block_decoder_kwargs["past_key_values"] = block_decoder_output.past_key_values
            block_decoder_kwargs["block_attention_mask"] = torch.cat(
                [block_decoder_kwargs["block_attention_mask"], unfinished_sequences.unsqueeze(-1)], dim=-1)

        if benchmark:
            end.record()

            torch.cuda.synchronize()
            block_decoder_times = []
            token_decoder_times = []
            for i in range(len(block_decoder_starts)):
                block_decoder_times.append(block_decoder_starts[i].elapsed_time(block_decoder_ends[i]))
            for i in range(len(token_decoder_starts)):
                token_decoder_times.append(token_decoder_starts[i].elapsed_time(token_decoder_ends[i]))

            print(" Generation time (ms) ".center(100, "-"))
            print("Total time: ", start.elapsed_time(end))
            print("Block decoder times: ", block_decoder_times)
            print("Token decoder times: ", token_decoder_times)
            print("Block decoder total time: ", sum(block_decoder_times))
            print("Token decoder total time: ", sum(token_decoder_times))

        if vanilla_mode:
            input_ids = self.revert_block_input_ids_to_vanilla(
                input_ids, added_initial_block_padding, last_block_unfilled_length=block_length - next_token_count)
        return input_ids

    def embed_blocks(self, input_ids, attention_mask):
        """
        Embeds blocks using the embedder
        :param input_ids: input_ids (for embedding model tokenizer)
        :param attention_mask:
        :return: [batch_size, n_blocks, n_embedding_tokens, projection_hidden_size]
        """
        assert len(input_ids.shape) == 3 and len(attention_mask.shape) == 3
        batch_size, n_blocks, block_length = input_ids.shape
        input_ids = input_ids.view(-1, input_ids.shape[-1])  # [bs * n_blocks, block_length]
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        block_embeds = self.embedder(input_ids, attention_mask)
        # -> [bs * n_blocks, n_embedding_tokens, projection_hidden_size]
        block_embeds = block_embeds.view((batch_size, n_blocks) + block_embeds.shape[-2:])
        return block_embeds
