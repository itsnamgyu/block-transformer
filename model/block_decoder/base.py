import warnings
from abc import abstractmethod, ABCMeta
from typing import Optional, Tuple

import torch
from torch.nn import MSELoss


class BaseBlockDecoder(metaclass=ABCMeta):
    """
    Abstract class for token decoder models
    """

    def __init__(self, hidden_size: int, n_embedding_tokens=8, use_block_decoding_loss=False, block_decoding_loss_type="contrastive"):
        """
        :param hidden_size: used for embedder and token_decoder intialization
        :param n_embedding_tokens:
        :param use_block_decoding_loss:
        :param block_decoding_loss_type: "mse" or "contrastive"
        """
        self.hidden_size = hidden_size
        self.n_embedding_tokens = n_embedding_tokens
        self.use_block_decoding_loss = use_block_decoding_loss
        self.block_decoding_loss_type = block_decoding_loss_type
        
    def compute_loss(self, hidden_states, attention_mask, labels_embeds):
        loss = None
        if self.use_block_decoding_loss and labels_embeds is not None:
            # move labels to correct device to enable model parallelism
            # labels_embeds (bs, n_blocks * n_embeddding_tokens, hidden_size)
            
            # Compute loss in fp32 to match with logit-based loss in normal tokenwise GPTNeo
            # Conversion to 32bit is not applied in GPTNeoX. But, we will follow GPTNeo for full-precision
            hidden_size = hidden_states.shape[-1]

            # Shift so that tokens < n predict n
            # This is identical to super().forward() except we use mse between embeddings, and we apply explicit loss
            # masking (note that loss masking in vanilla transformers is done by setting labels to -100)
            label_mask = attention_mask[..., self.n_embedding_tokens:].unsqueeze(-1)
            hidden_states = hidden_states[..., :-self.n_embedding_tokens, :] * label_mask
            labels_embeds = labels_embeds[..., self.n_embedding_tokens:, :] * label_mask
            
            # -> (bs, n_blocks * n_embeddding_tokens - n_embedding_tokens, hidden_size)
            hidden_states = hidden_states.view(-1, hidden_size)
            labels_embeds = labels_embeds.view(-1, hidden_size)
            # -> (bs * (n_blocks * n_embeddding_tokens - n_embedding_tokens), hidden_size)
            if self.block_decoding_loss_type == "mse":
                loss_fct = MSELoss()
                loss = loss_fct(hidden_states.float(), labels_embeds.float().detach())
            elif self.block_decoding_loss_type == "contrastive":
                # normalize embeddings
                hidden_states = hidden_states / hidden_states.norm(dim=-1, keepdim=True)
                labels_embeds = labels_embeds / labels_embeds.norm(dim=-1, keepdim=True)
                
                # compute cosine similarity between embeddings (N x D) x (N x D) -> N x N
                temperature = 0.07
                logits = torch.div(torch.mm(hidden_states, labels_embeds.t().detach()), temperature)
                
                logits_max, _ = logits.max(dim=-1, keepdim=True)
                logits = logits - logits_max.detach()
                
                # compute log_prob
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
                
                # compute mean of negative log-likelihood
                loss = -(log_prob.diag()).mean()
            else:
                raise ValueError(f"block_decoding_loss_type {self.block_decoding_loss_type} not supported")
            
        elif self.use_block_decoding_loss and labels_embeds is None:
            warnings.warn("use_block_decoding_loss is True but labels_embeds is None, so no loss will be computed")
            
        return loss

    @abstractmethod
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
        labels_embeds: Optional[torch.Tensor] = None,  # added
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        raise NotImplementedError("forward() must be implemented in subclass")
