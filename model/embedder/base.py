import warnings
from abc import abstractmethod, ABCMeta
from typing import Optional

import torch
import torch.nn as nn

SUPPORTED_PROJECTION_METHODS = ["projection_layer", "concat"]


class BaseEmbedder(metaclass=ABCMeta):
    """
    Abstract class for embedder models
    """

    def __init__(self, config, block_length=4, n_embedding_tokens=1, projection_method=None,
                 projection_hidden_size=None, init_projection=True):
        super().__init__()
        self.block_length = block_length
        self.n_embedding_tokens = n_embedding_tokens
        self.hidden_size = config.hidden_size
        self.projection_method = projection_method
        self.projection_hidden_size = projection_hidden_size

        if self.projection_method is None:
            self.projection_method = "concat"
            warnings.warn("[BaseEmbedder] projection_method is None, defaulting to concat")

        self.projection_layer = None
        if self.projection_method == "projection_layer":
            if self.projection_hidden_size is None:
                self.projection_hidden_size = self.hidden_size
                warnings.warn("projection_hidden_size not specified, defaulting to hidden_size")

            assert self.block_length % self.n_embedding_tokens == 0, \
                f"block_length must be divisible by n_embedding_tokens, got {block_length} and {n_embedding_tokens}"

            kernel_size = self.block_length // self.n_embedding_tokens
            self.projection_layer = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.projection_hidden_size,
                                              kernel_size=kernel_size, stride=kernel_size, padding=0)
            if init_projection:
                self._init_projection_layer()
            else:
                warnings.warn("[BaseEmbedder] `init_projection=False`")

        elif self.projection_method == "concat":
            if block_length % n_embedding_tokens != 0:
                message = f"block_length must be divisible by n_embedding_tokens, but got {block_length} " \
                          f"and {n_embedding_tokens}"
                raise NotImplementedError(message)
            input_tokens_per_block = block_length // n_embedding_tokens

            if config.hidden_size is None:
                raise ValueError("config.hidden_size must be specified when projection_method is concat")
            # make sure that projection_hidden_size == config.hidden_size * input_tokens_per_block
            if config.hidden_size is not None:
                if projection_hidden_size != config.hidden_size * input_tokens_per_block:
                    message = f"projection_hidden_size must be equal to config.hidden_size * (block_length // " \
                              f"n_embedding_tokens), but got {projection_hidden_size} and " \
                              f"{config.hidden_size * (block_length // n_embedding_tokens)}"
                    raise ValueError(message)
            elif config.hidden_size is not None:
                self.projection_hidden_size = config.hidden_size * input_tokens_per_block

        else:
            raise ValueError(f"projection_method must be one of {SUPPORTED_PROJECTION_METHODS}, "
                             f"got {self.projection_method}")

    def project_hidden_state(self, hidden_state: torch.Tensor):
        """
        :param hidden_state:
        :return: [batch_size, n_embedding_tokens, projection_hidden_size]
        """
        if self.projection_method == "projection_layer":
            hidden_state = hidden_state.transpose(1, 2)
            # -> [batch_size, hidden_size, block_length]
            hidden_state = self.projection_layer(hidden_state).transpose(1, 2)
            # -> [batch_size, n_embedding_tokens, projection_hidden_size]
        elif self.projection_method == "concat":
            hidden_state = hidden_state.view(hidden_state.shape[0], -1)
            # -> [batch_size, embedder_hidden_size * block_length]
            # == [batch_size, hidden_size]
        else:
            raise NotImplementedError(f"projection_method must be one of {SUPPORTED_PROJECTION_METHODS}, "
                                      f"got {self.projection_method}")

        hidden_state = hidden_state.view(hidden_state.shape[0], self.n_embedding_tokens, -1)
        # -> [batch_size, n_embedding_tokens, embedder_hidden_size]
        return hidden_state

    def _init_projection_layer(self):
        factor = 1.0  # TODO
        self.projection_layer.weight.data.normal_(mean=0.0,
                                                  std=factor * ((self.hidden_size * self.block_length) ** -0.5))
        if hasattr(self.projection_layer, "bias") and self.projection_layer.bias is not None:
            self.projection_layer.bias.data.zero_()

    @abstractmethod
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        raise NotImplementedError("forward() must be implemented in subclass")
