import warnings

import torch
from tokenizers import Tokenizer


class TokenMapper:
    def __init__(self, embedder_tokenizer: Tokenizer, token_decoder_tokenizer: Tokenizer):
        self.embedder_tokenizer = embedder_tokenizer
        self.token_decoder_tokenizer = token_decoder_tokenizer
        self._build_maps()

    def embedder_to_token_decoder(self, embedder_ids: torch.LongTensor) -> torch.LongTensor:
        device = embedder_ids.device
        self.embedder_to_token_decoder_map = self.embedder_to_token_decoder_map.to(device)
        
        token_decoder_ids = self.embedder_to_token_decoder_map[embedder_ids]
        
        # -100 should be mapped to -100
        token_decoder_ids[embedder_ids == -100] = -100
        
        if (token_decoder_ids == -1).any():
            warnings.warn("Some tokens in the input_ids are not found in the token_decoder_tokenizer")
        return token_decoder_ids

    def token_decoder_to_embedder(self, token_decoder_ids: torch.LongTensor) -> torch.LongTensor:
        # only used for generation
        warnings.warn("All eos tokens are mapped to the embedder eos token")
        
        device = token_decoder_ids.device
        self.token_decoder_to_embedder_map = self.token_decoder_to_embedder_map.to(device)
        
        embedder_ids = self.token_decoder_to_embedder_map[token_decoder_ids]
        
        # -100 should be mapped to -100
        embedder_ids[token_decoder_ids == -100] = -100
        
        if (embedder_ids == -1).any():
            warnings.warn("Some tokens in the input_ids are not found in the embedder_tokenizer")
        return embedder_ids

    def _build_maps(self):
        token_pairs = []
        embedder_vocab = self.embedder_tokenizer.get_vocab()
        token_decoder_vocab = self.token_decoder_tokenizer.get_vocab()

        # add common tokens to token_pairs
        common_tokens = set(embedder_vocab.keys()).intersection(set(token_decoder_vocab.keys()))

        for token in common_tokens:
            eid = embedder_vocab[token]
            tid = token_decoder_vocab[token]
            token_pairs.append((eid, tid))

        # initialize as -1 (not found)
        self.embedder_to_token_decoder_map = torch.ones(len(embedder_vocab), dtype=torch.long) * -1
        self.token_decoder_to_embedder_map = torch.ones(len(token_decoder_vocab), dtype=torch.long) * -1

        for eid, tid in token_pairs:
            self.embedder_to_token_decoder_map[eid] = tid
            self.token_decoder_to_embedder_map[tid] = eid
            
        # add special tokens
        embedder_bos = self.embedder_tokenizer.bos_token_id
        embedder_eos = self.embedder_tokenizer.eos_token_id
        embedder_pad = self.embedder_tokenizer.pad_token_id
        token_decoder_bos = self.token_decoder_tokenizer.bos_token_id
        token_decoder_eos = self.token_decoder_tokenizer.eos_token_id
        token_decoder_pad = self.token_decoder_tokenizer.pad_token_id
        
        self.embedder_to_token_decoder_map[embedder_bos] = token_decoder_bos if token_decoder_bos else token_decoder_eos
        self.embedder_to_token_decoder_map[embedder_eos] = token_decoder_eos
        self.embedder_to_token_decoder_map[embedder_pad] = token_decoder_pad if token_decoder_pad else token_decoder_eos
        
        if token_decoder_pad and token_decoder_pad != token_decoder_eos:  # T5
            self.token_decoder_to_embedder_map[token_decoder_pad] = embedder_pad
            self.token_decoder_to_embedder_map[token_decoder_eos] = embedder_eos
        else:  # GPT2
            self.token_decoder_to_embedder_map[token_decoder_eos] = embedder_pad
        
        # map missing keys in embedder to token_decoder_eos
        missing_keys = set(embedder_vocab.keys()) - set(token_decoder_vocab.keys())
        for key in missing_keys:
            eid = embedder_vocab[key]
            self.embedder_to_token_decoder_map[eid] = token_decoder_eos
        