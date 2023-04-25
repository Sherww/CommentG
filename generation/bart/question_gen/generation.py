import torch
from typing import Tuple, List, Callable

import tqdm

class DecodeStrategy(object):
    def __init__(self, pad_id, bos_id, eos_id, min_length, max_length):
        self.pad_id=pad_id
        self.bos_id=bos_id
        self.eos_id=eos_id
        self.min_length=min_length
        self.max_length=max_length

class BeamSearch(DecodeStrategy):
    def __init__(self, pad_id, bos_id, eos_id, min_length, max_length, top_k=10):
        super(GreedySearch, self).__init__(pad_id, bos_id, eos_id, min_length, max_length)
        self.top_k = top_k

    def search(self, source_inputs: torch.Tensor, init_states: torch.Tensor, predit_fn: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
        # source_inputs: (batch_size, seq_len)
        batch_size = source_inputs.size(0)
        max_step = 256
        translation_ids = torch.full((batch_size, self.top_k, max_step), self.pad_id)
        scores = torch.zeros(batch_size, self.top_k)

        # init_states： (batch_size, top_k, max_step)
        states = init_states
        for i in tqdm.tqdm(iterable=range(max_step)):
            # output: (batch_size, seq_len, vocab_size)
            output = predit_fn(source_inputs, states.view(batch_size * self.top_k, -1))
            output = output.view(batch_size, self.top_k, output.shape[1], output.shape[2])
            for batch in range(batch_size):
                topk_prob, topk_ids = torch.topk(output[batch, :, :, :].squeeze(), self.top_k, dim=2)
                states = torch.cat((states, torch.unsqueeze(topk_ids[:, i], dim=0)), dim=1)
            translation_ids[:, i] = topk_ids[:, i]
        print(translation_ids)
        return translation_ids

class GreedySearch(DecodeStrategy):
    def __init__(self, pad_id, bos_id, eos_id, min_length, max_length):
        super(GreedySearch, self).__init__(pad_id, bos_id, eos_id, min_length, max_length)

    def search(self, source_inputs: torch.Tensor, init_states: torch.Tensor, predit_fn: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
        # source_inputs: (batch_size, seq_len)
        batch_size = source_inputs.size(0)
        max_step = 256
        translation_ids = torch.full((batch_size, max_step), self.pad_id)

        states = init_states

        for i in tqdm.tqdm(iterable=range(max_step)):
            output = predit_fn(source_inputs, states)
            # output: (batch_size, seq_len, vocab_size)
            max_prob, max_ids = torch.max(output, dim=2)

            states = torch.cat((states, torch.unsqueeze(max_ids[:, i], dim=1)), dim=1)
            translation_ids[:, i] = max_ids[:, i]
        print(translation_ids)

        return translation_ids