import torch
from torch.utils.data import IterableDataset
from typing import Dict, Iterable, List
from transformers import PreTrainedTokenizer
import random

class TextWrapper(IterableDataset):
    """
    =================================================================================================
    A wrapper for text dataset.
    Every item in the dataset is a dictionary with a key 'text' and a value of the text.
    This wrapper will process the dataset and return in the form of a generator.
    Each new item will be like: {'input_ids': torch.Tensor[1, 2, 3], 'labels': torch.Tensor[1, 2, 3]}
    =================================================================================================
    dataset: Iterable[Dict[str, str]]: The dataset to be wrapped. This wrapper does not support shuffling.
             So the dataset should be shuffled before passing to this wrapper.
             Note: The dataset should properly handle the parallel loading, like properly split the 
                dataset into multiple shards when using multiple workers. e.g. when using DataLoader and 
                num_workers > 1, the dataset should be properly split into multiple shards. Or in the distributed
                setting. Whatever the case, TextWrapper simply wrap the text dataset and process it, TextWrapper 
                will not handle the parallel loading or something like that.
             
    tokenizer: PreTrainedTokenizer: The tokenizer to be used for tokenization.
    
    context_length: int: The length of the context.
    
    add_bos: bool: Whether to add the BOS token at the beginning of the context.
    add_eos: bool: Whether to add the EOS token at the end of the context.
    loop: bool: Whether to repeat the dataset infinitely.
    """
    def __init__(self, dataset: Iterable[Dict[str, str]],
                 tokenizer: PreTrainedTokenizer, 
                 context_size: int = 2048,
                 add_bos: bool = True,
                 add_eos: bool = True, 
                 loop: bool = False):
        
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.context_size = context_size
        
        self.add_bos = add_bos
        self.add_eos = add_eos
        
        self.loop = loop
        
    def __iter__(self):
        add_bos = self.add_bos
        add_eos = self.add_eos
        
        context_size = self.context_size
        buffer = torch.empty(context_size, dtype=torch.long)
        idx = 0
        
        while True:  # Infinite loop, dataset can be iterated multiple times
            for example in self.dataset:
                input_ids = self.tokenizer.encode(example['text'], add_special_tokens=False)
                if add_bos:
                    input_ids = [self.tokenizer.bos_token_id] + input_ids
                if add_eos:
                    input_ids = input_ids + [self.tokenizer.eos_token_id]
                
                length = len(input_ids)
                start = 0
                while start < length:
                    remain = context_size - idx
                    end = start + remain
                    if end >= length:
                        end = length
                    # print(f'start: {start}, end: {end}, idx: {idx} input_ids: {input_ids[start:end]}')
                    buffer[idx:idx + end - start] = torch.tensor(input_ids[start:end], dtype=torch.long)
                    idx += end - start
                
                    if idx == context_size:
                        yield {'input_ids': buffer.clone(), 'labels': buffer.clone()}
                        idx = 0
                    start = end  
                    
            if not self.loop:
                break
            
import copy

class CombinedDataset(IterableDataset):
    """
    a class to sample from multiple datasets with weights.
    like TextWrapper, this class will not handle the parallel loading or something like that.
    so dataset of datasets should be properly split into multiple shards when using multiple workers.
    """
    def __init__(self, datasets: List[Iterable], weights: List[float]=None, length: int=None, seed: int=1234,
                 skip_step: int = 0):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        self._len = length
        self._skip_step = skip_step
        
        assert len(datasets) == len(weights), "The number of datasets and weights should be the same."
         
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

        # weights should sum to 1
        # assert sum(self._weights) == 1, "The sum of the weights should be equal to 1."
    
    def __len__(self):
        if self._len is None:
            raise None #ValueError("The length of the dataset is not specified.")
        return self._len - self._skip_step
        
    def __iter__(self):
        return CombinedDatasetIterator(copy.deepcopy(self._datasets), self._seed, self._weights.copy(), len(self), 
                                       self._skip_step) # copy the datasets and weights to avoid changing the original ones.

import inspect

class CombinedDatasetIterator:
    def __init__(self, datasets: List[Iterable], seed: int, weights: List[float], length: int=None,
                 skip_step: int = -1):
        if len(datasets) != len(weights):
            raise ValueError("Datasets and weights must have the same length.")
        
        # self._original_datasets = datasets
        # self._datasets = [(iter(dataset), dataset) for dataset in datasets]
        self._weights = weights
        self._rng = random.Random(seed)
        self._len = length
        self._skip_step = skip_step
        
        # # simulate skipping the first skip_step items
        # skipped_steps = 0
        # skipped_steps_per_dataset = [0] * len(datasets)
        # remaining_steps = [dataset.remaining_steps for dataset in datasets]
        # # print("start to skip")
        # while skipped_steps < self._skip_step and datasets:
        #     chosen_index = self._rng.choices(range(len(datasets)), weights=self._weights, k=1)[0]
        #     remaining_steps[chosen_index] -= 1
        #     skipped_steps_per_dataset[chosen_index] += 1
        #     skipped_steps += 1
        #     if remaining_steps[chosen_index] < 0:
        #         # print("pop out!!!!!!!!")
        #         datasets.pop(chosen_index)
        #         self._weights.pop(chosen_index)
        #         remaining_steps.pop(chosen_index)
        #         skipped_steps_per_dataset.pop(chosen_index)
        #         skipped_steps -= 1
                
        # # normalize the weights
        # if self._weights and sum(self._weights) != 1:
        #     total_weight = sum(self._weights)
        #     self._weights = [w / total_weight for w in self._weights]
        
        # for i, dataset in enumerate(datasets):
        #     if inspect.ismethod(getattr(dataset, 'set_skip_steps', None)):
        #         dataset.set_skip_steps(skipped_steps_per_dataset[i])
        #     else:
        #         print("Warning: skip method is not implemented in the dataset.")
        # # print(f"skip_steps_per_dataset: {skipped_steps_per_dataset}")
        
        assert len(datasets) == len(weights), "The number of datasets and weights should be the same."
        if sum(weights) != 1:
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            self._weights = weights
        
        for weight, dataset in zip(weights, datasets):
            skip_steps = round(self._skip_step * weight)
            # print(f"skip_steps: {skip_steps}")
            if inspect.ismethod(getattr(dataset, 'set_skip_steps', None)):
                dataset.set_skip_steps(skip_steps)
            else:
                print("Warning: skip method is not implemented in the dataset.")
        
        self._iter_datasets = [iter(dataset) for dataset in datasets]
            


    def __iter__(self):
        return self

    def __next__(self):
        if not self._iter_datasets:
            raise StopIteration("All datasets have been exhausted.")

        num_item = 0
        # Try to get the next item from a randomly weighted chosen dataset
        while True:
            if self._len is not None and num_item >= self._len:
                raise StopIteration("The specified length has been reached.")
            
            chosen_index = self._rng.choices(range(len(self._iter_datasets)), weights=self._weights, k=1)[0]
            chosen_dataset_iter = self._iter_datasets[chosen_index]
            try:
                num_item += 1
                d = next(chosen_dataset_iter)
                assert d is not None, "The dataset should not return None."
                return d
            except StopIteration:
                # Remove the exhausted dataset and its weight
                # print("pop out!!!!!!!!")
                self._iter_datasets.pop(chosen_index)
                self._weights.pop(chosen_index)
                # Normalize weights if necessary
                if self._weights and sum(self._weights) != 1:
                    total_weight = sum(self._weights)
                    self._weights = [w / total_weight for w in self._weights]

                # If no datasets are left, raise StopIteration
                if not self._iter_datasets:
                    raise StopIteration("All datasets have been exhausted.")

if __name__ == '__main__':
    pass
