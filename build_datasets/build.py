import json
import random
import os
import re
from typing import List

from build_datasets.text_datasets import match_files
from build_datasets.datasets import TextWrapper, CombinedDataset
from build_datasets import PackedDataset
from build_datasets import CombinedDataset
from datasets import load_dataset

def build_with_config(all_files: List[str], config: dict, context_size: int = 2048,
          rank: int = 0, world_size: int = 1,
          verbose: bool = False,
          seed: int = 1234, skip_step: int = -1):
    # config is a json in the form like:
    # {
    #     "prefix1": {
    #         "type": ..., # "number" of "rate"
    #         "train": ...
    #         "validation": ...
    #     }, 
    #     ...
    # }
    assert len(all_files) > 0, "No files found"
    random.seed(seed)
    
    files_with_prefix = {
        prefix: [f for f in all_files if os.path.basename(f).startswith(prefix)]
        for prefix in config.keys()
    }
    
    if verbose:
        for prefix, files in files_with_prefix.items():
            print(f'{prefix}: {len(files)} files')
    
    train_datasets = []
    val_datasets = []

    # 遍历配置文件中每个子目录的设置
    for prefix, entry in config.items():
        files = files_with_prefix.get(prefix, None)
        # print(f'prefix: {prefix}')
        # print(f'entry: {entry}')
        # print(f'files: {files}')
        if files is None or len(files) == 0:
            print(f"warning: No files found for prefix {prefix}")
            continue
        
        if entry['type'] == 'number':
            # 按数量选择文件
            train_count = entry['train']
            validation_count = entry.get('validation', len(files) - train_count)
        elif entry['type'] == 'rate':
            # 按比例选择文件
            train_count = int(len(files) * entry['train'])
            validation_count = int(len(files) * entry['validation']) if 'validation' in entry else len(files) - train_count
        
        total_count = train_count + validation_count
        if total_count > len(files):
            raise ValueError(f"Not enough files with prefix {prefix} to select {total_count} files (only {len(files)} available).")
            
        selected_files = random.sample(files, total_count)
        
        split_point = train_count - 1 if validation_count <= 0 else train_count # make sure there is at least one file in the validation set
        train_files = selected_files[:split_point]
        validation_files = selected_files[split_point:total_count]
        
        train_dataset = PackedDataset(train_files, 
                                        rank=rank,
                                        world_size=world_size,
                                        n_buffered_files=2, 
                                        sample_size=context_size, 
                                        seed=seed)
        val_dataset = PackedDataset(validation_files, 
                                        rank=rank,
                                        world_size=world_size,
                                        n_buffered_files=2, 
                                        sample_size=context_size, 
                                        seed=seed)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        
    
    train_datasets_weight = [len(d) for d in train_datasets]
    sum_weight = sum(train_datasets_weight)
    train_datasets_weight = [w / sum_weight for w in train_datasets_weight]
    train_dataset = CombinedDataset(train_datasets, train_datasets_weight, length=sum_weight, seed=seed, skip_step=skip_step)
    
    if verbose:
        print(f'prefixes: {list(files_with_prefix.keys())}')
        print(f'train weight: {train_datasets_weight}')
        print(f'train total: {len(train_dataset)} samples')
    
    val_datasets_weight = [len(d) for d in val_datasets]
    sum_weight = sum(val_datasets_weight)
    val_datasets_weight = [w / sum_weight for w in val_datasets_weight]
    val_dataset = CombinedDataset(val_datasets, val_datasets_weight, length=sum_weight, seed=seed)
    
    if verbose:
        print(f'prefixes: {list(files_with_prefix.keys())}')
        print(f'val weight: {val_datasets_weight}')
        print(f'val total: {len(val_dataset)} samples')
    
    return train_dataset, val_dataset
        

def extract_prefix(file: str) -> str:
    # 正则表达式匹配尾部的数字模式
    match = re.search(r"(.+)-\d{3}-\d{5}\.data$", file)
    if match:
        # 整个匹配的前半部分是prefix
        return match.group(1).split('/')[-1]  # 也处理掉路径部分，只返回文件名的前缀
    raise ValueError("File path format is incorrect")
    
    
def build_with_prefix(all_files: List[str], context_size: int = 2048,
          rank: int = 0, world_size: int = 1,
          verbose: bool = False,
          seed: int = 1234, skip_step: int = 0,
          n_buffered_file=0, dispatch: bool = False):
    
    random.seed(seed)
    files_with_prefix = {}
    for f in all_files:
        filename = os.path.basename(f)
        prefix = extract_prefix(filename)
        if prefix not in files_with_prefix:
            files_with_prefix[prefix] = []
        files_with_prefix[prefix].append(f)
        
    if verbose:
        for prefix, files in files_with_prefix.items():
            print(f'{prefix}: {len(files)} files')
        
    
    datasets = []
    
    for _, files in files_with_prefix.items():
        dataset = PackedDataset(files, rank=rank, world_size=world_size,
                                n_buffered_files=n_buffered_file, sample_size=context_size, seed=seed, dispatch=dispatch)
        datasets.append(dataset)
    
    datasets_weight = [len(d) for d in datasets]
    sum_weight = sum(datasets_weight)
    datasets_weight = [w / sum_weight for w in datasets_weight]
    dataset = CombinedDataset(datasets, datasets_weight, length=sum_weight, seed=seed, skip_step=skip_step)
    if verbose:
        print(f'prefixes: {list(files_with_prefix.keys())}')
        print(f'weight: {datasets_weight}')
        print(f'Total: {len(dataset)} samples')
    
    return dataset



def extract_dataset_name(filename):
    """
    Extracts the dataset name from the provided filename.
    
    Args:
    filename (str): The filename from which to extract the dataset name.
    
    Returns:
    str or None: The extracted dataset name, or None if no valid name is found.
    """
    # 正则表达式匹配数据集名称
    match = re.match(r"\d+_(.*?)_\d+\.parquet", filename)
    if match:
        return match.group(1)  # 返回匹配到的数据集名称
    return None  # 如果没有匹配到，返回 None
    

from transformers import PreTrainedTokenizer

def build_with_untokenized_data(all_files: List[str], 
                                config: dict,
                                tokenizer: PreTrainedTokenizer,
                                context_size: int = 2048,
                                verbose: bool = False,
                                seed: int = 1234):
    """
    all_files: List[str]: A list of parquet file paths to the untokenized data files.
    there is only a text column in the parquet file.
    
    config: dict like:
    {
        dataset_name: {
            "type": "number" or "rate",
            "train": int or float,
            "validation": int or float,
            "train_weight": float (ratio of the dataset)
            "validation_weight": float (ratio of the dataset)
        },
        ...
    }
    """
    
    random.seed(seed)
    
    dataset_files = {}
    
    for filename in all_files:
        dataset_name = extract_dataset_name(os.path.basename(filename))
        if dataset_name is None:
            print(f"warning: No dataset name found in {filename}")
            continue
        
        if dataset_name in config.keys():
            if dataset_name not in dataset_files:
                dataset_files[dataset_name] = []
            dataset_files[dataset_name].append(filename)
            
    train_datasets = []
    val_datasets = []
    train_weights = []
    val_weights = []
    
    for dataset_name, entry in config.items():
        files = dataset_files.get(dataset_name, None)
        if files is None or len(files) == 0:
            print(f"warning: No files found for dataset {dataset_name}")
            continue
        
        if entry['type'] == 'number':
            train_count = entry['train']
            validation_count = entry.get('validation', len(files) - train_count)
        elif entry['type'] == 'rate':
            train_count = int(len(files) * entry['train'])
            validation_count = int(len(files) * entry['validation']) if 'validation' in entry else len(files) - train_count
        
        total_count = train_count + validation_count
        if total_count > len(files):
            raise ValueError(f"Not enough files for dataset {dataset_name} to select {total_count} files (only {len(files)} available).")
            
        selected_files = random.sample(files, total_count)
        
        split_point = train_count - 1 if validation_count <= 0 else train_count
        train_files = selected_files[:split_point]
        validation_files = selected_files[split_point:total_count]
        
        train_dataset = load_dataset('parquet', data_files=train_files, split="train", streaming=True)
        train_dataset = train_dataset.shuffle(seed=seed, buffer_size=4000)
        train_dataset = TextWrapper(train_dataset, tokenizer=tokenizer, add_bos=False,
                                    context_size=context_size, loop=True)
        train_datasets.append(train_dataset)
        train_weights.append(entry['train_weight'])
        
        val_dataset = load_dataset('parquet', data_files=validation_files, split="train", streaming=True)
        val_dataset = val_dataset.shuffle(seed=seed, buffer_size=4000)
        val_dataset = TextWrapper(val_dataset, tokenizer=tokenizer, add_bos=False,
                                  context_size=context_size, loop=False)
        val_datasets.append(val_dataset)
        val_weights.append(entry.get('validation_weight', entry['train_weight']))
        
    # normalize the weights
    sum_train_weight = sum(train_weights)
    train_weights = [w / sum_train_weight for w in train_weights]
    sum_val_weight = sum(val_weights)
    val_weights = [w / sum_val_weight for w in val_weights]    
    
    train_dataset = CombinedDataset(train_datasets, train_weights, seed=seed)
    val_dataset = CombinedDataset(val_datasets, val_weights, seed=seed)
    
    if verbose:
        print(f'Datasets: {list(dataset_files.keys())}')
        print(f'Train weight: {train_weights}')
        print(f'Validation weight: {val_weights}')
    
    return train_dataset, val_dataset    


from transformers import AutoTokenizer

if __name__ == "__main__":
    data_path = "/data/xllm_pretrain_data/text_data/processed"
    num_samples = 5
    files = match_files(data_path, '*.parquet')
    config_path = 'dataset_config/untokenized_data.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    tokenizer_path = 'tokenizer3'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    train_dataset, val_dataset = build_with_untokenized_data(files, config, tokenizer, context_size=2048, verbose=True)
    
    i = 0
    for sample in train_dataset:
        print(sample)
        i += 1
        if i >= num_samples:
            break
        
    i = 0
    for sample in val_dataset:
        print(sample)
        i += 1
        if i >= num_samples:
            break
    