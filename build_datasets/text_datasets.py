from datasets import load_dataset, DatasetDict, Dataset
from pathlib import Path
import random
from typing import List


def match_files(file_dir: str, pattarn: str):
    # return all files that match the pattern in the dir path
    file_dir = Path(file_dir)
    return [str(filename) for filename in file_dir.rglob(pattarn)]


import pyarrow.parquet as pq

def parquet_files_generator(files: List[str]):
    for file in files:
        parquet_file = pq.ParquetFile(file)
        
        # iterate over all row groups in the parquet file
        for batch in parquet_file.iter_batches(batch_size=1000, columns=['content']):
            # convert to pandas DataFrame
            df = batch.to_pandas()
            
            for content in df['content']:
                yield {'text': content}
                
def starcoderdata_dataset(path: str,
                  validation_split: float = 0.1):
    
    train_files, validation_files = split_data_files(path, validation_split=validation_split, pattern='*.parquet')

    return {
        "train": parquet_files_generator(train_files),
        "validation": parquet_files_generator(validation_files)
    }
    


def split_data_files(base_path, validation_split=0.1, pattern: str = '*.jsonl'):
    # 列出目录中所有的jsonl文件
    all_files = list(match_files(base_path, pattern))
    # 根据文件名进行排序确保有序
    random.shuffle(all_files)

    # 计算切割点
    split_point = int(len(all_files) * (1 - validation_split))

    if split_point == len(all_files):
        split_point -= 1

    # 分割文件名为训练和验证
    train_files = all_files[:split_point]
    validation_files = all_files[split_point:]

    assert len(train_files) > 0, 'No training files found'
    assert len(validation_files) > 0, 'No validation files found'

    return train_files, validation_files


def build_dataset(path: str, shuffle: bool = False, buffer_size: int = 10000,
                  validation_split: float = 0.1, pattern: str = '*.jsonl', data_type: str = 'json') -> DatasetDict:
    # 获取训练集和验证集文件列表
    train_files, validation_files = split_data_files(path, validation_split=validation_split, pattern=pattern)

    # 加载数据集
    data_files = {
        'train': train_files,
        'validation': validation_files
    }

    # 加载数据集，指定 streaming=True 来启用流式处理
    dataset = load_dataset(data_type, data_files=data_files, streaming=True)

    if shuffle:
        dataset = dataset.shuffle(seed=42, buffer_size=buffer_size)

    return dataset


def wanjuan_text(examples):
    return {'text': [f'{title} {content}' for title, content in zip(examples['title'], examples['content'])]}

def get_content(example):
    return {'text': example['content']}


def build_wanjuan_cc(path: str, shuffle: bool = False, buffer_size: int = 10000,
                     validation_split: float = 0.1) -> DatasetDict:
    dataset = build_dataset(path, shuffle=shuffle, buffer_size=buffer_size, validation_split=validation_split)
    return dataset.map(wanjuan_text, batched=True,
                       remove_columns=['id', 'content', 'title', 'language', 'date', 'token_num', 'cbytes_num',
                                       'line_num', 'char_num', 'toxic_score',
                                       'fluency_score', 'not_ad_score', 'porn_score'])


def build_sky_pile(path: str, shuffle: bool = False, buffer_size: int = 10000,
                   validation_split: float = 0.1) -> DatasetDict:
    return build_dataset(path, shuffle, buffer_size, validation_split)


fake_texts = {
    'text': [
        'This is a fake text',
        'This is another fake text',
        'How many fake texts are there?',
        'This is the last fake text',
    ]
}


def build_fake_texts() -> DatasetDict:
    dataset = Dataset.from_dict(fake_texts)
    # train and validation split are the same
    return DatasetDict(train=dataset, validation=dataset)


if __name__ == '__main__':
    pass
