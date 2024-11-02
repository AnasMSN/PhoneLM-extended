from typing import BinaryIO, Dict, List, Optional, Tuple, Union, Callable, Generator 
from dataclasses import dataclass, field
import gzip
import tarfile
import json
import struct
import numpy as np
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, wait
from torch.utils.data import IterableDataset, get_worker_info
import os
import random
from transformers import AutoTokenizer
import hashlib

DATATYPE_2_CODE: Dict[str, int] = {
    'i2': 0,  
    'i4': 1,
    'i8': 2,
}

CODE_2_DATATYPE: Dict[int, str] = {v: k for k, v in DATATYPE_2_CODE.items()}

@dataclass
class Header:
    HEADER_SIZE: int = field(default=struct.calcsize('4siiii'), init=False, repr=True)
    MAGIC: bytes = field(default=b'XLLM', init=False, repr=True)
    VERSION: int = field(default=1, init=False, repr=True)

    num_tokens: int   # all tokens including special tokens
    num_special_tokens: int
    dtype: str 

    @classmethod
    def from_file(cls, file: BinaryIO) -> 'Header':
        data = file.read(cls.HEADER_SIZE)
        if len(data) != cls.HEADER_SIZE:
            raise ValueError('Invalid header size')

        unpacked_data = struct.unpack('<4siiii', data)
        if unpacked_data[0] != cls.MAGIC:
            raise ValueError('Invalid magic number')

        if unpacked_data[1] != cls.VERSION:
            raise ValueError('Invalid version')

        if unpacked_data[4] not in CODE_2_DATATYPE:
            raise ValueError('Invalid data type')

        return Header(num_tokens=unpacked_data[2],
                      num_special_tokens=unpacked_data[3],
                      dtype=CODE_2_DATATYPE[unpacked_data[4]])

    def to_file(self, file: BinaryIO):
        dtype_code = DATATYPE_2_CODE[self.dtype]
        # Ensure MAGIC and dtype_code are appropriately formatted
        data = struct.pack('<4siiii', self.MAGIC, self.VERSION, self.num_tokens, self.num_special_tokens, dtype_code)
        file.write(data)


class DataFile:
    def __init__(self, filename: str, max_size: int=50*1024*4096, dtype: str='i4'):
        self.filename = filename
        self.max_size = max_size
        self.dtype = dtype

    def __enter__(self):
        self.header = Header(num_special_tokens=0, num_tokens=0, dtype=self.dtype)
        self.buffer = np.empty((self.max_size,), dtype=self.dtype)
        self.idx = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with open(self.filename, 'wb') as f:
            self.header.to_file(f)
            self.buffer[:self.idx].tofile(f)

    def prepare(self):
        self.__enter__()

    def close(self):
        if getattr(self, 'closed', False):
            return

        self.__exit__(None, None, None)
        self.closed = True

    def remaining_size(self):
        return self.max_size - self.idx

    def write_array(self, array: np.ndarray, num_special_tokens: int=2):
        l = len(array)
        if self.idx + l > self.max_size:
            raise ValueError('Buffer overflow')

        assert array.dtype == self.dtype,  "Invalid array type"
        assert array.ndim == 1, "Array must be 1D"

        self.buffer[self.idx:self.idx + l] = array
        self.idx += l
        self.header.num_tokens += l
        self.header.num_special_tokens += num_special_tokens

    @staticmethod
    def get_header(filename: str) -> Header:
        with open(filename, 'rb') as f:
            return Header.from_file(f)

    @staticmethod
    def open(filename: str)-> Tuple[Header, np.memmap]:
        with open(filename, 'rb') as f:
            header = Header.from_file(f)
        mmap = np.memmap(filename, dtype=header.dtype, mode='r', offset=Header.HEADER_SIZE)
        return header, mmap


def handle_parquet(file: str, fn: Optional[Callable]=None):
    parquet_file = pq.ParquetFile(file)

    # iterate over all row groups in the parquet file
    for batch in parquet_file.iter_batches(batch_size=1000):
        # convert to pandas DataFrame
        res = batch.to_pylist()

        if fn is not None:
            res = map(fn, res)

        for item in res:
            yield item


def handle_jsonl(file: str, fn: Optional[Callable]=None) -> Generator[dict, None, None]:
    """
    Process a JSON Lines file, which can be plain, gzipped, or within a gzipped tar archive,
    and optionally apply a function to each JSON object.

    :param file: The path to the JSONL file.
    :param fn: An optional function to apply to each JSON object.
    :return: A generator that yields processed items.
    """
    # Check if the file is a gzipped tar archive
    if file.endswith('.tar.gz'):
        # Open the tar.gz file
        with tarfile.open(file, 'r:gz') as tar:
            # Iterate over each member in the tar archive
            for member in tar.getmembers():
                # We assume there's only one JSONL file in the archive
                if member.name.endswith('.jsonl'):
                    # Extract and read the JSONL file
                    f = tar.extractfile(member)
                    if f:
                        with f:
                            for line in f:
                                item = json.loads(line.decode('utf-8').strip())
                                if fn:
                                    item = fn(item)
                                yield item
    elif file.endswith('.gz'):
        # Open the gzipped JSON Lines file
        with gzip.open(file, 'rt', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if fn:
                    item = fn(item)
                yield item
    else:
        # Open the normal JSON Lines file
        with open(file, 'rt', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if fn:
                    item = fn(item)
                yield item
                

def handle_json_file(file: str, fn: Optional[Callable]=None) -> Generator[dict, None, None]:
    """
    Process a JSON file, which can be plain, gzipped, or within a gzipped tar archive,
    and optionally apply a function to each JSON object.

    :param file: The path to the JSON file.
    :param fn: An optional function to apply to each JSON object.
    :return: A generator that yields processed items.
    """
    # Check if the file is a gzipped tar archive
    if file.endswith('.tar.gz'):
        # Open the tar.gz file
        with tarfile.open(file, 'r:gz') as tar:
            # Iterate over each member in the tar archive
            for member in tar.getmembers():
                # We assume there's only one JSON file in the archive
                if member.name.endswith('.json'):
                    # Extract and read the JSON file
                    f = tar.extractfile(member)
                    if f:
                        with f:
                            item_arr = json.load(f)
                            for item in item_arr:
                                if fn:
                                    item = fn(item)
                                yield item
    elif file.endswith('.gz'):
        # Open the gzipped JSON file
        with gzip.open(file, 'rt', encoding='utf-8') as f:
            item_arr = json.load(f)
            for item in item_arr:
                if fn:
                    item = fn(item)
                yield item
            
    else:
        # Open the normal JSON file
        with open(file, 'rt', encoding='utf-8') as f:
            item_arr = json.load(f)
            for item in item_arr:
                if fn:
                    item = fn(item)
                yield item

                
import json

def handle_text_file(file: str, fn: Optional[Callable]=None):
    with open(file, 'rt', encoding='utf-8') as f:
        try:
            formatted = json.dumps((json.load(f)), indent=4)
        except json.JSONDecodeError:
            formatted = ''
    
    if formatted == '':
        from lxml import etree
        try:
            tree = etree.parse(file)
            formatted = etree.tostring(tree, pretty_print=True).decode('utf-8')
        except etree.XMLSyntaxError:
            formatted = ''
    
    if formatted == '':
        with open(file, 'rt', encoding='utf-8') as f:
            formatted = f.read()
    
    res = {'text': formatted}
    if fn is not None:
        res = fn(res)
    yield res


HANDLER = {
    'parquet': handle_parquet,
    'jsonl': handle_jsonl,
    'text': handle_text_file,
    'xml': handle_text_file,
    'json': handle_json_file,
}

def handle(file: str, fn: Optional[Callable]=None, handler: str='jsonl'):
    if handler not in HANDLER:
        raise ValueError(f'Invalid handler: {handler}')
    
    return HANDLER[handler](file, fn)


from transformers import PreTrainedTokenizer
from tqdm import tqdm

def split_n(lst, n):
    k, m = divmod(len(lst), n)
    return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


class DataFileBuilder:

    def __init__(self, files: List[str], out_dir: str, prefix: str, tokenizer: str,
                handler: str='jsonl',
                fn: Optional[Callable]=None,
                add_bos: bool=True,
                add_eos: bool=True,
                max_size: int=50*4*1024*1024):
        self.files = files
        self.out_dir = out_dir
        self.prefix = prefix
        self.max_size = max_size
        self.tokenizer = tokenizer
        self.fn = fn
        self.handle = HANDLER[handler]
        self.add_bos = add_bos
        self.add_eos = add_eos
        print(f'Building dataset with {len(files)} files, max size: {max_size/1024/1024} M')
        
    @staticmethod
    def _signature(message: str):
        # generate sha256 hash
        return hashlib.sha256(message.encode()).hexdigest()

    def _process_files(self, files: List[str], uid: int):
        try:
            cnt = 0
            # we should also resume from the last file
            while os.path.exists(f'{self.out_dir}/{self.prefix}-{uid:03d}-{cnt:05d}.data'):
                cnt += 1
            
            out_file = DataFile(f'{self.out_dir}/{self.prefix}-{uid:03d}-{cnt:05d}.data', max_size=self.max_size)
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
            out_file.prepare()
            total_files = len(files)
            print(f'task {uid}: started to process {total_files} files')
            for idx, file in enumerate(files):
                # check if file is already processed
                sig = self._signature(file)
                if os.path.exists(f'{self.out_dir}/.build_data/{sig}'):
                    print(f'task {uid}: {file} is already processed')
                    continue
                for item in self.handle(file, fn=self.fn):
                    if item is None:  # fn can return None to skip item, this can be used to filter items
                        continue
                    content = item['text']
                    tokens = tokenizer.encode(content, add_special_tokens=False)

                    if self.add_bos:
                        tokens = [tokenizer.bos_token_id] + tokens

                    if self.add_eos:
                        tokens = tokens + [tokenizer.eos_token_id]

                    while len(tokens) > 0:
                        l = min(len(tokens), out_file.remaining_size())
                        special_tokens = 0
                        if tokens[0] == tokenizer.bos_token_id:
                            special_tokens += 1
                        if tokens[l-1] == tokenizer.eos_token_id:
                            special_tokens += 1
                        out_file.write_array(np.array(tokens[:l], dtype='i4'), num_special_tokens=special_tokens)
                        tokens = tokens[l:]

                        if out_file.remaining_size() == 0:
                            out_file.close()
                            cnt += 1
                            out_file = DataFile(f'{self.out_dir}/{self.prefix}-{uid:03d}-{cnt:05d}.data', max_size=self.max_size)
                            out_file.prepare()

                print(f'task {uid}: processed {idx + 1}/{total_files} files')

                # mark file as processed
                # create an empty file with the name of signature in out_dir/.build_data
                with open(f'{self.out_dir}/.build_data/{sig}', 'w') as f:
                    f.write('')

                    
            out_file.close()
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            raise Exception(f"An error occurred: {e}\nTraceback: {err}")

    def build(self, num_process: int=os.cpu_count()):
        random.shuffle(self.files)

        # create output directory recursively if not exists
        os.makedirs(self.out_dir, exist_ok=True)
        
        # create out_dir/.build_data to mark which files have been processed
        # if file is processed, then an empty file with the same name will be created in this directory
        os.makedirs(os.path.join(self.out_dir, ".build_data"), exist_ok=True)

        n_splits = split_n(self.files, num_process * 2)
        with ProcessPoolExecutor(max_workers=num_process) as executor:
            futures = []
            for i, files in enumerate(n_splits):
                if len(files) == 0:
                    print(f"task {i} has no files to process")
                    continue
                futures.append(executor.submit(self._process_files, files, i))
            wait(futures)

             # 检查可能发生的异常
            for future in futures:
                if future.exception():
                    print(f"exception: {future.exception()}")

import copy
import torch

def binary_search_first_greater(arr, target):
    left, right = 0, len(arr) - 1
    result = len(arr)  # Default to length of array if all elements are <= target
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] > target:
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    return result

class PackedDataset(IterableDataset):

    def __init__(self,
                 files: List[str],
                 rank, world_size,
                 n_buffered_files: int=0,
                 sample_size: int=8192,
                 # if true, we don't need to split the dataset manually since data will be dispatch to each process later
                 # if false, we need to split the dataset manually, echo process will read directly from the dataset so if we don't split the dataset, the data will be duplicated
                 dispatch=False,
                 skip_steps: int = 0,
                 seed: int=42):
        files = sorted(files)
        self.seed = seed
        random.seed(seed)
        random.shuffle(files)
        self.files = [(f, DataFile.get_header(f)) for f in tqdm(files, desc='loading files metadata')]
        self.total_tokens = sum((h.num_tokens // sample_size * sample_size) for _, h in self.files)
        # some bug exist here, cannot calculate special tokens used in training correctly, so we commented the following 2 line
        # self.special_tokens = sum((h.num_special_tokens // sample_size * sample_size) for _, h in self.files)
        # self.effect_tokens = self.total_tokens - self.special_tokens
        self.sample_size = sample_size
        self.n_buffered_files = n_buffered_files
        print(
            f"Total tokens: {self.total_tokens}, rank: {rank}"
        )
        print(f'Sample Size: {sample_size} tokens; Number of samples: {len(self)}')
        self.rank = rank
        self.world_size = world_size
        assert rank < world_size, 'rank must be less than world_size'
        self.dispatch = dispatch
        self.skip_steps = skip_steps
        
    def set_skip_steps(self, skip_steps: int):
        self.skip_steps = skip_steps
        
    @property
    def remaining_steps(self):
        worker_info = get_worker_info()
        # if we don't use dataloader, worker_info will be None
        num_workers = worker_info.num_workers if worker_info is not None else 1 # subprocesses used by a DataLoader
        worker_id = worker_info.id if worker_info is not None else 0  # id of the current worker
        if not self.dispatch:
            # split files manually
            num_shards = self.world_size * num_workers
            shard_id = self.rank * num_workers + worker_id
        else:
            num_shards = num_workers
            shard_id = worker_id

        files_shards = list(split_n(self.files, num_shards))

        files = copy.deepcopy(files_shards[shard_id])
        
        total_steps = sum((h.num_tokens // self.sample_size) for _, h in files)
        return total_steps - self.skip_steps

    def __iter__(self):
        worker_info = get_worker_info()
        # if we don't use dataloader, worker_info will be None
        num_workers = worker_info.num_workers if worker_info is not None else 1 # subprocesses used by a DataLoader
        worker_id = worker_info.id if worker_info is not None else 0  # id of the current worker
        if not self.dispatch:
            # split files manually
            num_shards = self.world_size * num_workers
            shard_id = self.rank * num_workers + worker_id
        else:
            num_shards = num_workers
            shard_id = worker_id

        files_shards = list(split_n(self.files, num_shards))

        files = copy.deepcopy(files_shards[shard_id])
        
        if self.n_buffered_files <= 1:
            accumulated_steps = [0 for _ in range(len(files))]
            if not accumulated_steps:
                # print(f"worker {worker_id} has no files to process")
                return
            accumulated_steps[0] = files[0][1].num_tokens // self.sample_size
            for i, (_, header) in enumerate(files[1:]):
                accumulated_steps[i] = header.num_tokens // self.sample_size + accumulated_steps[i-1]
            
            skip_steps = self.skip_steps // num_workers
            remainder = self.skip_steps % num_workers
            if worker_id == 0:
                skip_steps += remainder
            
            # find the exact file to read data from
            # TODO: optimize this by using binary search
            # skipped_steps = 0
            # for i, accumulated_step in enumerate(accumulated_steps):
            #     if accumulated_step > skip_steps:
            #         file_index = i
            #         break
            #     skipped_steps = accumulated_step
            
            file_index = binary_search_first_greater(accumulated_steps, skip_steps)
            if file_index >= len(files):
                return
            skipped_steps = accumulated_steps[file_index - 1] if file_index > 0 else 0
            
            sample_size = self.sample_size
            remained_steps = skip_steps - skipped_steps
            pos = remained_steps * sample_size
            # print(f"worker {worker_id} start from file {file_index}, pos {pos}")
            for filename, header in files[file_index:]:
                h, mmap = DataFile.open(filename)
                while pos + sample_size <= h.num_tokens:
                    t = torch.from_numpy(mmap[pos:pos + sample_size].astype(np.int64))
                    yield {'input_ids': t, 'labels': t.clone()}
                    pos += sample_size
                pos = 0
        else:
            random.seed(self.seed)
            assert self.skip_steps <= 0, "skip_steps > 0 when buffered file > 0 and use random sampling"
            buff_size = self.n_buffered_files
            file_buffer = []
            sample_size = self.sample_size

            # enqueue files
            while len(file_buffer) < buff_size and len(files) > 0:
                f, _ = files.pop()
                h, mmap = DataFile.open(f)
                tokens = h.num_tokens
                file_buffer.append([tokens, mmap, 0])
                assert tokens == len(mmap), 'Invalid file'


            while file_buffer:
                idx = random.randint(0, len(file_buffer) - 1)
                tokens, mmap, pos = file_buffer[idx]
                if pos + sample_size <= tokens:
                    t = torch.from_numpy(mmap[pos:pos + sample_size].astype(np.int64))
                    yield {'input_ids': t, 'labels': t.clone()}
                    file_buffer[idx][2] = pos + sample_size
                else:
                    file_buffer.pop(idx)

                if len(files) > 0 and len(file_buffer) < buff_size:
                    f, _ = files.pop()
                    h, mmap = DataFile.open(f)
                    tokens = h.num_tokens
                    file_buffer.append([tokens, mmap, 0])
                    assert tokens == len(mmap), 'Invalid file'

    def __len__(self):
        return self.total_tokens // self.sample_size

if __name__ == '__main__':
    from transformers import AutoTokenizer
    from build_datasets.text_datasets import match_files
    
    tokenizer = AutoTokenizer.from_pretrained('tokenizer')
    
    files = match_files('../datasets/data/starcoderdata', '*.data')
    
    datasets = PackedDataset(files, sample_size=2048)
    print(len(datasets))
    
    i = 0
    n = 1
    for item in datasets:
        print(item)
        ids = item['input_ids']
        print(tokenizer.decode(ids))
        i += 1
        if i >= n:
            break
