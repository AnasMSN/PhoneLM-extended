from jsonargparse import CLI
from build_datasets import DataFileBuilder
from build_datasets.text_datasets import get_content, match_files

PATTERN_MAP = {
    'parquet': ['*.parquet'],
    'jsonl': ['*.jsonl', '*json', '*.json.gz', '*.jsonl.gz', '*.json.tar.gz', '*.jsonl.tar.gz'],   
    'xml': ['*.xml'],
    'json': ['*.json'],
    'text': ['*.txt', '*.json', '*.xml'],
}

FIELD_FN_MAP = {
    'content': get_content,
    'text': None,
}

def main(source_dir: str,
         out_dir: str,
         prefix: str='data',
         handler: str='parquet',
         field: str='content',
         num_workers: int=16,
         tokenizer_path: str='tokenizer_qwen2',
         max_size: int=200 * 1024 * 1024):
    pattern = PATTERN_MAP.get(handler, None)
    if pattern is None:
        raise ValueError(f'Handler {handler} not supported')
    all_files = []
    for p in pattern:
        all_files.extend(match_files(source_dir, p))
    
    print(f'tokenize field: {field}')
    builder = DataFileBuilder(all_files, out_dir, prefix, 
                              tokenizer=tokenizer_path, handler=handler, 
                              add_bos=False, add_eos=True,
                              fn=FIELD_FN_MAP[field],
                              max_size=max_size)
    
    print(f'Building dataset with {num_workers} workers')
    builder.build(num_workers)

if __name__ == '__main__':
    CLI(main)
