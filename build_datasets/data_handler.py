import os
import random
import dataclasses

from datatrove.pipeline.readers import JsonlReader, ParquetReader
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import ParquetWriter
from datatrove.data import Document
from datatrove.data import DocumentsPipeline

from abc import ABC, abstractmethod
from typing import Callable, Dict, Union, List, Iterable
from datatrove.io import DataFolderLike
from datatrove.pipeline.readers.base import BaseDiskReader
from datatrove.utils.logging import logger

from typing import Any, Literal
from string import Template

default_input_start = "<|im_start|>"
default_input_end = "<|im_end|>"

TEMPLATE = Template(f"{default_input_start}$role:\n $content{default_input_end}\n")

def apply_chat_template(chat, start=default_input_start, end=default_input_end):
    """
    chat is a dict in the form of:
    [
        {"role": "...", "content": ...},
        ...
    ]
    """
    return ''.join([TEMPLATE.substitute(role=chat["role"], content=chat["content"]) for chat in chat])

class TextReader(BaseDiskReader):
    """Read data from Text files.
        Will read the entire text file as a separate document.

    Args:
        data_folder: the data folder to read from
        compression: the compression to use (default: "infer")
        limit: limit the number of JSON lines to read
        skip: skip the first n items
        file_progress: show progress bar for files
        doc_progress: show progress bar for documents
        adapter: function to adapt the data dict from the source to a Document.
            Take as input: data: dict, path: str, id_in_file: int | str
            Return: a dict with at least a "text" key
        text_key: key to use for the text in the default adapter (default: "text"). Ignored if you provide your own `adapter`
        id_key: key to use for the id in the default adapter (default: "id"). Ignored if you provide your own `adapter`
        default_metadata: default metadata to add to all documents
        recursive: if True, will read files recursively in subfolders (default: True)
        glob_pattern: a glob pattern to filter files to read (default: None)
        shuffle_files: shuffle the files within the returned shard. Mostly used for data viz. purposes, do not use
            with dedup blocks
    """

    name = "🐿 Text"

    def __init__(
        self,
        data_folder: DataFolderLike,
        compression: Literal["infer", "gzip", "zstd"] | None = "infer",
        limit: int = -1,
        skip: int = 0,
        file_progress: bool = False,
        doc_progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        shuffle_files: bool = False,
    ):
        super().__init__(
            data_folder=data_folder,
            limit=limit,
            skip=skip,
            file_progress=file_progress,
            doc_progress=doc_progress,
            adapter=adapter,
            text_key=text_key,
            id_key=id_key,
            default_metadata=default_metadata,
            recursive=recursive,
            glob_pattern=glob_pattern,
            shuffle_files=shuffle_files,
        )
        self.compression = compression

    def read_file(self, filepath: str):
        try:
            with self.data_folder.open(filepath, "r", compression=self.compression) as f:
                content = f.read()
                document = self.get_document_from_dict({"text": content}, filepath, 0)
                yield document
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
            return

import json
from json import JSONDecodeError

class JsonReader(BaseDiskReader):
    """Read data from JSON files.
        Json should be in the format like: [ {...}, {...}, ...].
        i.e. a list of dictionaries, where each dictionary represents a document.

    Args:
        data_folder: the data folder to read from
        compression: the compression to use (default: "infer")
        limit: limit the number of JSON lines to read
        skip: skip the first n rows
        file_progress: show progress bar for files
        doc_progress: show progress bar for documents
        adapter: function to adapt the data dict from the source to a Document.
            Take as input: data: dict, path: str, id_in_file: int | str
            Return: a dict with at least a "text" key
        text_key: key to use for the text in the default adapter (default: "text"). Ignored if you provide your own `adapter`
        id_key: key to use for the id in the default adapter (default: "id"). Ignored if you provide your own `adapter`
        default_metadata: default metadata to add to all documents
        recursive: if True, will read files recursively in subfolders (default: True)
        glob_pattern: a glob pattern to filter files to read (default: None)
        shuffle_files: shuffle the files within the returned shard. Mostly used for data viz. purposes, do not use
            with dedup blocks
    """

    name = "🐿 Json"

    def __init__(
        self,
        data_folder: DataFolderLike,
        compression: Literal["infer", "gzip", "zstd"] | None = "infer",
        limit: int = -1,
        skip: int = 0,
        file_progress: bool = False,
        doc_progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        shuffle_files: bool = False,
    ):
        super().__init__(
            data_folder=data_folder,
            limit=limit,
            skip=skip,
            file_progress=file_progress,
            doc_progress=doc_progress,
            adapter=adapter,
            text_key=text_key,
            id_key=id_key,
            default_metadata=default_metadata,
            recursive=recursive,
            glob_pattern=glob_pattern,
            shuffle_files=shuffle_files,
        )
        self.compression = compression

    def read_file(self, filepath: str):
        with self.data_folder.open(filepath, "r", compression=self.compression) as f:
            try:
                documents = json.load(f)
            except (EOFError, JSONDecodeError) as e:
                logger.warning(f"Error when reading `{filepath}`: {e}")
        for i, doc in enumerate(documents):
            yield self.get_document_from_dict(doc, filepath, i)

class DataHandler(ABC):
    name: str = None
    type: str = "default"
    
    reader_map = {
        "jsonl": JsonlReader,
        "parquet": ParquetReader,
        "text": TextReader,
        "json": JsonReader,
        "default": JsonlReader
    }
    
    def __init__(self, data_path: str, output_path: str, **kw_args):
        self.data_path = data_path
        self.output_path = output_path
    
    @abstractmethod
    def get_writer_adapter(self)->Callable[[Any, Document], dict]:
        raise NotImplemented
    
    @abstractmethod
    def get_reader_adapter(self)->Callable[[Any, dict, str, Union[int, str]], dict]:
        raise NotImplemented
    
    @abstractmethod
    def reader_kwargs(self)->dict:
        raise NotImplemented
    
    @abstractmethod
    def writer_kwargs(self)->dict:
        raise NotImplemented
    
    @abstractmethod
    def executor_kwargs(self)->dict:
        raise NotImplemented
    
    # the following 4 methods can be overwrite by subclass
    def get_reader(self, data_path: str)->PipelineStep:
        reader_cls = self.reader_map[self.type]
        adapter = self.get_reader_adapter()
        kw = self.reader_kwargs()
        return reader_cls(data_path, adapter=adapter, **kw)
    
    def get_writer(self, output_path: str)->PipelineStep:
        adapter = self.get_writer_adapter()
        kw = self.writer_kwargs()
        writer_cls = ParquetWriter
        return writer_cls(output_path, adapter=adapter, **kw)
    
    def middle_steps(self)->List[PipelineStep]:
        return []
    
    def pipelines(self) -> List[PipelineStep]:
        # 创建读取数据的步骤
        reader_step = self.get_reader(self.data_path)

        # 获取中间处理步骤（如果有的话）
        middle_steps = self.middle_steps()

        # 创建写入数据的步骤
        writer_step = self.get_writer(self.output_path)

        # 将所有步骤合并成一个列表并返回
        return [reader_step] + middle_steps + [writer_step]
    

class MagicoderHandler(DataHandler):
    name = "Magicoder"
    type = "jsonl"
    
    def get_reader_adapter(self):
        
        def magicoder_reader_adapter(self, data: dict, path: str, id_in_file: int | str) -> dict:
            """
            The Magicoder data adapter to adapt input data into the datatrove Document format

            Args:
                data: a dictionary with the "raw" representation of the data
                path: file path or source for this sample
                id_in_file: its id in this particular file or source

            Returns: a dictionary with text, id, media and metadata fields

            """
            
            metadata = data.pop("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"metadata": metadata}
            
            instruction = {
                "zh": "你是一位经验丰富的程序员，请根据我的需求帮助我编写代码。",
                "en": "You are an experienced programmer, please help me write some code according to my needs."
            }
            
            chat = [
                {
                "role": "instruction",
                "content": instruction["en"] if random.random() < 0.8 else instruction["zh"]
                },
                {"role": "user", "content": data["problem"]},
                {"role": "assistant", "content": data["solution"]}
            ]
            
            metadata["chat"] = chat
            
            return {
                "text": apply_chat_template(chat),
                "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
                "media": data.pop("media", []),
                "metadata": metadata | data,  # remaining data goes into metadata
            }
        
        return magicoder_reader_adapter
    
    def get_writer_adapter(self):
        def magicoder_writer_adapter(self, document: Document) -> dict:
            """
            You can create your own adapter that returns a dictionary in your preferred format
            Args:
                document: document to format

            Returns: a dictionary to write to disk

            """
            data = {key: val for key, val in dataclasses.asdict(document).items() if val}
            return {
                "id": data["id"], 
                "text": data["text"],
                "chat": data["metadata"]["chat"]
            }
        
        return magicoder_writer_adapter
    
    def reader_kwargs(self):
        return {
            "default_metadata": {"dataset_name": self.name},
            "glob_pattern": "*.jsonl"
        }
        
    def writer_kwargs(self):
        return {
            "max_file_size": 400 * 1024 * 1024,
            "output_filename": "${dataset_name}_${rank}.parquet"
        }
        
    def executor_kwargs(self) -> Dict:
        return {
            "tasks": 1,
            "logging_dir": os.path.join("logs", self.name)
        }
        
class COIG_CQIAHandler(DataHandler):
    name = "COIG-CQIA"
    type = "jsonl"
    
    def get_reader_adapter(self):
        
        file_set = set()
        
        def reader_adapter(self, data: dict, path: str, id_in_file: int | str) -> dict:
            """
            The COIG_CQIA data adapter to adapt input data into the datatrove Document format

            Args:
                data: a dictionary with the "raw" representation of the data
                path: file path or source for this sample
                id_in_file: its id in this particular file or source

            Returns: a dictionary with text, id, media and metadata fields

            """
            
            metadata = data.pop("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"metadata": metadata}
            
            chat = []
            inst = data.get("instruction", "")
            if len(inst) > 0:
                chat.append({"role": "instruction", "content": inst})
                
            inp = data.get("input")
            if len(inp) > 0:
                chat.append({"role": "user", "content": inp})
                
            chat.append({"role": "assistant", "content": data["output"]})
            
            metadata["chat"] = chat
            
            if path not in file_set:
                print(f"Processing file {path}")
                file_set.add(path)
            
            return {
                "text": apply_chat_template(chat),
                "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
                "media": data.pop("media", []),
                "metadata": metadata | data,  # remaining data goes into metadata
            }
        
        return reader_adapter
    
    def get_writer_adapter(self):
        def writer_adapter(self, document: Document) -> dict:
            """
            You can create your own adapter that returns a dictionary in your preferred format
            Args:
                document: document to format

            Returns: a dictionary to write to disk

            """
            data = {key: val for key, val in dataclasses.asdict(document).items() if val}
            return {
                "id": data["id"], 
                "text": data["text"],
                "chat": data["metadata"]["chat"]
            }
        
        return writer_adapter
    
    def reader_kwargs(self):
        return {
            "default_metadata": {"dataset_name": self.name},
            "glob_pattern": "*.jsonl"
        }
        
    def writer_kwargs(self):
        return {
            "max_file_size": 400 * 1024 * 1024,
            "output_filename": "${dataset_name}_${rank}.parquet"
        }
        
    def executor_kwargs(self) -> Dict:
        return {
            "tasks": 4,
            "logging_dir": os.path.join("logs", self.name)
        }
        
        
class SlimOrcaHandler(DataHandler):
    name = "SlimOrca"
    type = "jsonl"
    
    def get_reader_adapter(self):
        
        def reader_adapter(self, data: dict, path: str, id_in_file: int | str) -> dict:
            """
            The SlimOrca data adapter to adapt input data into the datatrove Document format

            Args:
                data: a dictionary with the "raw" representation of the data
                path: file path or source for this sample
                id_in_file: its id in this particular file or source

            Returns: a dictionary with text, id, media and metadata fields

            """
            
            metadata = data.pop("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"metadata": metadata}
            
            chat = []
            
            for item in data["conversations"]:
                chat.append({"role": item["from"], "content": item["value"]})
            
            metadata["chat"] = chat
            return {
                "text": apply_chat_template(chat),
                "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
                "media": data.pop("media", []),
                "metadata": metadata | data,  # remaining data goes into metadata
            }
        
        return reader_adapter
    
    def get_writer_adapter(self):
        def writer_adapter(self, document: Document) -> dict:
            """
            You can create your own adapter that returns a dictionary in your preferred format
            Args:
                document: document to format

            Returns: a dictionary to write to disk

            """
            data = {key: val for key, val in dataclasses.asdict(document).items() if val}
            return {
                "id": data["id"], 
                "text": data["text"],
                "chat": data["metadata"]["chat"]
            }
        
        return writer_adapter
    
    def reader_kwargs(self):
        return {
            "default_metadata": {"dataset_name": self.name},
            "glob_pattern": "*.jsonl"
        }
        
    def writer_kwargs(self):
        return {
            "max_file_size": 400 * 1024 * 1024,
            "output_filename": "${dataset_name}_${rank}.parquet"
        }
        
    def executor_kwargs(self) -> Dict:
        return {
            "tasks": 4,
            "logging_dir": os.path.join("logs", self.name)
        }

class UltraChatHandler(DataHandler):
    name = "ultrachat"
    type = "jsonl"
    
    def get_reader_adapter(self):
        files = set()
        def reader_adapter(self, data: dict, path: str, id_in_file: int | str) -> dict:
            """
            The SlimOrca data adapter to adapt input data into the datatrove Document format

            Args:
                data: a dictionary with the "raw" representation of the data
                path: file path or source for this sample
                id_in_file: its id in this particular file or source

            Returns: a dictionary with text, id, media and metadata fields

            """
            
            metadata = data.pop("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"metadata": metadata}
            
            conversations = data["data"]
            roles = ["user", "assistant"]
            chat = [
                {"role": roles[i % 2], "content": item} for i, item in enumerate(conversations)
            ]
            
            metadata["chat"] = chat
            
            if path not in files:
                print(f"Processing file {path}")
                files.add(path)
            
            return {
                "text": apply_chat_template(chat),
                "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
                "media": data.pop("media", []),
                "metadata": metadata | data,  # remaining data goes into metadata
            }
        
        return reader_adapter
    
    def get_writer_adapter(self):
        def writer_adapter(self, document: Document) -> dict:
            """
            You can create your own adapter that returns a dictionary in your preferred format
            Args:
                document: document to format

            Returns: a dictionary to write to disk

            """
            data = {key: val for key, val in dataclasses.asdict(document).items() if val}
            return {
                "id": data["id"], 
                "text": data["text"],
                "chat": data["metadata"]["chat"]
            }
        
        return writer_adapter
    
    def reader_kwargs(self):
        return {
            "default_metadata": {"dataset_name": self.name},
            "glob_pattern": "*.jsonl"
        }
        
    def writer_kwargs(self):
        return {
            "max_file_size": 400 * 1024 * 1024,
            "output_filename": "${dataset_name}_${rank}.parquet"
        }
        
    def executor_kwargs(self) -> Dict:
        return {
            "tasks": 4,
            "logging_dir": os.path.join("logs", self.name)
        }
        

class EvolInstructCodeHandler(DataHandler):
    name = "Evol-Instruct-Code-80k-v1"
    type = "json"
    
    def get_reader_adapter(self):
        files = set()
        def reader_adapter(self, data: dict, path: str, id_in_file: int | str) -> dict:
            """
            The SlimOrca data adapter to adapt input data into the datatrove Document format

            Args:
                data: a dictionary with the "raw" representation of the data
                path: file path or source for this sample
                id_in_file: its id in this particular file or source

            Returns: a dictionary with text, id, media and metadata fields

            """
            
            metadata = data.pop("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"metadata": metadata}
            
            chat = [
                {"role": "instruction", "content": data["instruction"]},
                {"role": "assistant", "content": data["output"]},
            ]
            
            
            metadata["chat"] = chat
            
            if path not in files:
                print(f"Processing file {path}")
                files.add(path)
            
            return {
                "text": apply_chat_template(chat),
                "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
                "media": data.pop("media", []),
                "metadata": metadata | data,  # remaining data goes into metadata
            }
        
        return reader_adapter
    
    def get_writer_adapter(self):
        def writer_adapter(self, document: Document) -> dict:
            """
            You can create your own adapter that returns a dictionary in your preferred format
            Args:
                document: document to format

            Returns: a dictionary to write to disk

            """
            data = {key: val for key, val in dataclasses.asdict(document).items() if val}
            return {
                "id": data["id"], 
                "text": data["text"],
                "chat": data["metadata"]["chat"]
            }
        
        return writer_adapter
    
    def reader_kwargs(self):
        return {
            "default_metadata": {"dataset_name": self.name},
            "glob_pattern": "*.json"
        }
        
    def writer_kwargs(self):
        return {
            "max_file_size": 400 * 1024 * 1024,
            "output_filename": "${dataset_name}_${rank}.parquet"
        }
        
    def executor_kwargs(self) -> Dict:
        return {
            "tasks": 4,
            "logging_dir": os.path.join("logs", self.name)
        }
        
        
class CommitpackftHandler(DataHandler):
    name = "commitpackft"
    type = "jsonl"
    
    def get_reader_adapter(self):
        files = set()
        def reader_adapter(self, data: dict, path: str, id_in_file: int | str) -> dict:
            """
            The SlimOrca data adapter to adapt input data into the datatrove Document format

            Args:
                data: a dictionary with the "raw" representation of the data
                path: file path or source for this sample
                id_in_file: its id in this particular file or source

            Returns: a dictionary with text, id, media and metadata fields

            """
            
            metadata = data.pop("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"metadata": metadata}
            
            instruction = {
                'zh': f'我将给你一段{data['lang']}代码，请你按照我的要求修改代码',
                'en': f'I will give you a piece of {data['lang']} code, please modify the code according to my requirements',
            }
            
            input = {
                'zh': f'{data['old_contents']}\n\n修改要求：{data['message']}',
                'en': f'{data['old_contents']}\n\nModification requirements: {data['message']}',
            }
            
            lang = "en" if random.random() < 0.9 else "zh"
            
            chat = [
                {"role": "instruction", "content": instruction[lang]},
                {"role": "user", "content": input[lang]},
                {"role": "assistant", "content": data["new_contents"]}
            ]
            
            metadata["chat"] = chat
            
            if path not in files:
                print(f"Processing file {path}")
                files.add(path)
            
            return {
                "text": apply_chat_template(chat),
                "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
                "media": data.pop("media", []),
                "metadata": metadata | data,  # remaining data goes into metadata
            }
        
        return reader_adapter
    
    def get_writer_adapter(self):
        def writer_adapter(self, document: Document) -> dict:
            """
            You can create your own adapter that returns a dictionary in your preferred format
            Args:
                document: document to format

            Returns: a dictionary to write to disk

            """
            data = {key: val for key, val in dataclasses.asdict(document).items() if val}
            return {
                "id": data["id"], 
                "text": data["text"],
                "chat": data["metadata"]["chat"]
            }
        
        return writer_adapter
    
    def reader_kwargs(self):
        return {
            "default_metadata": {"dataset_name": self.name},
            "glob_pattern": "**/*.jsonl"
        }
        
    def writer_kwargs(self):
        return {
            "max_file_size": 400 * 1024 * 1024,
            "output_filename": "${dataset_name}_${rank}.parquet"
        }
        
    def executor_kwargs(self) -> Dict:
        return {
            "tasks": 4,
            "logging_dir": os.path.join("logs", self.name)
        }
        

class OpenhermesHandler(DataHandler):
    name = "openhermes"
    type = "json"
    
    def get_reader_adapter(self):
        
        file_set = set()
        
        def reader_adapter(self, data: dict, path: str, id_in_file: int | str) -> dict:
            """
            The COIG_CQIA data adapter to adapt input data into the datatrove Document format

            Args:
                data: a dictionary with the "raw" representation of the data
                path: file path or source for this sample
                id_in_file: its id in this particular file or source

            Returns: a dictionary with text, id, media and metadata fields

            """
            
            metadata = data.pop("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"metadata": metadata}
            
            chat = []
            inst = data.get("instruction", "")
            if len(inst) > 0:
                chat.append({"role": "instruction", "content": inst})
                
            inp = data.get("input")
            if len(inp) > 0:
                chat.append({"role": "user", "content": inp})
                
            chat.append({"role": "assistant", "content": data["output"]})
            
            metadata["chat"] = chat
            
            if path not in file_set:
                print(f"Processing file {path}")
                file_set.add(path)
            
            return {
                "text": apply_chat_template(chat),
                "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
                "media": data.pop("media", []),
                "metadata": metadata | data,  # remaining data goes into metadata
            }
        
        return reader_adapter
    
    def get_writer_adapter(self):
        def writer_adapter(self, document: Document) -> dict:
            """
            You can create your own adapter that returns a dictionary in your preferred format
            Args:
                document: document to format

            Returns: a dictionary to write to disk

            """
            data = {key: val for key, val in dataclasses.asdict(document).items() if val}
            return {
                "id": data["id"], 
                "text": data["text"],
                "chat": data["metadata"]["chat"]
            }
        
        return writer_adapter
    
    def reader_kwargs(self):
        return {
            "default_metadata": {"dataset_name": self.name},
            "glob_pattern": "*.json"
        }
        
    def writer_kwargs(self):
        return {
            "max_file_size": 400 * 1024 * 1024,
            "output_filename": "${dataset_name}_${rank}.parquet"
        }
        
    def executor_kwargs(self) -> Dict:
        return {
            "tasks": 4,
            "logging_dir": os.path.join("logs", self.name)
        }


class Oasst2Handler(DataHandler):
    name = "oasst2"
    type = "jsonl"
    
    def get_reader_adapter(self):
        
        file_set = set()
        
        def reader_adapter(self, data: dict, path: str, id_in_file: int | str) -> dict:
            """
            The COIG_CQIA data adapter to adapt input data into the datatrove Document format

            Args:
                data: a dictionary with the "raw" representation of the data
                path: file path or source for this sample
                id_in_file: its id in this particular file or source

            Returns: a dictionary with text, id, media and metadata fields

            """
            
            metadata = data.pop("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"metadata": metadata}
            
            
            if path not in file_set:
                print(f"Processing file {path}")
                file_set.add(path)
            
            return {
                "text": "*", # later pipeline will fill this
                "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
                "media": data.pop("media", []),
                "metadata": metadata | data,  # remaining data goes into metadata
            }
        
        return reader_adapter
    
    def get_writer_adapter(self):
        def writer_adapter(self, document: Document) -> dict:
            """
            You can create your own adapter that returns a dictionary in your preferred format
            Args:
                document: document to format

            Returns: a dictionary to write to disk

            """
            data = {key: val for key, val in dataclasses.asdict(document).items() if val}
            return {
                "id": data["id"], 
                "text": data["text"],
                "chat": data["metadata"]["chat"]
            }
        
        return writer_adapter
    
    def reader_kwargs(self):
        return {
            "glob_pattern": "*.jsonl"
        }
        
    def writer_kwargs(self):
        return {
            "max_file_size": 400 * 1024 * 1024,
            "output_filename": "${dataset_name}_${rank}.parquet"
        }
        
    def executor_kwargs(self) -> Dict:
        return {
            "tasks": 4,
            "logging_dir": os.path.join("logs", self.name)
        }
        
    def middle_steps(self) -> List[PipelineStep]:
        
        def get_chat(data: dict)->Iterable[List[dict]]:
            def iter_chat(path: List[dict], node: dict):
                current_text = {"role": node["role"], "content": node["text"]}
                new_path = path + [current_text]
                if "replies" in node and node["replies"]:
                    for reply in node["replies"]:
                        yield from iter_chat(new_path, reply)
                else:
                    if len(new_path) > 1:
                        yield new_path
            return iter_chat([], data)
                
        def extract_chats(data: Iterable[Document], rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
            for doc in data:
                id = doc.id
                prompt = doc.metadata["prompt"]
                for idx, chat in enumerate(get_chat(prompt)):
                    text = apply_chat_template(chat)
                    doc_id = f"{id}/{idx}"
                    yield Document(text=text, id=doc_id, media=[], metadata={"chat": chat, "dataset_name": self.name})
        
        return [extract_chats]

class MathInstructHandler(DataHandler):
    name = "MathInstruct"
    type = "json"
    
    def get_reader_adapter(self):
        
        file_set = set()
        
        def reader_adapter(self, data: dict, path: str, id_in_file: int | str) -> dict:
            """
            The COIG_CQIA data adapter to adapt input data into the datatrove Document format

            Args:
                data: a dictionary with the "raw" representation of the data
                path: file path or source for this sample
                id_in_file: its id in this particular file or source

            Returns: a dictionary with text, id, media and metadata fields

            """
            
            metadata = data.pop("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"metadata": metadata}
            
            chat = []
            inst = data.get("instruction")
            
            chat.append({"role": "instruction", "content": inst})
                
            chat.append({"role": "assistant", "content": data["output"]})
            
            metadata["chat"] = chat
            
            if path not in file_set:
                print(f"Processing file {path}")
                file_set.add(path)
            
            return {
                "text": apply_chat_template(chat),
                "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
                "media": data.pop("media", []),
                "metadata": metadata | data,  # remaining data goes into metadata
            }
        
        return reader_adapter
    
    def get_writer_adapter(self):
        def writer_adapter(self, document: Document) -> dict:
            """
            You can create your own adapter that returns a dictionary in your preferred format
            Args:
                document: document to format

            Returns: a dictionary to write to disk

            """
            data = {key: val for key, val in dataclasses.asdict(document).items() if val}
            return {
                "id": data["id"], 
                "text": data["text"],
                "chat": data["metadata"]["chat"]
            }
        
        return writer_adapter
    
    def reader_kwargs(self):
        return {
            "default_metadata": {"dataset_name": self.name},
            "glob_pattern": "*.json"
        }
        
    def writer_kwargs(self):
        return {
            "max_file_size": 400 * 1024 * 1024,
            "output_filename": "${dataset_name}_${rank}.parquet"
        }
        
    def executor_kwargs(self) -> Dict:
        return {
            "tasks": 4,
            "logging_dir": os.path.join("logs", self.name)
        }

    
class HandlerFactory:
    handler_map = {}
    
    @classmethod
    def get_handler(cls, name: str):
        return cls.handler_map[name]
    
    @classmethod
    def register_handler(cls, handler: DataHandler):
        cls.handler_map[handler.name] = handler
    

HandlerFactory.register_handler(MagicoderHandler)
HandlerFactory.register_handler(COIG_CQIAHandler)
HandlerFactory.register_handler(SlimOrcaHandler)
HandlerFactory.register_handler(UltraChatHandler)
HandlerFactory.register_handler(EvolInstructCodeHandler)
HandlerFactory.register_handler(CommitpackftHandler)
HandlerFactory.register_handler(OpenhermesHandler)
HandlerFactory.register_handler(Oasst2Handler)
HandlerFactory.register_handler(MathInstructHandler)
