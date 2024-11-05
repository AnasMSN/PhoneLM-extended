import argparse

import sys
sys.path.append("./src")

from datatrove.executor import LocalPipelineExecutor

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="Path to the data")
parser.add_argument("output_path", type=str, help="Path to the output")
parser.add_argument("--dataset_name", type=str, help="Name of the dataset", required=True)

args = parser.parse_args()
        
from build_datasets.data_handler import HandlerFactory

if __name__ == "__main__":
    handler_cls = HandlerFactory.get_handler(args.dataset_name)
    
    handler = handler_cls(args.data_path, args.output_path)
    
    pipeline = handler.pipelines()
    kwargs = handler.executor_kwargs()
    
    exec = LocalPipelineExecutor(pipeline, **kwargs)
    
    stat = exec.run()
    print(stat)

