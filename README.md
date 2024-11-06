This repository contains the code and documents in pre-training, fine-tuning, and evaluating PhoneLM [ref], a highly capable and efficient small language model family.
The end-to-end demo of PhoneLM running on smartphone is available at [mllm](https://github.com/UbiquitousLearning/mllm).

## Model Downloads

  | HuggingFace |
  |-------------|
  |[PhoneLM-1.5B-Base](https://huggingface.co/mllmTeam/PhoneLM-1.5B-Base)|
  |[PhoneLM-1.5B-Instruct](https://huggingface.co/mllmTeam/PhoneLM-1.5B-Instruct)|
  |[PhoneLM-1.5B-Call](https://huggingface.co/mllmTeam/PhoneLM-1.5B-Call)|
  |[PhoneLM-0.5B-Base](https://huggingface.co/mllmTeam/PhoneLM-0.5B-Base)|
  |[PhoneLM-0.5B-Instruct](https://huggingface.co/mllmTeam/PhoneLM-0.5B-Instruct)|

## Evaluation Results

### Comprehensive Evaluation
| Model | HellaSwag | WinoGrande | PIQA | SciQ | BoolQ | ARC Easy | ARC Challenge | Average |
|-----------|-----------|------------|------|------|-------|----------|---------------|---------|
| Pythia-1.4B | 52.0 | 57.2 | 71.1 | 79.2 | 63.2 | 53.9 | 28.3 | 57.84 |
| OPT-1.3B | 53.7 | 59.0 | 71.0 | 78.1 | 57.2 | 51.3 | 28.0 | 56.90 |
| BLOOM-1.1B | 43.0 | 54.9 | 67.2 | 74.6 | 59.1 | 45.4 | 25.6 | 52.83 |
| TinyLlama-1.1B | 59.1 | 58.9 | 73.0 | 82.3 | 58.6 | 55.7 | 31.0 | 59.80 |
| MobileLLaMA-1.4B | 56.1 | 59.4 | 73.0 | 81.9 | 56.7 | 55.8 | 30.3 | 59.03 |
| MobiLlama-1B | 62.2 | 59.3 | 74.8 | 82.8 | 60.3 | 56.4 | 31.7 | 61.07 |
| OpenELM-1.1B | 64.8 | 61.7 | 75.6 | 83.6 | 63.6 | 55.4 | 32.3 | 62.43 |
| DCLM-1.4B | 53.6 | 66.3 | 77.0 | 94.0 | 71.4 | 74.8 | 41.2 | 68.33 |
| SmolLM-1.7B | 49.6 | 60.9 | 75.8 | 93.2 | 66.0 | 76.4 | 43.5 | 66.49 |
| Qwen 1.5-1.8B | 60.9 | 60.5 | 74.2 | 89.4 | 66.5 | 59.1 | 34.7 | 63.61 |
| Galactica-1.3B | 41.0 | 54.4 | 63.8 | 87.7 | 62.0 | 58.6 | 30.5 | 56.86 |
| StableLM 2-1.6B | 68.8 | 64.1 | 75.1 | 76.9 | 80.0 | 60.3 | 39.2 | 66.34 |
| Cerebras-GPT-1.3B | 38.4 | 51.9 | 66.8 | 73.0 | 59.3 | 45.8 | 25.3 | 51.50 |
| MiniCPM-1B | 67.5 | 63.7 | 75.1 | 91.0 | 70.5 | 62.9 | 38.1 | 66.97 |
| MiniCPM-2B | 67.2 | 63.9 | 76.1 | 92.5 | 74.6 | 69.0 | 42.7 | 69.43 |
| Gemma-2B | 71.4 | 65.2 | 78.4 | 91.4 | 69.9 | 72.3 | 42.0 | 70.09 |
| Gemma 2-2B | 55.0 | 68.7 | 78.7 | 96.0 | 73.6 | 80.3 | 46.9 | 71.31 |
| **PhoneLM-1.5B** | **66.9** | **63.0** | **77.3** | **88.8** | **65.5** | **69.7** | **39.9** | **67.31** |

### Android Function Call


## Runnning PhoneLM

### Huggingface
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'mllmTeam/PhoneLM-1.5B-Instruct'
question = "Hello, who are you?"
prompt = [{"role": "user", "content": question}]


model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda', trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)


inp = tokenizer(input_text, return_tensors="pt")
inp = {k: v.to('cuda') for k, v in inp.items()}
out = model.generate(**inp, 
                     max_length=256,
                     do_sample=True,
                     temperature=0.7,
                     top_p=0.7
                     )
text = tokenizer.decode(out[0], skip_special_tokens=True)
print(text)
```
### mllm

We have provided the [mllm formats]((https://huggingface.co/mllm/phonelm-mllm)) of PhoneLM, which can be used in [mllm](https://github.com/UbiquitousLearning/mllm).

Install mllm
```shell
git clone https://github.com/UbiquitousLearning/mllm.cpp
cd mllm/scripts/
build.sh
```
Inference
```shell
cd ../bin
./demo_phonelm
```

## Training PhoneLM

### Install Python Environment

```bash
pip install -r requirement.txt
```

### Stable Training Stage

We use the following dataset in stable training stage.
| type     | dataset                         | token  |
|----------|---------------------------------|--------|
| web      | [DCLM-baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet) | 1.35T  |
| code     | [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) | 112.75B|
| math     | [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | 13.25B |
| academic | [Dolma-algebraic](https://huggingface.co/datasets/allenai/dolma) | 12.75B |
| academic | [Dolma-arxiv](https://huggingface.co/datasets/allenai/dolma) | 29B   |
| **total**|                                 | **1.5T** |

**Download The Original Data**

You can download the dataset from the links provided in the table above using any method.As an example, we use `huggingface-cli` to download DCLM-baseline. Here is an example command:
```bash
huggingface-cli download --repo-type dataset --local-dir ./dclm-baseline --local-dir-use-symlinks False --resume-download mlfoundations/dclm-baseline-1.0-parquet
```
You can decide how to download the dataset through the links in the table above.

**Preprocess the dataset**

Before pretraining, it is necessary to perform tokenization on the dataset in advance. Before tokenization, you should first know the format of the dataset and the field in the dataset used to pretrain. Take [`dclm-baseline`](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet) as an example, the data files format is parquet. And in its Dataset Card, it can be seen that the `text` field of each data entry is used for pretraining. After knowing the format type, we can use the following command to tokenize the data in advance
```bash
python path/to/dataset path/to/output_dir\
  --prefix prefix_of_output_file\ 
  --handler file_format\
  --field field_used_to_pretrain\
  --num_workers  workers_to_process\
  --tokenizer_path path/to/tokenizer\
  --max_size max_tokens_of_each_output_file
```
For example, to tokenize dclm-baseline, use following command in `PhoneLM`
```bash
python pretokenize.py path/to/dclm-baseline ./train_datasets/dclm-baseline 
  --prefix dclm-baseline 
  --handler parquet 
  --field text
  --tokenizer_path tokenizer
```
The output will look like:
```
train_datasets/
└── dclm-baseline
    ├── dclm-baseline-000-00000.data
    ├── dclm-baseline-001-00000.data
    ├── dclm-baseline-002-00000.data
    ├── dclm-baseline-003-00000.data
    ...
```

**Train**

After performing the same operation on all datasets, the tokenized datasets are stored in `train_datasets`. Subsequently, you can start pretraining with the following command:
```bash
deepspeed train.py --config config_phonelm_1.5b.yaml
```

### Decay Stage

In the decay stage, the data contains some dataset from stable training stage, including DCLM-baseline, StarCoderData, and Dolma. And it also
contains some high-quality fine-tuning data, which is used in fine-tuning stage. Following table shows the data
| Type      | Dataset                     | Token |
|-----------|-----------------------------|-------|
| web       | [DCLM-baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet) | 10B   |
| code       | [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) | 1.575B |
| code      | [The Stack Smol](https://huggingface.co/datasets/bigcode/the-stack-smol)              | 0.95B |
| acadamic  | [Dolma-arxiv](https://huggingface.co/datasets/allenai/dolma) | 2.325B |
| acadamic  | [Dolma-pes2o](https://huggingface.co/datasets/allenai/dolma) | 2.35B |
| math instruct | [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) | 65.25M |
| chat instruct | [UltraChat](https://huggingface.co/datasets/stingning/ultrachat) | 1.775B |
| chat instruct | [OpenAssistant 2](https://huggingface.co/datasets/OpenAssistant/oasst2) | 42.25M |
| chat instruct | [OpenHermes](https://huggingface.co/datasets/teknium/openhermes) | 77.25M |
| code instruct | [Magicoder Evol Instruct](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | 30.25M |
| code instruct | [CommitPackFT](https://huggingface.co/datasets/bigcode/commitpackft) | 0.35B |
| code instruct | [Magicoder OSS Instruct](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | 43.5M |
| function calling | [SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) | 209.75M |
| function calling | [APIGen](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 48.25M |
| function calling | [Glaive Function Calling](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | 57.5M |
| **total**      |                            | **20B**   |

Unfortunately, the datasets in the table above, excluding those used for pretraining, each have their own format. To standardize the datasets in this phase, we have processed all SFT data into a chat format and formatted them as text using a unified template.

We will show you an example. First download the dataset as shown above.Then use the following command to process:
```bash
python prepare_chat.py path/to/MathInstruct chat/MathInstruct --dataset_name MathInstruct # process MathInstruct

python prepare_chat.py ../datasets/Magicoder-OSS-Instruct-75K/ chat/Magicoder --dataset_name Magicoder # process Magicoder
```
After processing the dataset, the `chat` directory will looks like
```bash
chat/
├── Magicoder
│   └── 000_Magicoder_00000.parquet
└── MathInstruct
    └── 000_MathInstruct_00000.parquet
```
Format of processed data is as following:
```json
{
  "text": "pretrain data",
  "chat": [
    {"role": "...", "content": "..."},
    ...
  ]
}
```

Then you can tokenize the `text` field to get the Decay Stage pretrain data using `pretokenize.py`.

**Train**

Subsequently, you can start decay stage training with the following command:
```bash
deepspeed train.py --config config_phonelm_1.5b_stage2.yaml
```

### Instruct Following Stage
In this stage you need to initial dataset structure as followed:
```
- train_datasets_instruct
|
- - - dataset_00
    |
    - - -data_001.parquet
    |
    - - -data_002.parquet
    |
    - - - ...
```
**Train**

Launch train command
```shell
deepspeed train_instruct.py --config config_phonelm_1.5b_instruct.yaml
```
If it is the first time loading train_datasets_instruct, two directories train_dataset_test and val_dataset_test will be generated in the train_datasets_instruct directory. Subsequently, data will be read directly from these two directories.
