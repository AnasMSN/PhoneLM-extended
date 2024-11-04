## how to run
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'mllmTeam/PhoneLM-1.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

model = model.to('cuda')

question = "Hello, who are you?"
chat = [
    {"role": "user", "content": question},
]

prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

inp = tokenizer(prompt, return_tensors="pt")
inp = {k: v.to('cuda') for k, v in inp.items()}
out = model.generate(**inp, 
                     max_length=256,
                     do_sample=False
                     )
text = tokenizer.decode(out[0], skip_special_tokens=True)

```

## how to sft
Initial dataset structure
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

Launch train command
```shell
deepspeed train_instruct.py --config config_phonelm_1.5b_instruct.yaml
```
If it is the first time loading train_datasets_instruct, two directories train_dataset_test and val_dataset_test will be generated in the train_datasets_instruct directory. Subsequently, data will be read directly from these two directories.
