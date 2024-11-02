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