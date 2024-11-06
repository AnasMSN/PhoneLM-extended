# !/usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["WANDB_PROJECT"] = "phonelm"

import json
import random
import wandb
import argparse
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

from config import Config
from modeling_phonelm import PhoneLMForCausalLM
from configuration_phonelm import PhoneLMConfig
from utils import *

# ==========================================
args = argparse.ArgumentParser(description="Train a model.")
args.add_argument("--local_rank", type=int, default=-1)
args.add_argument("--config", type=str, default="config.yaml")
arg = args.parse_args()
config = Config(arg.config)
FLG_WANDB = config.get("wandb", False)
PROF = bool(config.get("profile", False))
RESUME = bool(config.get("resume", False))
train_name = config.get("name", "phonelm")
use_bf16 = config.get("training.use_bf16", True)
context_size = config.get("training.context_size", 2048)
output_dir = config.get("training.output_dir", f"./checkpoints/{train_name}")
resume_ckpt_dir = config.get("training.resume_ckpt_dir", "latest")
num_train_epochs = config.get("training.num_train_epochs", 10)
learning_rate = float(config.get("training.learning_rate", 1e-4))
adam_beta1 = config.get("training.adam_beta1", 0.9)
adam_beta2 = config.get("training.adam_beta2", 0.95)
adam_epsilon = float(config.get("training.adam_epsilon", 1e-8))
weight_decay = config.get("training.weight_decay", 0.1)
deepspeed_config =  config.get("training.deepspeed_config", "./ds_config_coslr.json")
per_device_train_batch_size = int(config.get("training.per_device_train_batch_size", 32))
per_device_eval_batch_size = int(config.get("training.per_device_eval_batch_size", 48))
gradient_accumulation_steps = int(config.get("training.gradient_accumulation_steps", 1))
set_logging_steps = int(config.get("training.set_logging_steps", 20))
set_eval_steps = int(config.get("training.set_eval_steps", 2000))
set_save_steps = int(config.get("training.set_save_steps", 2000))
bad_epochs_limit = int(config.get("training.bad_epochs_limit", 5))
warmup_steps = int(config.get("training.warmup_steps", 1000))
max_steps = int(config.get("training.max_steps", -1))
no_eval = bool(config.get("training.no_eval", False))
# ==========================================
print(f"config: {config.config}")

def match_files(file_dir: str, pattarn: str):
    # return all files that match the pattern in the dir path
    file_dir = Path(file_dir)
    return [str(filename) for filename in file_dir.rglob(pattarn)]

def build_dataset(config_, skip_step=0):    
    path = config_.get("datasets.path","./train_datasets")
    print("Loading data at",os.path.abspath(path))
    context_size = config_.get("training.context_size", 2048)
    seed = config_.get("training.seed", 1234)
    tokenized = config_.get("datasets.tokenized", True)
    
    if tokenized:
        all_files = match_files(path, "*.data")
    else:
        all_files = match_files(path, "*.parquet")
    
    config_path = config_.get("datasets.config", os.path.join(path, "data_config.json"))
    import json
    if not tokenized:
        from build_datasets.build import build_with_untokenized_data
        with open(config_path, 'r') as file:
            config = json.load(file)
        
        tokenizer_path = config_.get("model.tokenizer_path", "./tokenizer")
        print(f"--------{tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        train_dataset, val_dataset = build_with_untokenized_data(all_files, config, tokenizer, 
                                                                 context_size=context_size, seed=seed,
                                                                 verbose=True)
        
        return train_dataset, val_dataset
    
    random.seed(seed)
    random.shuffle(all_files)

    train_exists = os.path.isdir(os.path.join(path, 'train'))
    validation_exists = os.path.isdir(os.path.join(path, 'validation'))
    
    local_rank: int = get_local_rank(arg.local_rank)
    cross_rank = get_cross_rank()
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print(
        f"Local Rank: {local_rank}, Cross Rank: {cross_rank}, World Size: {world_size}"
    )
    
    if train_exists and validation_exists:
        train_files = match_files(os.path.join(path, 'train'), "*.data")
        validation_files = match_files(os.path.join(path, 'validation'), "*.data")
        random.shuffle(train_files)
        random.shuffle(validation_files)
    elif os.path.exists(config_path):
        from build_datasets.build import build_with_config
        with open(config_path, 'r') as file:
            config = json.load(file)
        train_dataset, val_dataset = build_with_config(all_files, config, context_size=context_size, 
                                           rank=cross_rank, world_size=world_size, seed=seed, verbose=True, skip_step=skip_step)
        return train_dataset, val_dataset
    else:
        val_rate = config_.get("datasets.val_rate", 0.005)
        print(len(all_files))
        split_point = int(len(all_files) * (1 - val_rate))
        if split_point == len(all_files):
            split_point -= 1
        train_files = all_files[:split_point]
        validation_files = all_files[split_point:]

    from build_datasets.build import build_with_prefix
    
    # note here we should always set the dispatch to True
    # though in accelerate config we set dispatch_batches to False ...
    # if we set dispatch_batches to True in accelerate config, accelerate will
    # kindly wrap our dataset with IterableDatasetShard, and it will get batch for each process so 
    # we don't need to split the dataset for each process manually whatever the dispatch_batches is True or False ... (# - . -) 
    train_dataset = build_with_prefix(train_files, context_size=context_size, rank=cross_rank, world_size=world_size, seed=seed, skip_step=skip_step, dispatch=True)
    val_dataset = build_with_prefix(validation_files, context_size=context_size, rank=cross_rank, world_size=world_size, seed=seed, dispatch=True)

    return train_dataset, val_dataset

def train(tokenizer, model, train_dataset, val_dataset):
    # 设置训练参数
    training_args = TrainingArguments(
        # We do not dispatch the dataloader, so each process will load the full dataset and pick by process index.
        accelerator_config={
            "dispatch_batches": False,
        },
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        metric_for_best_model="eval_loss",
        max_steps=max_steps,
        evaluation_strategy="steps" if not no_eval else "no",
        # eval_accumulation_steps=2,
        # eval_accumulation_steps=2,
        # predict_with_generate=True,
        bf16=use_bf16,
        fp16=not use_bf16,
        learning_rate=learning_rate,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        weight_decay=weight_decay,
        num_train_epochs=num_train_epochs,
        warmup_steps=warmup_steps,
        # logging & evaluation strategies
        logging_dir="logs",
        logging_strategy="steps",
        logging_steps=set_logging_steps,
        eval_steps=set_eval_steps if not no_eval else None,
        save_steps=set_save_steps,
        save_total_limit=6,
        load_best_model_at_end=True if not no_eval else False,
        deepspeed=deepspeed_config,
        report_to="all" if FLG_WANDB else "none",
        
        ignore_data_skip=True,
    )
    train_id = "local"
    if RESUME:
        resume_base_dir = os.path.join(output_dir, resume_ckpt_dir)
        print("=============resume_base_dir: ", resume_base_dir)
    if PROF:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                skip_first=3, wait=1, warmup=1, active=2, repeat=1
            ),
            on_trace_ready=lambda p: trace_handler(p, arg.local_rank),
            with_stack=True,
            record_shapes=True,
            profile_memory=True,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        ) as prof:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics= None,
                tokenizer=tokenizer,  # Remove the extra comma here
                callbacks=(
                    [EvaluateCallback(bad_epochs_limit, arg.local_rank, FLG_WANDB), TraceCallback(prof)]
                    if PROF and is_main_process_using_local_rank(arg.local_rank)
                    else [EvaluateCallback(bad_epochs_limit, arg.local_rank, FLG_WANDB)]
                ),
            )
            if RESUME:
                trainer.train(resume_base_dir)
            else:
                trainer.train(resume_from_checkpoint=RESUME)
    else:
        try:
            train_id = wandb.run.id if (FLG_WANDB and is_main_process_using_local_rank(arg.local_rank)) else "local"
            print(f"Training ID: {train_id}")
        except:
            train_id = "latest"
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics= None,
            tokenizer=tokenizer,  # Remove the extra comma here
            callbacks=[EvaluateCallback(bad_epochs_limit, arg.local_rank, FLG_WANDB)],
        )
        if RESUME:
            trainer.train(resume_base_dir)
        else:
            trainer.train(resume_from_checkpoint=False)

    print(trainer.evaluate(val_dataset))
    best_path = os.path.join(output_dir, "best_ckpt")
    trainer.save_model(best_path)
    save_phoinelm_hf(output_dir, trainer.model.dtype)
    if FLG_WANDB and is_main_process_using_local_rank(arg.local_rank):
        wandb.config.update({"best_path": best_path})

if __name__ == "__main__":
    if FLG_WANDB:
        if is_main_process_using_local_rank(arg.local_rank):
            wandb.init(
                # set the wandb project where this run will be logged
                project="phonelm",
                name=train_name,
                config={
                    "output_dir": output_dir,
                    "num_train_epochs": num_train_epochs,
                    "learning_rate": learning_rate,
                    "deepspeed_config": deepspeed_config,
                    "per_device_train_batch_size": per_device_train_batch_size,
                    "per_device_eval_batch_size": per_device_eval_batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "set_logging_steps": set_logging_steps,
                    "set_eval_steps": set_eval_steps,
                    "set_save_steps": set_save_steps,
                    "config_file": config.config,
                },
                # track hyperparameters and run metadata
            )
            wandb.alert(title="PhoneLM", text="Start training")
    tokenizer_path = config.get("model.tokenizer_path", "./tokenizer")
    if not os.path.exists(tokenizer_path):
        print(f"!!! Not Found Tokenizer {tokenizer_path}, use default tokenizer.")
        tokenizer_path = "./tokenizer"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            legacy=False,
            max_length=context_size,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Tokenizer Error: {e}")

    stage1_ckpt = config.get("training.stage1_ckpt", None)
    if stage1_ckpt is not None:
        print(f"[Training Stage2]: Loading model from {stage1_ckpt}")
        model = PhoneLMForCausalLM.from_pretrained(stage1_ckpt, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    else:
        phonelm_config = PhoneLMConfig()
        phonelm_config.vocab_size=config.get("model.vocab_size", 49152)
        phonelm_config.hidden_size = config.get("model.hidden_size", 768)
        phonelm_config.intermediate_size = config.get("model.intermediate_size", 2046)
        phonelm_config.num_hidden_layers = config.get("model.num_hidden_layers", 12)
        phonelm_config.hidden_act = config.get("model.hidden_act", "relu")
        phonelm_config.num_attention_heads = config.get("model.num_attention_heads", 32)
        phonelm_config.num_key_value_heads = config.get("model.num_key_value_heads", 4)
        phonelm_config.tie_word_embeddings = config.get("model.tie_word_embeddings", True)
        phonelm_config._attn_implementation = "flash_attention_2"
        model = PhoneLMForCausalLM(phonelm_config)


    print_model_size(model)

    skip_step = 0
    resume_base_dir = os.path.join(output_dir, resume_ckpt_dir)
    if RESUME:
        with open(os.path.join(resume_base_dir, "trainer_state.json"), "r") as f:
            trainer_state = json.load(f)
        global_step = trainer_state["global_step"]
        skip_step = per_device_eval_batch_size * global_step
    
    train_dataset, val_dataset = build_dataset(config, skip_step)
    train(tokenizer, model, train_dataset, val_dataset)
    
    if FLG_WANDB:
        wandb.finish()