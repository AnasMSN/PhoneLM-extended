# !/usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["WANDB_PROJECT"] = "phonelm"
os.environ["TOKENIZERS_PARALLELISM"] = "false" # avoid deadlock

import wandb
import argparse
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)
from trl import SFTTrainer
from datasets import load_dataset, concatenate_datasets, load_from_disk


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
PROF = config.get("profile", False)
RESUME = config.get("resume", False)
train_name = config.get("name", "phonelm")
use_bf16 = config.get("training.use_bf16", True)
context_size = config.get("training.context_size", 2048)
output_dir = config.get("training.output_dir", f"./checkpoints/{train_name}")
num_train_epochs = config.get("training.num_train_epochs", 10)
learning_rate = float(config.get("training.learning_rate", 1e-4))
adam_beta1 = config.get("training.adam_beta1", 0.9)
adam_beta2 = config.get("training.adam_beta2", 0.95)
adam_epsilon = float(config.get("training.adam_epsilon", 1e-8))
weight_decay = config.get("training.weight_decay", 0.1)
deepspeed_config =  config.get("training.deepspeed_config", "./ds_config_decaylr.json")
per_device_train_batch_size = int(config.get("training.per_device_train_batch_size", 32))
per_device_eval_batch_size = int(config.get("training.per_device_eval_batch_size", 48))
gradient_checkpointing = config.get("training.gradient_checkpointing", True)
gradient_accumulation_steps = int(config.get("training.gradient_accumulation_steps", 1))
set_logging_steps = int(config.get("training.set_logging_steps", 20))
set_eval_steps = int(config.get("training.set_eval_steps", 1000))
set_save_steps = int(config.get("training.set_save_steps", 2000))
set_compute_metrics = config.get("training.set_compute_metrics", False)
bad_epochs_limit = int(config.get("training.bad_epochs_limit", 5))
warmup_ratio = float(config.get("training.warmup_ratio", 0.10))
warmup_steps = config.get("training.warmup_steps", None)
warmup_steps = int(warmup_steps) if warmup_steps is not None else None
max_steps = int(config.get("training.max_steps", -1))
# ==========================================
print(f"config: {config.config}")



def load_and_split_data(file_path, split_ratio=0.005):
    """
    Load the specified file and split it into training and validation sets.
    
    Parameters:
    - file_path: Path to the file.
    - split_ratio: Proportion of the validation set in the total dataset.
    
    Returns:
    - Split datasets, including training and validation sets.
    """
    print("Split ratio:", split_ratio)
    # File format
    data_type = config.get("datasets.data_type", "parquet")
    dataset = load_dataset(data_type, data_files=file_path)
    
    # Split dataset
    split_dataset = dataset['train'].train_test_split(test_size=split_ratio)
    
    # Ensure the split dataset contains 'train' and 'test' keys
    if 'train' in split_dataset and 'test' in split_dataset:
        return split_dataset['train'], split_dataset['test']
    else:
        raise ValueError("Dataset split failed, training and validation sets not generated.")


def build_sft_dataset(data_path, split_ratio=0.005):
    """
    Build the training and validation datasets for SFT.

    Parameters:
    - data_path: Path to the directory containing the sft data files.
    - split_ratio: Proportion of the validation set in the total dataset.

    Returns:
    - Training and validation datasets.
    """
    train_datasets = []
    val_datasets = []

    train_dataset_path = os.path.join(data_path, 'train_dataset_test')
    val_dataset_path = os.path.join(data_path, 'val_dataset_test')
    print(train_dataset_path)
    print(val_dataset_path)

    if os.path.isdir(train_dataset_path) and os.path.isdir(val_dataset_path):
        # Load previously saved datasets
        train_datasets = load_from_disk(train_dataset_path)
        print("Train dataset loaded successfully")
        val_datasets = load_from_disk(val_dataset_path)
        print("Validation dataset loaded successfully")
        return train_datasets, val_datasets

    # Traverse all files in the processed_chat directory
    for root, dirs, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"Processing file: {file_path}")
        
            # Load and split dataset
            train_dataset, val_dataset = load_and_split_data(file_path)
        
            # Add the current file's training and validation sets to the list
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)

    # Combine all files' training and validation sets
    combined_train_dataset = concatenate_datasets(train_datasets).shuffle(seed=42)
    combined_val_dataset = concatenate_datasets(val_datasets).shuffle(seed=42)

    print(f"Total training set size: {len(combined_train_dataset)}")
    print(f"Total validation set size: {len(combined_val_dataset)}")

    return combined_train_dataset, combined_val_dataset


def train(tokenizer, model, train_dataset, val_dataset):
    # Set training arguments
    args = dict(
        # We do not dispatch the dataloader, so each process will load the full dataset and pick by process index.
        accelerator_config={
            "dispatch_batches": False
        },
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        metric_for_best_model="eval_loss",
        max_steps=max_steps,
        bf16=use_bf16,
        fp16=not use_bf16,
        learning_rate=learning_rate,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        weight_decay=weight_decay,
        num_train_epochs=num_train_epochs,
        warmup_ratio=warmup_ratio,
        # logging & evaluation strategies
        logging_dir="logs",
        logging_strategy="steps",
        logging_steps=set_logging_steps,  # Log every set_logging_steps steps
        evaluation_strategy="steps",
        eval_steps=set_eval_steps,  # Evaluate every set_eval_steps steps
        save_steps=set_save_steps,
        save_total_limit=6,
        load_best_model_at_end=True,
        deepspeed=deepspeed_config,  # Path to deepspeed config file
        report_to="all" if FLG_WANDB else "none",
    )
    if warmup_steps is not None:
        args["warmup_steps"] = warmup_steps
    training_args = TrainingArguments(
        **args
    )
    print("Training arguments:", training_args)
    train_id = "local"
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
            profile_memory=True,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        ) as prof:
            # Model training
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                # dataset_text_field="text",
                max_seq_length=2048,
                packing=config.get("datasets.packing", True),
                compute_metrics=compute_metrics if set_compute_metrics else None,
                tokenizer=tokenizer,
                callbacks=(
                    [EvaluateCallback(bad_epochs_limit, arg.local_rank, FLG_WANDB), TraceCallback(prof)]
                    if PROF and is_main_process_using_local_rank(arg.local_rank)
                    else [EvaluateCallback(bad_epochs_limit, arg.local_rank, FLG_WANDB)]
                ),
            )
            trainer.train(resume_from_checkpoint=RESUME)
    else:
        try:
            train_id = wandb.run.id if (FLG_WANDB and is_main_process_using_local_rank(arg.local_rank)) else "local"
            print(f"Training ID: {train_id}")
        except:
            train_id = "latest"
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            # dataset_text_field="text",
            packing=config.get("datasets.packing", True),
            max_seq_length=2048,
            compute_metrics=compute_metrics if set_compute_metrics else None,
            tokenizer=tokenizer,
            callbacks=[EvaluateCallback(bad_epochs_limit, arg.local_rank, FLG_WANDB)],
        )
        print("Start training")
        trainer.train(resume_from_checkpoint=RESUME)
        print("Training finished")
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
   

    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("load base model")
    checkpoint_path = config.get("training.base_dir", f"./checkpoints/phonelm-1.5B_stage2/best_ckpt")
    print("================BASE:", checkpoint_path, "================\n")
    model = PhoneLMForCausalLM.from_pretrained(checkpoint_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    if gradient_checkpointing:
        model.config.use_cache = False
    
    if config.get("training.use_lora", False):
        from peft import LoraConfig, get_peft_model
        from peft import PeftModel
        
        peft_config = LoraConfig(
            r=config.get("lora.r", 1),
            lora_alpha=config.get("lora.alpha", 16),
            lora_dropout=config.get("lora.dropout", 0.1),
            bias="none",
            task_type="CAUSAL_LM",
            # q_proj,k_proj,gate_proj,down_proj,up_proj
            target_modules=["q_proj", "k_proj", "gate_proj", "down_proj", "up_proj"],
        )
        if gradient_checkpointing:
            # https://discuss.huggingface.co/t/peft-lora-gpt-neox-backward-pass-failing/35641/6
            # need this to fix when using gradient checkpointing and lora
            model.enable_input_require_grads()
        
        model = get_peft_model(model, peft_config)
    print(f"model loaded: {model}")
    
    
    print("load data")    
    data_path = config.get("datasets.path","./train_datasets_instruct")
    train_dataset, val_dataset = build_sft_dataset(data_path)
    
    print(f"train dataset: {train_dataset}")

    print("train")
    train(tokenizer, model, train_dataset, val_dataset)
    
    if FLG_WANDB:
        wandb.finish()
