
from modeling_phonelm import PhoneLMForCausalLM
from configuration_phonelm import PhoneLMConfig

import os
import numpy as np
import wandb
import time
import torch
from transformers import (
    AutoTokenizer,
    TrainerCallback,
)

def print_model_size(model):
    num_parameters = sum(p.numel() for p in model.parameters())
    num_parameters_in_billions = num_parameters / 1e9
    print(f"The model has {num_parameters_in_billions:.7f} billion parameters.")

def get_cross_rank():
    # get the cross rank from env
    cross_rank = -1
    try:
        cross_rank = int(os.getenv("RANK", -1))
    except:
        pass
    return cross_rank

def get_local_rank(local_rank_):
    # get the local rank from  args
    local_rank = -1
    try:
        local_rank = local_rank_ #arg.local_rank
    except:
        pass
    return local_rank

def is_main_process_using_local_rank(local_rank_) -> bool:
    """
    Determines if it's the main process using the local rank.

    based on print statements:
        local_rank=0
        local_rank=1

    other ref:
        # - set up processes a la l2l
        local_rank: int = get_local_rank()
        print(f'{local_rank=}')
        init_process_group_l2l(args, local_rank=local_rank, world_size=args.world_size, init_method=args.init_method)
        rank: int = torch.distributed.get_rank() if is_running_parallel(local_rank) else -1
        args.rank = rank  # have each process save the rank
        set_devices_and_seed_ala_l2l(args)  # args.device = rank or .device
        print(f'setup process done for rank={args.rank}')
    """

    local_rank: int = get_local_rank(local_rank_)
    cross_rank = get_cross_rank()
    return (local_rank == -1 or local_rank == 0) and (
        cross_rank == -1 or cross_rank == 0
    )  # -1 means serial, 0 likely means parallel

class EvaluateCallback(TrainerCallback):
    def __init__(self, bad_epochs_limit_, local_rank_, FLG_WANDB_):
        self.epoch = 0
        self.bad_epochs = 0
        self.last_eval_loss = 1000.0
        self.bad_epochs_limit = bad_epochs_limit_
        self.local_rank = local_rank_
        self.FLG_WANDB = FLG_WANDB_

    def on_evaluate(
        self,
        args,
        state,
        control,
        model,
        tokenizer,
        metrics,
        **kwargs,
    ):
        self.epoch += 1
        eval_loss = metrics["eval_loss"]
        if eval_loss < self.last_eval_loss:
            self.last_eval_loss = eval_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        if self.bad_epochs >= self.bad_epochs_limit:
            local_rank = get_local_rank(self.local_rank)
            if self.FLG_WANDB and (local_rank == 0 or local_rank == -1):
                try:
                    wandb.alert(title="PhoneLM", text="Early stopping")
                except:
                    pass

class TraceCallback(TrainerCallback):

    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()


def trace_handler(p, local_rank_):

    if is_main_process_using_local_rank(local_rank_):
        output = p.key_averages(group_by_input_shape=True).table(
        sort_by="self_cuda_memory_usage", row_limit=100
    )
        print(output)
        p.export_chrome_trace("./trace_" + str(p.step_num) + ".json")
        time.sleep(1)
        exit()




def save_phoinelm_hf(model_directory, dtype=torch.float32):

    save_directory = model_directory.replace("checkpoints/", "checkpoints/mllmTeam/")

    tokenizer = AutoTokenizer.from_pretrained(model_directory+"best_ckpt")
    model = PhoneLMForCausalLM.from_pretrained(model_directory+"best_ckpt")

    model = model.to(dtype)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    config = model.config
    phonelmconfig = PhoneLMConfig()

    new_config_dict = {key: config.to_dict()[key] for key in phonelmconfig.to_dict().keys() if key in config.to_dict()}

    new_config = PhoneLMConfig(**new_config_dict)

    new_config.auto_map = {
        "AutoConfig": "configuration_phonelm.PhoneLMConfig",
        "AutoModelForCausalLM": "modeling_phonelm.PhoneLMForCausalLM",
    }
    new_config._name_or_path = "./"
    new_config.save_pretrained(save_directory)

    with open(f"{save_directory}/modeling_phonelm.py", "w") as f:
        f.write(open("modeling_phonelm.py").read())
    with open(f"{save_directory}/configuration_phonelm.py", "w") as f:
        f.write(open("configuration_phonelm.py").read())

    print(f"save moodels in {save_directory}")

if __name__ == "__main__":
    save_phoinelm_hf("checkpoints/PhoneLM-1.5B-Call/", torch.bfloat16)