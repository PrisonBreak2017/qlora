#hard code for kaggle project
import os


os.environ["WANDB_PROJECT"] = "qlora_on_kaggle"
os.environ["WANDB_WATCH"] = "all"
os.environ["WANDB_LOG_MODEL"] = "true"

import wandb
wandb.init(project=os.environ["WANDB_PROJECT"])
wandb.watch(os.environ["WANDB_WATCH"] )
wandb.log_model = os.environ["WANDB_LOG_MODEL"]

