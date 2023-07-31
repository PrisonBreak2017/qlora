#hard code for kaggle project
import os
os.environ["WANDB_PROJECT"] = "qlora_on_kaggle"
os.environ["WANDB_WATCH"] = "all"
os.environ["WANDB_LOG_MODEL"] = "true"