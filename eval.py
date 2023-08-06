
import os
import argparse
import yaml
import random

import numpy as np
import torch


from models.model import Model
from datasets.datasets import create_dataloader
from utils.loss import Category_Loss
from utils.test_scripts import profile_model
from utils.evaluate import evaluate
from utils.general import load_ckpt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_config",
        type=str,
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "config/model.yaml"),
        help="config for constructing the model",
    )
    parser.add_argument(
        "hyp",
        type=str,
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                             "config/hyp.yaml"),
        help="Path to the hyp config directory",
    )
    parser.add_argument(
        "--weights",
        type=str,
        help="Path to the weights file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                             "results"),
        help="Path to the save directory",
    )
    return parser.parse_args()

def main(args):
    with open(args.hyp, "r") as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.model_config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    save_dir = args.save_dir

    random.seed(hyp['np_seed'])
    np.random.seed(hyp['np_seed'])
    torch.manual_seed(hyp['torch_seed'])

    device = torch.device("cuda" if torch.cuda.is_available() and hyp["gpu"] \
                          else "cpu")
    _, val_dataloader = create_dataloader(None, hyp["dataset"], num_workers=1)
    
    model = Model(cfg, hyp["dataset"]["size"])
    load_ckpt(model, args.weights, device)
    # profile_model(model, hyp["dataset"]["size"][::-1], save_dir)

    compute_loss = Category_Loss()
    model = model.to(device)
    loss, acc = evaluate(model, compute_loss, val_dataloader, device, 
                         save_dir)

    

if __name__ =="__main__":
    args = get_args()
    main(args)
