
import os
import argparse
import yaml
import random

import numpy as np
import torch


from models.model import Model, TTAWrapper
from datasets.datasets import create_dataloader, create_test_dataloader
from utils.loss import Category_Loss
from utils.test_scripts import profile_model
from utils.evaluate import evaluate
from utils.torch_utils import load_ckpt

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
        "--data_root",
        type=str,
        default=None,
        help="Path to the data directory",
    )
    parser.add_argument(
        "--use_tta",
        action="store_true",
        help="Use test time augmentation"
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
    if args.data_root is None:
        ## test extra dataset
        _, val_dataloader = create_dataloader(None, hyp["dataset"], 
                                              num_workers=1)
    else:
        ## reproduce results of valid dataset as training process
        val_dataloader = create_test_dataloader(args.data_root, hyp["dataset"], 
                                                num_workers=1)
    
    model = Model(cfg, hyp["dataset"]["size"])
    load_ckpt(model, args.weights, device)
    # profile_model(model, hyp["dataset"]["size"][::-1], save_dir)
    
    ## Set loss
    label_weight = [1.0 for _ in range(val_dataloader.dataset.nc)]
    compute_loss = Category_Loss(label_weight=label_weight)
    compute_loss.to(device)

    ## use tta
    if args.use_tta:
        model = TTAWrapper(model)

    model.eval()
    model = model.to(device)
    val_metrics = evaluate(model, compute_loss, val_dataloader, device, 
                         save_dir)

    

if __name__ =="__main__":
    args = get_args()
    main(args)
