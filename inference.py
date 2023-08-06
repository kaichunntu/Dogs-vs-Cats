

import os
import sys
import argparse
import yaml

import torch

from datasets.datasets import create_inference_dataloader
from models.model import Model
from utils.test_scripts import profile_model
from utils.general import infer, load_ckpt, to_csv

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_root",
        type=str,
        help="Path to the image directory"
    )
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
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--conf_thr",
        type=float,
        default=0.5,
        help="Confidence threshold for classification",
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
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = create_inference_dataloader(args.image_root, hyp["dataset"], 
                                              args.batch_size, num_workers=1)
    
    model = Model(cfg, hyp["dataset"]["size"])
    load_ckpt(model, args.weights, device)
    # profile_model(model, hyp["dataset"]["size"][::-1], save_dir)

    pred_cls, img_id = infer(model, data_loader, device)
    to_csv(pred_cls, img_id, save_dir)




if __name__ =="__main__":
    args = get_args()
    main(args)