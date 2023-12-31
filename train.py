
import os
import sys
import random
import argparse
import yaml
import copy

import numpy as np
import torch

from models.model import Model
from datasets.datasets import create_dataloader
from utils.loss import Category_Loss
from utils.trainer import Trainer
from utils.general import increment_dir, dump_config, plot_label_hist
from utils.test_scripts import profile_model




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_config",
        type=str,
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                             "config/model.yaml"),
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
        "--log_dir",  
        type=str, 
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs"),
        help="Path for saving logs",
    )
    return parser.parse_args()


def main(args):
    with open(args.hyp, "r") as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.model_config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    save_dir = increment_dir(args.log_dir)
    print("Log dir: {}".format(save_dir))
    os.makedirs(save_dir, exist_ok=False)
    os.makedirs(os.path.join(save_dir, "weights"), exist_ok=False)
    dump_config([args.model_config, args.hyp], save_dir)
    
    random.seed(hyp['np_seed'])
    np.random.seed(hyp['np_seed'])
    torch.manual_seed(hyp['torch_seed'])


    device = torch.device("cuda" if torch.cuda.is_available() and hyp["gpu"] else "cpu")

    train_dataloader, val_dataloader = create_dataloader(None, hyp["dataset"], num_workers=1)
    # test_dataloader(train_dataloader)
    labels = [train_dataloader.dataset.labels[i] for i in train_dataloader.dataset.idxs]
    plot_label_hist(labels, hyp["dataset"]["labels"],
                    os.path.join(save_dir, "train_label_hist.png"))
    
    model = Model(cfg, hyp["dataset"]["size"])
    _model = copy.deepcopy(model)
    profile_model(_model, hyp["dataset"]["size"][::-1], save_dir)
    del _model
    compute_loss = Category_Loss(label_weight=train_dataloader.dataset.get_label_weight())
    compute_loss.to(device)

    trainer = Trainer(model, compute_loss, hyp, device, save_dir)
    try:
        trainer.fit(train_dataloader, val_dataloader)
    except KeyboardInterrupt:
        print("Saving metrics...")
    except Exception as e:
        trainer.save_metrics()
        print("save logs at {}".format(save_dir))
        raise e
    trainer.save_metrics()
    print("save logs at {}".format(save_dir))



if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
    