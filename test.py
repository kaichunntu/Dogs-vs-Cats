
import os
import argparse

from utils.evaluate import evaluate


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_config",
        type=str,
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "config/model.yaml"),
        help="config for constructing the model",
    )
    parser.add_argument(
        "weights",
        type=str,
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "config/hyp.yaml"),
        help="Path to the model weights file",
    )
    return parser.parse_args()

def main(args):
    pass

if __name__ =="__main__":
    args = get_args()
    main(args)
