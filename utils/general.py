

import os
import shutil
import glob


from sklearn import metrics
from matplotlib import pyplot as plt

import torch


def increment_dir(dir, comment=''):
    # Increments a directory runs/exp1 --> runs/exp2_comment
    n = 0  # number
    dir = str(os.path.join(dir, "exp"))  # os-agnostic
    d = sorted(glob.glob(dir + '*'))  # directories
    if len(d):
        n = max([int(x[len(dir):x.find('_') if '_' in x else None]) for x in d]) + 1  # increment
    return dir + str(n) + ('_' + comment if comment else '')

def dump_config(configs, save_dir):
    for config in configs:
        shutil.copy(config, os.path.join(save_dir, os.path.basename(config)))

def plot_line_chart(data, title, labels, save_path, xlim=[0,300], ylim=None):
    fig = plt.figure(figsize=(8,6))
    plt.title(title)
    plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    if isinstance(data[0], list):
        for i in range(len(data)):
            plt.plot(data[i], label=labels[i])
    else:
        plt.plot(data, label=labels)
    plt.legend()
    plt.savefig(save_path)
    plt.close(fig)

def calculate_roc_curve(gt, pred_prob, save_path):
    fpr, tpr, thresholds = metrics.roc_curve(gt, pred_prob)
    auc = metrics.auc(fpr, tpr)
    fig = plt.figure(figsize=(8,6))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot(fpr, tpr)
    plt.savefig(save_path)
    plt.close(fig)
    return fpr, tpr, thresholds, auc

def infer(model, loader, device):
    model = model.to(device)
    model.eval()

    mode = "     [infer] "
    print_s = mode + "step: {0:>5d}/{1:<4d}\t"

    all_pred_cls = []
    all_img_id = []
    l = len(loader)

    for batch_idx, (data, img_id) in enumerate(loader):
        all_img_id.extend(img_id)
        print(print_s.format(batch_idx+1, l), end="\r")
        data = data.to(device)
        output, pred_prob = model(data)

        pred_cls = torch.argmax(output, dim=1).to(torch.long)
        all_pred_cls.append(pred_cls)
    print(print_s.format(batch_idx+1, l))

    all_pred_cls = torch.cat(all_pred_cls, dim=0).cpu().detach().numpy()
    return all_pred_cls, all_img_id

def load_ckpt(model, weights_path, device):
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)

def to_csv(np_data, np_id, save_to):
    save_to = os.path.join(save_to, "submission.csv")
    with open(save_to, 'w') as f:
        f.write("id,label\n")
        s = []
        for i, cls in enumerate(np_data):
            s.append("{0:d},{1:d}".format(np_id[i], cls))
        f.write("\n".join(s))
