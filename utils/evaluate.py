
import os
import sys

import torch
import numpy as np

from .general import calculate_roc_curve
from .torch_utils import calculate_metrics

def evaluate(model, compute_loss, test_loader, device, save_dir=None):
    model.eval()
    mode = "      [eval] "
    print_s = mode + "step: {0:>5d}/{1:<4d}\t"

    # init metric values
    total_loss = 0
    c = 0
    all_pred_prob = []
    all_pred_cls = [] 
    all_gt = []
    
    total_steps = len(test_loader)
    for batch_idx, (data, target) in enumerate(test_loader):
        print(print_s.format(batch_idx, total_steps), end="\r")
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output, pred_prob = model(data)
        loss = compute_loss(output, target)
        total_loss += loss.cpu().detach().numpy()*data.size(0)
        c += data.size(0)

        pred_cls = torch.argmax(output, dim=1).to(torch.long)
        ## transform softmax output to binary classification
        ## class 0 is cat and class 1 is dog. Thus, the sum of them will be 1.0
        ## The position of 1 is equal to the probability of class.
        pred_prob = pred_prob[:,1]
        all_pred_prob.append(pred_prob.cpu().detach().numpy())
        all_pred_cls.append(pred_cls.cpu().detach().numpy())
        all_gt.append(target.cpu().detach().numpy())

        # if batch_idx>50:
        #     break

    all_pred_prob = np.concatenate(all_pred_prob, axis=0)
    all_pred_cls = np.concatenate(all_pred_cls, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)

    # mean loss
    loss = total_loss/c

    # compute accuracy
    acc, precision, recall = calculate_metrics(all_gt, all_pred_cls)
    fpr, tpr, thr, auc = calculate_roc_curve(all_gt, all_pred_prob, 
                                        os.path.join(save_dir, "roc_curve.png"))
    metric_s = "Loss: {:.4f}\tAcc: {:.4f}\tAUC:{:.4f}\tPrecision:{:.4f}\tRecall:{:.4f}".\
        format(loss, acc, auc, precision, recall)

    print(print_s.format(batch_idx+1, total_steps) + metric_s)
    return {
        "loss": loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "auc": auc
    }