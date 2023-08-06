
import os
import sys

import torch
import numpy as np

from .general import calculate_roc_curve

def evaluate(model, compute_loss, test_loader, device, save_dir=None):
    model.eval()
    mode = "      [eval] "
    print_s = mode + "step: {0:>5d}/{1:<4d}\t"

    # init metric values
    total_loss = 0
    c = 0
    total_correct = []
    all_pred_prob = []
    all_gt = []
    
    total_steps = len(test_loader)
    for batch_idx, (data, target) in enumerate(test_loader):
        print(print_s.format(batch_idx, total_steps), end="\r")
        data, target = data.to(device), target.to(device)
        
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
        all_gt.append(target.cpu().detach().numpy())

        correct = pred_cls.eq(target).cpu().detach().numpy()
        total_correct.append(correct)
        # if batch_idx>50:
        #     break

    all_pred_prob = np.concatenate(all_pred_prob, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)

    # mean loss
    loss = total_loss/c

    # compute accuracy
    total_correct = np.concatenate(total_correct, axis=0)
    acc = total_correct.mean()
    fpr, tpr, thr, auc = calculate_roc_curve(all_gt, all_pred_prob, 
                                        os.path.join(save_dir, "roc_curve.png"))
    
    print(print_s.format(batch_idx, total_steps) +
          "Loss: {:.4f}\tAcc: {:4.2f}\tAUC:{:.3f}".format(loss, acc*100, auc))
    return loss, acc 