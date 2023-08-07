
import numpy as np
import torch

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
        
        with torch.no_grad():
            output, pred_prob = model(data)

        pred_cls = torch.argmax(output, dim=1).to(torch.long)
        # all_pred_cls.append(pred_cls.cpu().detach())
        all_pred_cls.append(pred_prob[:,1].cpu().detach())
    print(print_s.format(batch_idx+1, l))

    all_pred_cls = torch.cat(all_pred_cls, dim=0).numpy()
    return all_pred_cls, all_img_id

def load_ckpt(model, weights_path, device, optimizer=None):
    ckpt = torch.load(weights_path, map_location=device)
    # model.load_state_dict(ckpt["model"], strict=True)
    model.load_state_dict(ckpt, strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

def calculate_metrics(gt_cls, pred_cls):
    correct = pred_cls == gt_cls
    acc = np.mean(correct)
    bool_gt = gt_cls.astype(np.bool)

    tp = correct[bool_gt].sum()
    fp = pred_cls[np.logical_not(bool_gt)].sum()
    fn = (1-pred_cls[bool_gt]).sum()
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    return acc, precision, recall