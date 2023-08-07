
import os
import sys


import torch
import numpy as np
from .evaluate import evaluate
from .general import plot_line_chart
from .torch_utils import load_ckpt


class Trainer:
    def __init__(self, model, loss, opt_hyp, device, save_dir):
        self.model = model.to(device)
        self.epochs = opt_hyp["train"]["epochs"]
        self.opt_hyp = opt_hyp
        self.device = device
        self.save_dir = save_dir
        self.best_model_path = os.path.join(self.save_dir, 'weights/best_model.pt')
        self.last_model_path = os.path.join(self.save_dir, 'weights/last_model.pt')

        self.loss = loss
        self.optimizer = configure_optimizer(model, opt_hyp["optimizer"])
        self.scheduler = configure_schedular(self.optimizer, 
                                             opt_hyp["optimizer"], epochs=self.epochs)
        if opt_hyp["train"]["pretrained"] is not None:
            print("Loading pretrained model from {}".format(opt_hyp["train"]["pretrained"]))
            load_ckpt(self.model, opt_hyp["train"]["pretrained"], device, 
                      optimizer=self.optimizer)
        else:
            print("Random initialization of model weights")

        self.best_loss = np.inf
        self.best_acc = 0.0

        self.train_records = {
            "loss": [], 
            "acc": [],
        }
        self.val_records = {
            "loss": [],
            "acc": [],
            "precision": [],
            "recall": [],
            "auc": []
        }
        self.lr_record = []

        self.nw = None
        self.ni = 0
        self.warmup_lr = [0, opt_hyp["optimizer"]["lr"]]

    def fit(self, train_loader, val_loader):
        print("\n############# start training #############")

        self.nw = self.opt_hyp["optimizer"]["warmup_epochs"]*len(train_loader)

        for epoch in range(1, self.epochs+1):
            
            
            print('Epoch: {0:>4d}/{1}'.format(epoch, self.epochs))
            train_metrics = self.train_step(train_loader)

            val_metrics = self.evaluate(val_loader)
            val_acc = val_metrics["acc"]
            ckpt = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "acc": val_acc,
                "val_metric": self.val_records,
                "train_metric": self.train_records
            }
            ## save best model by val_acc
            if val_acc > self.best_acc:
                self.best_loss = val_metrics["loss"]
                self.best_acc = val_acc
                torch.save(ckpt, self.best_model_path)
            ## save last model
            torch.save(ckpt, self.last_model_path)

            self.process_index(train_metrics, val_metrics)

            self.scheduler.step()
        print("\n############# end training #############")
        
    def train_step(self, train_loader):
        
        self.model.train()
        total_steps = len(train_loader)
        mode = "  [{0}]".format("training" if self.model.training else "validaion")
        print_s = mode + " step: {0:>5d}/{1:<4d}\tLoss: {2:.4f}" 

        total_loss = 0
        corrects = []
        c = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            ## warmup model
            group_lr = self.warmup_model()

            ## forward
            self.optimizer.zero_grad()
            output = self.model(data)

            ## compute loss
            loss = self.loss(output, target)

            ## optimize model
            loss.backward()
            self.optimizer.step()

            ## record loss
            total_loss += (loss*data.size(0)).cpu().detach().numpy()
            c += data.size(0)
            pred_cls = torch.argmax(output, dim=1)
            corrects.append((pred_cls==target).cpu().detach().numpy())

            lr_s = "\tlr: {0:.5f}".format(group_lr)
            print(print_s.format(batch_idx , total_steps, 
                                 loss.cpu().detach().numpy())+lr_s, end="\r")
            if batch_idx>50:
                break
        mean_loss = total_loss/c    
        corrects = np.concatenate(corrects, axis=0)
        acc = corrects.mean()
        
        print(print_s.format(batch_idx, len(train_loader), mean_loss)+"\tAcc: {0:.4f}".format(acc))
        return {"loss": mean_loss, "acc": acc}
                
    def evaluate(self, val_loader):
        metrics = evaluate(self.model, self.loss, val_loader, self.device, self.save_dir)
        return metrics
    
    def warmup_model(self):
        self.ni+=1
        if self.ni > self.nw:
            return self.optimizer.param_groups[0]["lr"]
        for j, group in enumerate(self.optimizer.param_groups):
            group['lr'] = np.interp(self.ni, [0,self.nw], 
                                [group["initial_lr"]*0.05, group["initial_lr"]])
            group_lr = group['lr']
            if 'momentum' in group:
                group['momentum'] = np.interp(self.ni, [0,self.nw],
                                        [0.8, self.opt_hyp["optimizer"]["momentum"]])
        return group_lr
    
    def process_index(self, train_metrics, val_metric):
        ## record lr
        self.lr_record.append(self.optimizer.param_groups[0]["lr"])
        ## record train metric
        for key, v in train_metrics.items():
            self.train_records[key].append(v)
        ## record val metric
        for key, v in val_metric.items():
            self.val_records[key].append(v)

        plot_line_chart(self.lr_record, "lr", labels="cosine", 
                        save_path=os.path.join(self.save_dir, 'lr.png'),
                        xlim=[0, self.epochs], ylim=[0, max(self.lr_record)*1.05])
        for key in ["loss", "acc"]:
            plot_line_chart([self.train_records[key], self.val_records[key]], key, labels=['train', 'val'], 
                            save_path=os.path.join(self.save_dir, key+".png"),
                            xlim=[0, self.epochs])
        # plot_line_chart([self.train_loss, self.val_records["loss"]], "loss", labels=['train', 'val'], 
        #                 save_path=os.path.join(self.save_dir, 'loss.png'),
        #                 xlim=[0, self.epochs])
        # _data = [self.train_records["acc"], self.val_records["acc"]]
        # _label = ["train", "val"]
        # plot_line_chart(_data, "accuracy", labels=_label, 
        #                 save_path=os.path.join(self.save_dir, 'accuracy.png'),
        #                 xlim=[0, self.epochs])
        
        _data = []
        _label = []
        for key in ["precision", "recall", "auc"]:
            _data.append(self.val_records[key])
            _label.append(key)
        plot_line_chart(_data, "metrics", labels=_label, 
                        save_path=os.path.join(self.save_dir, 'metrics.png'),
                        xlim=[0, self.epochs])
        
    def save_metrics(self):
        with open(os.path.join(self.save_dir, "results.txt"), "w") as f:
            keys = ["epoch", "train_loss", "loss", "acc", 
                    "precision", "recall", "auc"]
            f.write("\t".join(keys)+"\n")
            s = []
            _s = "{0:d}"
            for i in range(1, len(keys)+1):
                _s += "\t{"+"{0:d}".format(i)+":.4f}"
            for i in range(1, len(self.train_train_records["loss"])+1):
                val_data = [self.val_records[key][i-1] for key in keys[2:]]
                s.append(_s.format(i, self.train_records["loss"][i-1], *val_data))
            f.write("\n".join(s))


        

def configure_optimizer(model, hyp):
    if hyp['opt'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=hyp['lr'])
    elif hyp['opt'] =='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=hyp['lr'], 
                        momentum=hyp['momentum'], weight_decay=hyp["weight_decay"])
    else:
        raise NotImplementedError

    return optimizer

def configure_schedular(optimizer, opt_hyp, epochs):
    if opt_hyp['lr_schedular'] == 'cosine':
        lr_schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                            T_max=epochs, eta_min=opt_hyp["lr"]*0.01)
    else:
        raise NotImplementedError
    return lr_schedular