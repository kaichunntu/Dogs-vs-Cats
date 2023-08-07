
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

        self.train_loss, self.val_loss, self.val_acc = [], [], []
        self.lr_record = []

        self.nw = None
        self.ni = 0
        self.warmup_lr = [0, opt_hyp["optimizer"]["lr"]]

    def fit(self, train_loader, val_loader):
        print("\n############# start training #############")

        self.nw = self.opt_hyp["optimizer"]["warmup_epochs"]*len(train_loader)

        for epoch in range(1, self.epochs+1):
            
            
            print('Epoch: {0:>4d}/{1}'.format(epoch, self.epochs))
            train_loss = self.train_step(train_loader)

            val_loss, val_acc = self.evaluate(val_loader)
            
            ckpt = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "acc": val_acc
            }
            ## save best model by val_acc
            if val_acc > self.best_acc:
                self.best_loss = val_loss
                self.best_acc = val_acc
                torch.save(ckpt, self.best_model_path)
            
            torch.save(ckpt, self.last_model_path)

            self.process_index(train_loss, val_loss, val_acc)

            self.scheduler.step()
        print("\n############# end training #############")
        self.save_metrics()
        
    def train_step(self, train_loader):
        
        self.model.train()
        total_steps = len(train_loader)
        mode = "  [{0}]".format("training" if self.model.training else "validaion")
        print_s = mode + " step: {0:>5d}/{1:<4d}\tLoss: {2:.4f}" 

        total_loss = 0
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

            lr_s = "\tlr: {0:.5f}".format(group_lr)
            print(print_s.format(batch_idx , total_steps, 
                                 loss.cpu().detach().numpy())+lr_s, end="\r")
            # if batch_idx>50:
            #     break
            
        print(print_s.format(batch_idx, len(train_loader), total_loss/c))
        return total_loss/c
                
    def evaluate(self, val_loader):
        with torch.no_grad():
            loss, acc = evaluate(self.model, self.loss, val_loader, self.device, self.save_dir)
        return loss, acc
    
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
    
    def process_index(self, train_loss, val_loss, val_acc):
        self.lr_record.append(self.optimizer.param_groups[0]["lr"])
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)

        plot_line_chart(self.lr_record, "lr", labels="cosine", 
                        save_path=os.path.join(self.save_dir, 'lr.png'),
                        xlim=[0, self.epochs], ylim=[0, max(self.lr_record)*1.05])
        plot_line_chart([self.train_loss, self.val_loss], "loss", labels=['train', 'val'], 
                        save_path=os.path.join(self.save_dir, 'loss.png'),
                        xlim=[0, self.epochs])
        plot_line_chart(self.val_acc, "accuracy", labels="val", 
                        save_path=os.path.join(self.save_dir, 'accuracy.png'),
                        xlim=[0, self.epochs])
        
    def save_metrics(self):
        with open(os.path.join(self.save_dir, "results.txt"), "w") as f:
            f.write("epoch\ttrain_loss\tval_loss\tval_acc\n")
            s = []
            _s = "{0:d}\t{1:.4f}\t{2:.4f}\t{3:.4}"
            for i in range(1, len(self.train_loss)+1):
                s.append(_s.format(i, self.train_loss[i-1], 
                                   self.val_loss[i-1], self.val_acc[i-1]))
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