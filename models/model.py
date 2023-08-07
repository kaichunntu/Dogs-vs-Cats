
import torch
from torch import nn

from .blocks import *
from .common_layers import BaseConv, get_act_layer, Stem, Concat
from .heads import ClassificationHead

class Model(nn.Module):
    def __init__(self, cfg, image_size, verbose=False):
        super(Model, self).__init__()
        self.cfg = cfg
        self.image_size = image_size
        print("## Create Model ##")
        self.layers, self.routes = parse_model_cfg(self.cfg, verbose=verbose)

        print("Finished creating model\n")

    def forward(self, x):
        record_tensor = []
        for i, (m, route) in enumerate(zip(self.layers, self.routes)):
            
            if m.f!=-1:
                if isinstance(m.f, int):
                    x = record_tensor[m.f]  
                else:
                    x = [x if j == -1 else record_tensor[j] for j in m.f] 
            x = m(x)
            if route:
                record_tensor.append(x)
            else:
                record_tensor.append(None)

        return x
    
class TTAWrapper(nn.Module):
    def __init__(self, model):
        super(TTAWrapper, self).__init__()
        self.model = model

    def generate_tta_sample(self, x):
        funcs = [
            lambda x: x,
            lambda x: torch.fliplr(x)
        ]
        for f in funcs:
            yield f(x)

    def forward(self, x):
        output_logits, output_prob = [],[]
        for _x in self.generate_tta_sample(x):
            logits, prob = self.model(_x)
            output_logits.append(logits)
            output_prob.append(prob)
        output_logits = torch.stack(output_logits, dim=0).mean(dim=0)
        output_prob = torch.stack(output_prob, dim=0).mean(dim=0)
        return output_logits, output_prob

    




def parse_model_cfg(cfg, verbose=False):
    layer_list = nn.ModuleList()
    ## routes indicate which layers should be memorized 
    routes = []

    for idx, layer_cfg in enumerate(cfg["model"]):
        if verbose: print(idx, layer_cfg)
        f, num_layer, func, args = layer_cfg
        
        if f!=-1:
            if isinstance(f, int):
                _f = [f]
            else:
                _f = f
            for _pos in f:
                if _pos == -1: continue
                # set True for the tensor which need to pass to deep layer
                routes[_pos] = True
        ## current tensor is set to False
        routes.append(False)

        if num_layer > 1:
            m = []
            for _ in range(num_layer):
                m.append(eval(func)(*args))
            layer = nn.Sequential(*m)
        else:
            layer = eval(func)(*args)
        layer.f = f
        layer_list.append(layer)

    return layer_list, routes
            
                