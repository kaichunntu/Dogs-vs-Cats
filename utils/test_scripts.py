
import os
import numpy as np
import torch
from thop import profile

def test_dataloader(dataloader):
    os.makedirs("runs/image", exist_ok=True)
    from PIL import Image

    labels = ["cat", "dog"]
    for img, target in dataloader:
        for i in range(dataloader.batch_size):
            np_im = (img[i].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            print(np_im.shape, np_im[12,23])
            im = Image.fromarray(np_im)
            im.save(os.path.join("runs/image", f"{labels[target[i]]}_{i}.png"))

        break

def profile_model(model, size, save_dir):
    i = 0
    # handles = []
    # def print_tensor_hook(m, in_t, out_t): 
    #     nonlocal i 
    #     print(i, in_t[0].size())
    #     i+=1

    # def apply_all_hook(m):
    #     nonlocal handles
    #     if isinstance(m, torch.nn.Conv2d):
    #         handles.append(m.register_forward_hook(print_tensor_hook))
    # model.apply(apply_all_hook)
    
    dummy_input = torch.randn(1, 3, *size)
    model.eval()
    output = model(dummy_input)
    
    # for h in handles:
    #     h.remove()

    torch.onnx.export(model, dummy_input, os.path.join(save_dir, "model.onnx"))

    macs, params = profile(model, inputs=(dummy_input, ))
    print("Model Flops: {0:6.3f}G, Params: {1:6.3f}M".\
          format( macs / 1e9, params / 1e6))

