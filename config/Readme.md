# Configuration

### model
We use list to configurate the model structure. The flow is from up to down. The layer can be customized at models/model where you can build your layer and use it at model.yaml.
Each layer is configured by 4 arguments as \[\[from\], \[num\], \[layer name\], \[args for layer]\]
Usage:
1. from: meas which layers are as inputs. please use relatice number, i.e. -1 means the previous layer.
2. num: construct #num layers as a sequence model
3. layer name: the class of the layer. Define it at models/model.
4. args for layer: the sequencial args passed to the layer constructor.

```
model:
  - [-1,1, Stem, [3,32]] ## stride = 4
  - [-1,2, ResNeXtBlock, [64, 4, 32, 1]] 
  - [-1,1, BaseConv, [64, 128, 1, 1]]
  - [[-1, -2], 1, Concat, [1]] 
```

### hyperparamenter

```
# set seed
np_seed: 3197 
torch_seed: 1921
# usage of gpu
gpu: True

train:
  epochs: 50
  petrained: None # pretained model path

eval:
  # use test time augment
  test_aug: False


optimizer:
  opt: sgd # optimizer method defined at utils/trainer.py
  lr: 0.01 # learning rate
  weight_decay: 0.0005 # weight decay
  momentum: 0.937 # momentum
  lr_schedular: cosine # scheduler defined at utils/trainer.py
  warmup_epochs: 2 # time for warmup model


dataset:
  # Data root. Be carefully with construct dataset
  root: ./data/train 
  # Training data ratio. The remained for validation
  ratio: 0.9
  labels: ["cat", "dog"]
  batch_size: 16
  
  size: [224, 224] # image size (w, h)
  fliplr: 0.5
  flipud: 0
  scale: [0.7, 1.0]
  rotation: 30
  mean: [0.485, 0.456, 0.406]
  std:  [0.229, 0.224, 0.225]
```