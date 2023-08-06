
import os


from PIL import Image
import numpy as np

from typing import Any
from torch.utils.data import Dataset, DataLoader
from .data_aug import get_transformer, get_preprocessor



def create_dataloader(data_path, da_hyp, num_workers=0):
    print("## Create DataLoader ##")

    label_encoder = {}
    for i, label in enumerate(da_hyp["labels"]):
        label_encoder[label] = i

    if data_path is None:
        data_path = da_hyp["root"]

    data_list = []
    for img_p in os.listdir(data_path):
        data_list.append(os.path.join(data_path, img_p))

    idxs = np.arange(len(data_list))
    np.random.shuffle(idxs)
    train_idxs = idxs[:int(len(idxs) * da_hyp["ratio"])]
    valid_idxs = idxs[int(len(idxs) * da_hyp["ratio"]):]

    train_dataset = PetsDataset(data_list, label_encoder, train_idxs, 
                                da_hyp=da_hyp, training=True)
    valid_dataset = PetsDataset(data_list, label_encoder, valid_idxs, 
                                da_hyp=da_hyp, training=False)

    train_loader = DataLoader(train_dataset, batch_size=da_hyp["batch_size"], shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=da_hyp["batch_size"], shuffle=False, 
                              num_workers=num_workers, pin_memory=True)
    
    print("Finish creating dataloader\n")
    return train_loader, valid_loader




class PetsDataset(Dataset):
    def __init__(self, image_list, label_encoder, idxs, da_hyp, training):
        super(PetsDataset, self).__init__()
        self.image_list = image_list
        self.labels = [label_encoder[os.path.basename(_).split(".")[0]] for _ in self.image_list]
        self.idxs = idxs
        self.training = training

        self.preprocessor = get_preprocessor(da_hyp)
        if self.training:
            self.transform = get_transformer(da_hyp)
        

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, index: Any) -> Any:
        img_p = self.image_list[self.idxs[index]]

        img = Image.open(img_p)

        if self.training:
            img = self.transform(img)

        img = self.preprocessor(img)
        return img, self.labels[self.idxs[index]]