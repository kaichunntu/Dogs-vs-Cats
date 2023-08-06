
from torchvision import transforms

def get_transformer(hyp):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(hyp["fliplr"]),
        transforms.RandomRotation(hyp["rotation"])
    ])

def get_preprocessor(hyp):
    ## Resize needs (h, w), transform (w,h) to (h, w)
    return transforms.Compose([
        transforms.Resize(hyp["size"][::-1]),
        transforms.ToTensor(),
        transforms.Normalize(hyp["mean"], hyp["std"], inplace=True)
    ])