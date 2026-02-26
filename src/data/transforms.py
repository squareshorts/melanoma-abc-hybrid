from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transforms(image_size: int, normalize: str = "imagenet", train: bool = True):
    tfms = [transforms.Resize((image_size, image_size))]
    if train:
        tfms += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=10),
        ]
    tfms.append(transforms.ToTensor())
    if normalize == "imagenet":
        tfms.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(tfms)
