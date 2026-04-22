from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import _pickle as cpickle


class _ResizedWrapper(Dataset):
    """Wraps an existing dataset and resizes the image tensor AFTER it has been
    produced by the dataset's own transform. This avoids conflicts with any
    Resize already embedded in the stored transform pipeline.
    Must be at module level (not nested) so multiprocessing DataLoader workers can pickle it."""
    def __init__(self, ds, size):
        self.ds = ds
        self.size = size

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, label = self.ds[i]
        # img is a tensor of shape (C, H, W); interpolate needs (1, C, H, W)
        img = F.interpolate(img.unsqueeze(0), size=(self.size, self.size),
                            mode='bilinear', align_corners=False).squeeze(0)
        return img, label


class DatasetSplitDirichlet(Dataset):
    """
    Custom dataset class to process and transform the given image and target data.

    Parameters:
        image (Tensor): The images to process.
        target (Tensor): The target labels corresponding to the images.
        transform (callable): The transformation to apply to the images.
        resize (int, optional): If set, resize the final tensor to (resize x resize)
                                AFTER the stored transform runs. This overrides whatever
                                spatial size the stored transform produces.
    """

    def __init__(self, image, target, transform, resize=None):
        self.image = image
        self.target = target
        self.transform = transform
        self.resize = resize

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        img = self.image[index] / 255.0
        img = self.transform(transforms.ToPILImage()(img))  # -> tensor (C, H, W)
        if self.resize is not None:
            img = F.interpolate(img.unsqueeze(0), size=(self.resize, self.resize),
                                mode='bilinear', align_corners=False).squeeze(0)
        return img, self.target[index]


def load_data(data_iid=True, dev_idx=-1, test_global=False, seed=42, resize=None):
    """
    Loads the appropriate data based on the provided parameters.

    Parameters:
        data_iid (bool, optional): Indicates whether the data is independently and identically distributed.
        dev_idx (int, optional): The index of the device used for loading data.
        test_global (bool, optional): If True, loads the global test data.
        seed (int, optional): Seed for experiment.
        resize (int, optional): If set, force all images to (resize x resize) AFTER the stored
                                transform runs. Use resize=16 with lenet5_slim.
                                This applies via F.interpolate so it always wins over any
                                Resize already baked into the stored transform pipeline.

    Returns:
        DataLoader: A DataLoader object for the loaded data.
    """

    if test_global:
        with open(f"dataset/mnist/seed{seed}/iid/global_test.pkl", 'rb') as f:
            dataset = cpickle.load(f)
        if resize is not None:
            dataset = _ResizedWrapper(dataset, resize)
        data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)
    else:
        if data_iid:
            iidtype = "iid"
        else:
            iidtype = "niid"
        with open(f"dataset/mnist/seed{seed}/{iidtype}/imgs_train_dev{dev_idx}.pkl", 'rb') as f:
            imgs = cpickle.load(f)
        with open(f"dataset/mnist/seed{seed}/{iidtype}/labels_train_dev{dev_idx}.pkl", 'rb') as f:
            labels = cpickle.load(f)
        with open(f"dataset/mnist/seed{seed}/iid/transform_train.pkl", 'rb') as f:
            transform = cpickle.load(f)

        data_loader = DataLoader(
            DatasetSplitDirichlet(image=imgs, target=labels, transform=transform, resize=resize),
            batch_size=8, shuffle=True, num_workers=2
        )

    return data_loader
