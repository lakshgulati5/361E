from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import _pickle as cpickle


class DatasetSplitDirichlet(Dataset):
    """
    Custom dataset class to process and transform the given image and target data.

    Parameters:
        image (Tensor): The images to process.
        target (Tensor): The target labels corresponding to the images.
        transform (callable): The transformation to apply to the images.

    """

    def __init__(self, image, target, transform):
        self.image = image
        self.target = target
        self.transform = transform

    def __len__(self):
        """
        Returns:
            int: The number of images in the dataset.
        """
        return len(self.image)

    def __getitem__(self, index):
        """
        Returns the transformed image and corresponding label at the given index.

        Parameters:
            index (int): Index of the image to retrieve.

        Returns:
            tuple: Transformed image and its corresponding label.
        """
        img = self.image[index] / 255.0
        img = self.transform(transforms.ToPILImage()(img))
        return img, self.target[index]


def load_data(data_iid=True, dev_idx=-1, test_global=False, seed=42):
    """
    Loads the appropriate data based on the provided parameters.

    Parameters:
        data_iid (bool, optional): Indicates whether the data is independently and identically distributed.
        dev_idx (int, optional): The index of the device used for loading data.
        test_global (bool, optional): If True, loads the global test data. This parameter is considered only when
                                      'test' is True.
        seed (int, optional): Seed for experiment.

    Returns:
        DataLoader: A DataLoader object for the loaded data.
    """

    if test_global:
        with open(f"dataset/mnist/seed{seed}/iid/global_test.pkl", 'rb') as f:
            dataset = cpickle.load(f)
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
        data_loader = DataLoader(DatasetSplitDirichlet(image=imgs, target=labels, transform=transform),
                                 batch_size=8, shuffle=True, num_workers=2)

    return data_loader




