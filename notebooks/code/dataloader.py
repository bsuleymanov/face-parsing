import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path



class CelebAMaskHQDataset:
    def __init__(self, image_path, mask_path, image_transform,
                 mask_transform, mode, verbose=0):
        self.image_dir = Path(image_path)
        self.mask_dir = Path(mask_path)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.train_dataset = []
        self.test_dataset = []
        self.mode = mode
        self.verbose = verbose

        self.preprocess()

        if mode == "train":
            self.n_images = len(self.train_dataset)
        else:
            self.n_images = len(self.test_dataset)

    def preprocess(self):
        for i in range(len(list(self.image_dir.glob("*.jpg")))):
            image_path = self.image_dir / f"{i}.jpg"
            mask_path = self.mask_dir / f"{i}.png"
            if self.verbose > 0:
                print(image_path, mask_path)
            if self.mode == "train":
                self.train_dataset.append([image_path, mask_path])
            else:
                self.test_dataset.append([image_path, mask_path])

        if self.verbose > 0:
            print("Finished preprocessing the CelebA dataset...")

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == "train" else self.test_dataset
        image_path, mask_path = dataset[index]
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        return self.image_transform(image), self.mask_transform(mask)

    def __len__(self):
        return self.n_images


class CelebAMaskHQLoader:
    def __init__(self, image_path, mask_path, image_size, batch_size, mode):
        self.image_dir = Path(image_path)
        self.mask_dir = Path(mask_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.mode = mode

    def transform(self):
        image_transform = transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()])

        mask_transform = transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Resize((self.image_size, self.image_size))])

        return image_transform, mask_transform

    def loader(self):
        image_transform, mask_transform = self.transform()
        dataset = CelebAMaskHQDataset(self.image_dir, self.mask_dir,
                                      image_transform, mask_transform, self.mode)
        loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size,
            shuffle=True if self.mode == "train" else False,
            num_workers=2, drop_last=False)

        return loader