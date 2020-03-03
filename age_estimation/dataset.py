from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms

transforms = transforms.Compose([
    transforms.Resize((60, 60)),
    transforms.ToTensor(),
    transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
])


def load_one_image(image_path, transforms):
    image = Image.open(image_path)
    image = transforms(image).float()
    # image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


class TrainValTestDataset(Dataset):
    def __init__(self, dataset, train_size=0.6, val_size=0.2, mode="train", random_state=42):
        assert mode in ["train", "validate", "test"]
        assert train_size + val_size <= 1.0
        assert train_size <= 1.0
        assert val_size <= 1.0

        self.dataset = dataset

        # make it reproducible
        np.random.seed(random_state)
        self.index = np.arange(len(self.dataset))
        np.random.shuffle(self.index)

        stop1 = int(train_size*len(self.index))
        stop2 = int((train_size + val_size)*len(self.index))

        if mode == "train":
            self.index = self.index[:stop1]
        elif mode == "validate":
            self.index = self.index[stop1:stop2]
        elif mode == "test":
            self.index = self.index[stop2:]

    def __getitem__(self, item: int) -> tuple:
        i = self.index[item]
        return self.dataset[i]

    def __len__(self) -> int:
        return len(self.index)

