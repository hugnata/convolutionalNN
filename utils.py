from matplotlib import pyplot as plt
import torch
import torch.utils.data as data
import cv2
import os
from glob import glob


def show_image_mask_real_loss(img, mask, real, loss, cmap='gray'):  # visualisation
    fig = plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(mask, cmap=cmap)
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(real, cmap=cmap)

    plt.subplot(2, 2, 4)
    plt.plot(loss)

class TrainDataset(data.Dataset):
    def __init__(self, root=''):
        super(TrainDataset, self).__init__()
        self.img_files = glob(os.path.join(root, 'image', '*.png'))
        self.mask_files = []
        for img_path in self.img_files:
            basename = os.path.basename(img_path)
            self.mask_files.append(os.path.join(root, 'mask', basename[:-4] + '_mask.png'))


    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        label = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return len(self.img_files)


class TestDataset(data.Dataset):
    def __init__(self, root=''):
        super(TestDataset, self).__init__()
        self.img_files = glob(os.path.join(root, 'image', '*.png'))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        return torch.from_numpy(data).float()

    def __len__(self):
        return len(self.img_files)

def convert_to_4_chan(img):
    ishape = img.shape
    result = torch.zeros((ishape[0], 4, ishape[1], ishape[2]))
    result[:, 0, :, :] = torch.where(img == 0, 1, 0)
    result[:, 1, :, :] = torch.where(img == 1, 1, 0)
    result[:, 2, :, :] = torch.where(img == 2, 1, 0)
    result[:, 3, :, :] = torch.where(img == 3, 1, 0)
    return result


def convert_to_1_chan(img):
    ishape = img.shape
    result = torch.zeros((ishape[0], ishape[1], ishape[2]))
    result = torch.argmax(img, dim=1)

    return result