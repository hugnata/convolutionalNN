import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils import convert_to_4_chan, show_image_mask_real_loss, convert_to_1_chan, TrainDataset

def softDice(pred, target):
    return 2. * (pred*target).sum() / (pred+target).sum()

def lossCheloue(pred, target):
    return torch.abs((pred - target)).sum()

def coding_Layer(in_channels, out_channels):
    return [nn.Conv2d(in_channels, out_channels, 3), nn.ReLU(), nn.BatchNorm2d(out_channels), nn.Conv2d(out_channels, out_channels, 3), nn.ReLU(), nn.BatchNorm2d(out_channels), nn.Conv2d(out_channels, out_channels, 3), nn.ReLU(), nn.BatchNorm2d(out_channels)]

def decoding_Layer(in_channels, out_channels):
    return [nn.Conv2d(in_channels, in_channels, 3), nn.ReLU(), nn.BatchNorm2d(in_channels), nn.Conv2d(in_channels, in_channels, 3), nn.ReLU(), nn.BatchNorm2d(in_channels), nn.Conv2d(in_channels, out_channels, 3), nn.ReLU(), nn.BatchNorm2d(out_channels)]


class CNNSEG(nn.Module): # Define your model
    def __init__(self):
        super(CNNSEG, self).__init__()
        self.layer = []
        self.layer += coding_Layer(1, 32)
        self.layer += [nn.MaxPool2d(2)]
        self.layer += coding_Layer(32, 64)
        self.layer += [nn.MaxPool2d(2)]
        self.layer += coding_Layer(64, 128)
        self.layer += [nn.Upsample(scale_factor=2)]
        self.layer += decoding_Layer(128, 64)
        self.layer += [nn.Upsample(scale_factor=2)]
        self.layer += decoding_Layer(64, 32)
        self.layer += [nn.Upsample(scale_factor=2)]
        self.layer += decoding_Layer(32, 4)
        self.layer += [nn.Upsample(size=(96, 96))]
        self.layers = nn.ModuleList(self.layer)
        #self.layers = nn.ModuleList([nn.Conv2d(1, 4, 3), nn.ReLU(), nn.Conv2d(4,8, 3), nn.ReLU(), nn.Conv2d(8, 4, 3), nn.ReLU(), nn.Upsample(size=(96, 96))])
        # fill in the constructor for your model here

    def forward(self, x):
        # fill in the forward function for your model here
        for layer in self.layers:
            # print("Shape: " + str(x.shape))
            # print("Layer:" + str(layer))
            x = layer(x)
        return x

if __name__ == '__main__':

    model = CNNSEG() # We can now create a model using your defined segmentation model

    lr = 0.01
    Loss = nn.L1Loss()
    Loss = softDice
    # Loss = nn.CrossEntropyLoss()
    # Loss = lossCheloue
    optimizer = optim.Adam(model.parameters(), lr=lr)

    data_path = './data/generated'
    num_workers = 1
    batch_size = 10
    train_set = TrainDataset(data_path)
    training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)

    loss_array = []

    for iteration, sample in enumerate(training_data_loader):
        print("Iteration", str(iteration + 1) + "/" + str(len(training_data_loader)))
        optimizer.zero_grad()
        img, mask = sample
        # Converting the mask image [10,96,96] to a matrix [10,1,94,94]. The 94 is to match the forward function output.
        mask_converted = convert_to_4_chan(mask)

        # Adding a new dimension to the img
        img = img[:, None, :, :]

        # Calculating the expected mask
        out = model.forward(img)
        print(out.shape)
        # Display the predicted mask (after conversion to the image format)
        img_1_chan = convert_to_1_chan(out)
        # Calculating the loss
        loss = Loss(out, mask_converted)
        loss_array.append(loss)

        # Display the image, the mask and the loss
        show_image_mask_real_loss(img[0, ...].squeeze(), mask[0, ...].squeeze(), img_1_chan[0, ...].squeeze(), loss_array)
        plt.pause(0.5)

        loss.backward()
        optimizer.step()

    plt.figure()
    plt.plot(loss_array)

# # Layers from Image to 64 Channels + downsampling
# self.layers = [nn.Conv2d(1, 64, 3), nn.ReLU(), nn.Conv2d(64, 64, 3), nn.ReLU(), nn.MaxPool2d(2)]
# # From 64 to 128 + downsampling
# self.layers += [nn.Conv2d(64, 128, 3), nn.ReLU(), nn.Conv2d(128, 128, 3), nn.ReLU(), nn.MaxPool2d(2)]
# # From 128 to 256 + upsampling
# self.layers += [nn.Conv2d(128, 256, 3), nn.ReLU(), nn.Conv2d(256, 256, 3), nn.ReLU(), nn.Upsample(scale_factor=2)]
# # From 256 to 128 + Upsampling
# self.layers += [nn.Conv2d(256, 128, 3), nn.ReLU(), nn.Conv2d(128, 128, 3), nn.ReLU(), nn.Upsample(scale_factor=2)]
# # From 128 to 64
# self.layers += [nn.Conv2d(128, 64, 3), nn.ReLU(), nn.Conv2d(64, 64, 3), nn.ReLU()]
# # From 64 to 4 + upsampling 96*96
# self.layers += [nn.Conv2d(64, 32, 3), nn.ReLU(), nn.Conv2d(32, 4, 1), nn.ReLU(), nn.Upsample(size=(96, 96))]
