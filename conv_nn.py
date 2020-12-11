import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils import convert_to_4_chan, show_image_mask_real, convert_to_1_chan, TrainDataset


class CNNSEG(nn.Module): # Define your model
    def __init__(self):
        super(CNNSEG, self).__init__()
        #torch.nn.Conv2d(in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T, T]])
        self.layers = nn.ModuleList([nn.Conv2d(1, 8, 3), nn.Conv2d(8, 6, 3), nn.Conv2d(6, 4, 3), nn.Upsample(size=(96, 96))])
        # fill in the constructor for your model here
    def forward(self, x):
        # fill in the forward function for your model here
        for layer in self.layers:
            x = layer(x)
        return x
if __name__ == '__main__':

    model = CNNSEG() # We can now create a model using your defined segmentation model

    lr = 0.00001
    Loss = nn.L1Loss()

    optimizer = optim.SGD(model.parameters(), lr=lr)

    data_path = './data/train'
    num_workers = 1
    batch_size = 10
    train_set = TrainDataset(data_path)
    training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)

    loss_array = []

    for iteration, sample in enumerate(training_data_loader):
        print("Iteration", str(iteration + 1) + "/" + str(len(training_data_loader)))

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
        # Display the image and the mask
        show_image_mask_real(img[0, ...].squeeze(), mask[0, ...].squeeze(), img_1_chan[0].squeeze())
        plt.pause(0.5)

        # Calculating the loss
        loss = Loss(out, mask_converted)
        loss_array.append(loss)

        loss.backward()
        optimizer.step()

    plt.figure()
    plt.plot(loss_array)

