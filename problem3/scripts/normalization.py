"""
This script computes the normalization parameters (mean and standard deviation), per channel,
for the training data (RGB images). So that we can use these values in the data loading pipeline when training our models
"""

import os
import argparse
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

def main(traindir):
    """
    Compute normalisation parameters - mean and std.
    """

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ]))

    channel0 = np.zeros((len(train_dataset), 64 * 64))
    channel1 = np.zeros((len(train_dataset), 64 * 64))
    channel2 = np.zeros((len(train_dataset), 64 * 64))
    for i in range(len(train_dataset)):
        img = train_dataset[i][0]
        channel0[i, :] = img[0, :, :].view(-1)
        channel1[i, :] = img[1, :, :].view(-1)
        channel2[i, :] = img[2, :, :].view(-1)


    # take mean and std of each channel
    means = np.mean(channel0), np.mean(channel1), np.mean(channel2)
    stds = np.std(channel0), np.std(channel1), np.std(channel2)
    print('means ', means)
    print('std ', stds)

def test(traindir):
    """
     Test computed parameters to confirm that data is actually normalised.
    """

    # Mean and std pre-computed from training set.
    normalize = transforms.Normalize(mean = [0.490, 0.455, 0.416],
                                     std = [0.252, 0.245, 0.247])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    for i in range(len(train_dataset)):
        img = train_dataset[i][0]

        print('mean ', torch.mean(img[0, :, :]))    # mean should be around 0
        print('std ', torch.std(img[0, :, :]))      # std should be around 1

        if i == 100:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default=r'C:\Users\Charles\Dropbox (NRP)\travaille\classes\IFT6135\Assignments\assignment1\practical\data',
                        help='root directory where datasets are stored.')
    args = parser.parse_args()

    traindir = os.path.join(args.data_dir, 'trainset')
    main(traindir)