"""
Scripts makes various plots for the report using log data saved to file.
"""
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import numpy as np
import os
from utils.ImageFolderWithPaths import ImageFolderWithPaths
from torchvision import transforms
import models
import torch
from pathlib import Path
import torch.nn.functional as F

def smooth_data(data, window_width):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return ma_vec

def plot_learning_curves(file_path):
    print('Plot learning curves -')
    print('Loading logs saved at: ', file_path)
    (train_logs, valid_logs) = joblib.load(file_path)

    train_logs = np.array(train_logs)
    valid_logs = np.array(valid_logs)
    print('train logs length ', len(train_logs))
    print('valid logs length ', len(valid_logs))

    plt.figure()

    # smooth accuracy
    window_width = 50
    train_steps = train_logs[:, 0]
    valid_steps = valid_logs[:, 0]
    train_accs_smooth = smooth_data(train_logs[:, 1], window_width)
    valid_accs_smooth = smooth_data(valid_logs[:,1], window_width)
    print('length of smooth train acc ', len(train_accs_smooth))
    print('length of smooth valid acc ', len(valid_accs_smooth))

    # plot accuracy
    plt.subplot(1,2,1)
    plt.plot(train_steps[:len(train_accs_smooth)], train_accs_smooth, label='training')
    plt.plot(valid_steps[:len(valid_accs_smooth)], valid_accs_smooth, label='validation')
    # plt.plot(train_logs[:, 0], train_logs[:, 1], label='training')
    # plt.plot(valid_logs[:,0], valid_logs[:,1], label='validation')
    plt.ylabel('accuracy(%)', fontsize=13)
    plt.xlabel('steps', fontsize=13)
    plt.title('(a)', fontsize=13)
    plt.grid()
    plt.legend(fontsize=13)

    # plot loss
    train_loss_smooth = smooth_data(train_logs[:, 2], window_width)
    valid_loss_smooth = smooth_data(valid_logs[:, 2], window_width)
    plt.subplot(1, 2, 2)
    plt.plot(train_steps[:len(train_loss_smooth)], train_loss_smooth, label='training')
    plt.plot(valid_steps[:len(valid_loss_smooth)], valid_loss_smooth, label='validation')
    plt.ylabel('loss', fontsize=13)
    plt.xlabel('steps', fontsize=13)
    plt.title('(b)', fontsize=13)
    plt.grid()
    plt.legend(fontsize=13)

    plt.show()

def print_misclassified_uncertain_examples(data_dir, model_name, model_path ):
    """
    This function runs tests a model on the validation data and outputs a list of images that:
    1. were misclassified
    2. had predicted probabilities around 50% for both classes.
    """

    # load validation data
    valdir = os.path.join(data_dir, 'valset')
    normalize = transforms.Normalize(mean=[0.490, 0.455, 0.416],
                                     std=[0.252, 0.245, 0.247])
    val_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(valdir, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1000, shuffle=False)

    # load saved model
    device = torch.device("cuda:0")
    print('Loading saved model {} at {}'.format(model_name, model_path))
    model_class = models.find_model(model_name)
    model = model_class().to(device)
    model.load(model_path)

    # test model.
    model.eval()
    test_loss = 0
    test_correct = 0
    predictions = []
    targets = []
    img_paths = []  # get names of each image without full path or extension. (we need names to order result)
    outputs = []

    with torch.no_grad():
        for batch_idx, (data, target, paths) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            outputs.append(output)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            predictions.append(pred)
            img_paths += [Path(path).name for path in paths]
            targets.append(target)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    test_acc = 100. * test_correct / len(val_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, test_correct, len(val_loader.dataset), test_acc))

    predictions = torch.cat(predictions, 0).cpu().numpy().flatten()
    targets = torch.cat(targets, 0).cpu().numpy().flatten()
    img_paths = np.array(img_paths)
    misclassified_idx = targets != predictions
    print('Paths of {} misclassified examples:\n{}'.format(sum(misclassified_idx), img_paths[misclassified_idx]))

    outputs = torch.cat(outputs, 0).cpu().numpy()   # concatenate outputs from all batches
    exp_outputs = np.exp(outputs)                       # return all log estimates to probabilities
    exp_outputs = np.round(exp_outputs, 1)
    uncertain_idx = exp_outputs[:,0] == 0.5
    print('Paths of {} uncertain examples:\n{}'.format(sum(uncertain_idx), img_paths[uncertain_idx]))

def main():
    data_dir = r'C:\Users\Charles\Dropbox (NRP)\black-faces-db\ift6135data'
    output_dir = r'C:\Users\Charles\Dropbox (NRP)\black-faces-db\ift6135output'

    plot_learning_curves(output_dir + '\logs.pkl')
    print_misclassified_uncertain_examples(data_dir, 'BaseConvNet2', output_dir + '\saved_model.pt')

if __name__=='__main__':
    main()