from __future__ import print_function
import comet_ml
from comet_ml import Experiment
import argparse
import torch
# import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import csv
from pathlib import Path
import random
import numpy as np
import models
from sklearn.externals import joblib

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def create_folder(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        print("created directory: " + str(newpath))

def configure_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Cat/Dog classification project')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--saved-model-path', type=str, default=None,
                        help='path of a saved model to load')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='root directory where datasets are stored.')
    parser.add_argument('--mode', type=str, default='train',
                        choices=('train', 'test'))
    parser.add_argument('--model', type=str, default='BaseConvNet',
                        choices=('BaseConvNet','VGG', 'ResNet', 'BaseConvNet2'),
                        help='the model architecture to use during training')
    parser.add_argument('--aug-min-crop', type=float, default=0.08,
                        help='the minimum crop scale fraction')

    args = parser.parse_args()
    return args

def create_dataloaders(args):
    """
    Create pytorch data loaders for training and validation set.
    Note that data must already be arranged in 2 sub-folders: 'trainset' and 'valset'.
    """

    # Data loading code
    traindir = os.path.join(args.data_dir, 'trainset')
    valdir = os.path.join(args.data_dir, 'valset')
    testdir = os.path.join(args.data_dir, 'testset')
    normalize = transforms.Normalize(mean=[0.490, 0.455, 0.416],
                                     std=[0.252, 0.245, 0.247])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(args.aug_min_crop,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=6)

    val_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(valdir, transforms.Compose([
            #transforms.Resize(80),
            # transforms.CenterCrop(64),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=6)
    test_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(testdir, transforms.Compose([
            #transforms.Resize(80),
            # transforms.CenterCrop(64),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(args, model, device, train_val_loaders, optimizer, experiment):
    train_loader, val_loader = train_val_loaders
    max_val_acc = 0
    val_acc = 0

    # load saved model, if any.
    if args.saved_model_path:
        model.load(args.saved_model_path)

    # set up learning rate scheduler
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            # mode='max', verbose=True, patience=10)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # train for all epochs
    train_logs = []
    valid_logs = []
    for epoch in range(1, args.epochs + 1):
        # log current epoch number on comet
        experiment.log_current_epoch(epoch)

        correct = 0
        total = 0
        # train for all minibatches
        # scheduler.step(val_acc)
        scheduler.step()
        for batch_idx, (data, target) in enumerate(train_loader):
            cur_step = epoch * batch_idx

            model.train()
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            #loss = F.nll_loss(output, target)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            # compute train accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target.data).sum()
            acc = 100. * correct / total

            # log to comet.ml
            experiment.log_metric("learning_rate",
                    get_learning_rate(optimizer), step=cur_step)
            experiment.log_metric("train_loss", loss.item(), step=cur_step)
            experiment.log_metric("train_accuracy",  acc.item(), step=cur_step)
            train_logs.append((cur_step, acc.item(), loss.item()))

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} of {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(
                    epoch, args.epochs, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), acc.item()))

                # evaluate on valid set and always keep copy of best model.
                model.eval()
                val_loss = 0
                val_correct = 0
                with torch.no_grad():
                    for data, target, paths in val_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        val_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        val_correct += pred.eq(target.view_as(pred)).sum().item()

                val_loss /= len(val_loader.dataset)
                val_acc = 100. * val_correct / len(val_loader.dataset)

                experiment.log_metric("validation_loss", val_loss, step=cur_step)
                experiment.log_metric("validation_accuracy", val_acc, step=cur_step)
                valid_logs.append((cur_step, val_acc, val_loss))

                print('\tValidation set: loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                    val_loss, val_correct, len(val_loader.dataset), val_acc))

                # save best model
                if val_acc > max_val_acc:
                    print("\tsaving best model...")
                    max_val_acc = val_acc
                    model_path = "output/" + args.model + "/saved_model.pt"
                    model.save(model_path)
                    experiment.log_asset(model_path, overwrite=True)
    log_file_path = 'output/' + args.model + '/' + experiment.get_key() + '/logs.pkl'
    joblib.dump((train_logs, valid_logs), log_file_path)
    print('Training logs were written to: ', log_file_path)

    return max_val_acc

def test(args, model, device, test_loader):

    # load saved model, if any.
    if args.saved_model_path:
        print('Loading saved model...')
        model.load(args.saved_model_path)
    else:
        raise('No saved model was specified in test mode.')

    model.eval()
    # test_loss = 0
    # test_correct = 0
    predictions = []
    # targets = []
    img_names = []   # get names of each image without full path or extension. (we need names to order result)

    with torch.no_grad():
        for batch_idx, (data, target, paths) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            predictions.append(pred)
            img_names += [Path(path).name.split('.')[0] for path in paths]
            # targets.append(target)
            # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            # test_correct += pred.eq(target.view_as(pred)).sum().item()

    # I commented it out since we don't have test set targets with which to compute performance metrics.
    # test_loss /= len(test_loader.dataset)
    # test_acc = 100. * test_correct / len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, test_correct, len(test_loader.dataset), test_acc))

    # save predictions
    predictions = torch.cat(predictions, 0)
    save_predictions(args, img_names, predictions)

def save_predictions(args, img_names, predictions):
    print('Saving predictions...')
    preds_path = "output/" + args.model + "/predictions.csv"

    img_names = np.array(img_names)
    predictions = predictions.cpu().numpy().flatten()
    result = dict(zip(img_names, predictions))  # combine image name and prediction into a dictionary.
    class_dict = {0: 'Cat', 1: 'Dog'}

    with open(preds_path, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['id', 'label'])

        for i in range(len(img_names)):
            id = str(i+1)
            label = result[id]
            csvwriter.writerow([id, class_dict[label]])

def tune_hyperparams():
    api_key = "w7QuiECYXbNiOozveTpjc9uPg"
    optimizer = comet_ml.Optimizer(api_key)

    # hyperparameters in PCS format
    params = """
    x integer [1, 10] [10]
    """
    optimizer.set_params(params)

    while True:
        suggestion = optimizer.get_suggestion()
        experiment = Experiment(api_key, project_name="project1-ac2g",
                workspace="ift6135")
        score = train(suggestion["x"])
        suggestion.report_score("accuracy", score)

def main():
    # Training settings
    args = configure_arguments()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # sets seeds to prevent any unwanted randomness.
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:1" if use_cuda else "cpu")

    train_loader, val_loader, test_loader = create_dataloaders(args)

    # get instance of model.
    print('Loading the {0} model...'.format(args.model))
    model_class = models.find_model(args.model)
    model = model_class().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    no_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model has {0} parameters'.format(no_of_params))

    if args.mode == 'train':
        print('Running in train mode...')

        #set up logging.
        experiment = Experiment(api_key="w7QuiECYXbNiOozveTpjc9uPg", project_name="project1-ac2g", workspace="ift6135")
        create_folder('output/' + args.model + '/' + experiment.get_key())
        hyper_params = vars(args)
        experiment.log_parameters(hyper_params)

        train(args, model, device, (train_loader, val_loader), optimizer, experiment)
    elif args.mode == 'test':
        print('Running in test mode...')
        test(args, model, device, test_loader)

if __name__ == '__main__':
    main()
