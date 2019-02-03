from __future__ import print_function
from comet_ml import Experiment
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import csv

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

class BaseModel(SerializableModule):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(13*13*50, 200)
        self.fc2 = nn.Linear(200, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 13*13*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def configure_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Cat/Dog classification project')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
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
                        help='For Saving the current Model')
    parser.add_argument('--data_dir', type=str, default=r'C:\Users\Charles\Dropbox (NRP)\travaille\classes\IFT6135\Assignments\assignment1\practical\data',
                        help='root directory where datasets are stored.')
    parser.add_argument('--mode', type=str, default='train', choices=('train', 'test'))

    # add 'mode' - training or eval. so that we can load saved model for either.

    # model arguments
    # parser.add_argument('--n_labels', )
    # parser.add_argument('--n_layers')
    # parser.add_argument('--n_feature_maps')
    # parser.add_argument('--res_pool')

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
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train(args, model, device, train_val_loaders, optimizer, experiment):
    train_loader, val_loader = train_val_loaders
    max_val_acc = 0

    # load saved model, if any.
    if args.saved_model_path:
        model.load(args.saved_model_path)

    # train for all epochs
    for epoch in range(1, args.epochs + 1):
        # log current epoch number on comet
        experiment.log_current_epoch(epoch)

        correct = 0
        total = 0
        # train for all minibatches
        for batch_idx, (data, target) in enumerate(train_loader):
            cur_step = epoch * batch_idx

            model.train()
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            # compute train accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target.data).sum()
            acc = 100. * correct / total

            # log to comet.ml
            experiment.log_metric("train_loss", loss.item(), step=cur_step)
            experiment.log_metric("train_accuracy",  acc.item(), step=cur_step)

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), acc.item()))

                # evaluate on valid set and always keep copy of best model.
                model.eval()
                val_loss = 0
                val_correct = 0
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        val_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        val_correct += pred.eq(target.view_as(pred)).sum().item()

                val_loss /= len(val_loader.dataset)
                val_acc = 100. * val_correct / len(val_loader.dataset)

                experiment.log_metric("validation_loss", val_loss, step=cur_step)
                experiment.log_metric("validation_accuracy", val_acc, step=cur_step)

                print('\tValidation set: loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                    val_loss, val_correct, len(val_loader.dataset), val_acc))

                # save best model
                if val_acc > max_val_acc:
                    print("\tsaving best model...")
                    max_val_acc = val_acc
                    # torch.save(model.state_dict(), "saved_model.pt")
                    model.save("output/saved_model.pt")

def test(args, model, device, test_loader):

    # load saved model, if any.
    if args.saved_model_path:
        model.load(args.saved_model_path)
    else:
        raise('No saved model was specified in test mode.')

    model.eval()
    test_loss = 0
    predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            predictions.append(pred)
    predictions = torch.cat(predictions, 0)

    # test_loss /= len(test_loader.dataset)
    # test_acc = 100. * correct / len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset), test_acc))

    # save predictions
    save_predictions(predictions)

def save_predictions(preds):
    print('Saving predictions...')

    preds = preds.cpu().numpy()
    class_dict = {0: 'Cat', 1: 'Dog'}

    with open('output/predictions.csv', 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['id', 'label'])

        for i in range(preds.shape[0]):
            csvwriter.writerow([str(i+1), class_dict[preds[i][0]]])

def main():
    # Training settings
    args = configure_arguments()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, val_loader, test_loader = create_dataloaders(args)

    model = BaseModel().to(device)
    # model = VGG('VGG11').to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # set up logging.
    experiment = Experiment(api_key="w7QuiECYXbNiOozveTpjc9uPg", project_name="project1-ac2g", workspace="ift6135")
    hyper_params = vars(args)
    experiment.log_parameters(hyper_params)

    # train (if not saved model path is provided)
    if args.mode == 'train':
        print('Running in train mode...')
        train(args, model, device, (train_loader, val_loader), optimizer, experiment)
    elif args.mode == 'test':
        print('Running in test mode...')
        test(args, model, device, test_loader)
        
if __name__ == '__main__':
    main()
