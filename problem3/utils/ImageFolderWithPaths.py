"""
Custom dataset that Extends torchvision.datasets.ImageFolder to includes image file paths.
https://github.com/pensieves/accio/blob/master/deep_learn/pytorch/dataset/image/ImageFolderWithPaths.py
"""

from torchvision import datasets

class ImageFolderWithPaths(datasets.ImageFolder):
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path