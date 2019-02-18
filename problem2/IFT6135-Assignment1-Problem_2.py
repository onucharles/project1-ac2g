
# coding: utf-8

# ## IFT6135: Assigment # 1
# 
# #### By Amlin Charles, Georgina Jiménez, Alejandra Jiménez Nieto, and Charles C. Onu.

# ## Problem #2: Classifying MNIST handwritten digits with ConvNets

# In this exercise, we follow the tutorial reference provided by the TAs for the structure of our CNN model (https://github.com/MaximumEntropy/welcome_tutorials/tree/pytorch/pytorch). We also use the PyTorch Library to build and  train the model.

# In[29]:


import time
import numpy as np
from __future__ import print_function


# In[30]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F


# In[31]:


import torchvision
import torchvision.transforms


# In[32]:


import matplotlib.pyplot as plt


# #### Define image transformations &  Initialize datasets

# In[33]:


mnist_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, transform=mnist_transforms, download=True)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, transform=mnist_transforms, download=True)


# #### Create multi-threaded DataLoaders

# In[34]:


train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=500, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=True, num_workers=2)


# #### Main classifier that subclasses nn.Module
# ##### We have 4 Conv2D layers, each follow by Dropout, ReLU activation and pooling layers.

# In[37]:


class Classifier(nn.Module):
    """Convnet Classifier for MLP-like structure"""
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=36, kernel_size=(3, 3), padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            
            # Layer 2
            nn.Conv2d(in_channels=36, out_channels=72, kernel_size=(3, 3), padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            
            # Layer 3
            nn.Conv2d(in_channels=72, out_channels=144, kernel_size=(3, 3), padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            
            # Layer 4
            nn.Conv2d(in_channels=144, out_channels=288, kernel_size=(3, 3), padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        # Logistic Regression
        self.clf = nn.Linear(288, 10)

    def forward(self, x):
        return self.clf(self.conv(x).squeeze())


# ##### Checking if we have a GPU available.

# In[39]:


cuda_available = torch.cuda.is_available()
print(cuda_available)


# ##### Defining our optimizer and the learning rate for the Gradient descent algorithm.

# In[47]:


clf = Classifier()
if cuda_available:
    clf = clf.cuda()
optimizer = torch.optim.Adam(clf.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
# This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.


# #### Training of the model (up to 50 epochs)

# In[42]:


num_epochs = 50
ep, train_error, valid_error = [], [], []
for epoch in range(num_epochs):
    losses = []
    ep.append(epoch+1)
  
    # Train
    total_train = 0
    correct_train = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if cuda_available:
                    inputs, targets = inputs.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        outputs = clf(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.data.item())

        outputs = clf(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_train += targets.size(0)
        correct_train += predicted.eq(targets.data).cpu().sum()
        
        '''if batch_idx%num_epochs==0:
            print('Epoch : %d Loss : %.3f ' % (epoch, np.mean(losses)))'''
            
    train_error.append(100-100.*correct_train/total_train)
    
    '''total_train = total_train
    correct_train = correct_train'''
    
    # Evaluate
    clf.eval()
    
    total_valid = 0
    correct_valid = 0
    
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if cuda_available:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = clf(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_valid += targets.size(0)
        correct_valid += predicted.eq(targets.data).cpu().sum()

    valid_error.append(100-100.*correct_valid/total_valid)
    
    print('Epoch: %d --> Train Acc.: %.3f, Test Acc.: %.3f' % (epoch+1, 100.*correct_train/total_train, 100.*correct_valid/total_valid))
    print('---------------------------------------------------')
    
    clf.train()


# ##### Structure of the trained model.

# In[43]:


print(clf.eval())


# #### Plotting the training and validation errors

# In[44]:


plt.plot(ep, train_error)
plt.plot(ep, valid_error)
plt.ylabel("Error (%)")
plt.xlabel("Training epochs")
plt.title('Training and validation errors by training epochs.')
#ax.set_xticklabels([''] + v)
plt.xticks(np.arange(min(ep), max(ep), 1))
#plt.yticks(np.arange(min(v)*100, max(v)*100, 0.05))


# More fancy (with labels)

# In[45]:


# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax.plot(ep, train_error, 'k--', label='Training error')
ax.plot(ep, valid_error, 'k:', label='Validation error')
#ax.plot(a, c + d, 'k', label='Total message length')

'''ax.get_xlabel()
ax.get_ylabel("Accuracy (%)")
#ax.get_xlabel("Training epochs")
ax.title('Training and validation errors by training epochs.')
#ax.set_xticklabels([''] + v)
ax.xticks(np.arange(min(ep), max(ep), 1))'''

legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')

plt.show()


# ##### Getting vector of results ready for LaTeX report.

# In[46]:


ep_train_err, ep_valid_err = [],[]
train_err_list=""
valid_err_list=""

for val in range(len(ep)):
    ep_train_err.append((ep[val], float(train_error[val])))
    ep_valid_err.append((ep[val], float(valid_error[val])))
    train_err_list= train_err_list + "(" + str(ep[val]) + ", " + str(float(train_error[val])) + ")"
    valid_err_list= valid_err_list + "(" + str(ep[val]) + ", " + str(float(valid_error[val])) + ")"
    
#print("Epochs: ",ep)
print("Training error: ", ep_train_err)
print("Validation error: " , ep_valid_err)
print("Training error: ", train_err_list)
print("Validation error: " , valid_err_list)

