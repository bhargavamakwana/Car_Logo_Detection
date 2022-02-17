import torch
import torchvision
import numpy as np
from torchvision import transforms
import torch.nn as  nn
import time
import copy
from torch.autograd import Variable
import torch.nn.functional as F



# Hyper parameters

num_epochs = 5
batchsize = 2
lr = 1e-3

EPOCHS = 2
BATCH_SIZE = 5
LEARNING_RATE = 3e-3
TRAIN_DATA_PATH = "./dataset_car_logo/Train"
TEST_DATA_PATH = "./dataset_car_logo/Test"


TRANSFORM_IMG = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH,
                                             transform=TRANSFORM_IMG)

test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH,
                                             transform=TRANSFORM_IMG)

CLASSES = train_data.classes.copy()


train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batchsize,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                           batch_size=batchsize,
                                           shuffle=True)

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"


class MultiClassCNN(torch.nn.Module):
  def __init__(self):
    super(MultiClassCNN, self).__init__()
    self.model = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=20, kernel_size=2, stride=4), # 10 * 63 * 63
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2),
                    torch.nn.Conv2d(in_channels=20, out_channels=40, kernel_size=4, stride=2), # 10 * 6 * 6
                    torch.nn.ReLU()
                    )
    self.classifier = torch.nn.Sequential(
                      torch.nn.Linear(in_features=1440, out_features=720),
                      torch.nn.ReLU(),
                      torch.nn.Linear(in_features=720, out_features=360),
                      torch.nn.Linear(in_features=360, out_features=18))

  def forward(self, x):
    x = self.model(x)
    x = x.reshape(x.shape[0], -1)
    transformed_x = self.classifier(x)
    return transformed_x

net = MultiClassCNN()

model.to(device)
error = nn.CrossEntropyLoss()

learning_rate = 0.0001
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


def train(model, optimizer, error, num_epochs=25):
    """
    Function to train the model. The model object is provided as input.
    It runs over specified number of epochs as mentioned in the argument.
    """
    train_loss_history = []
    test_loss_history = []

    print("Training Started!")
    for epoch in range(num_epochs):
        train_loss = 0.0
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        for i, data in enumerate(train_loader):
            images, labels = data
            images = images
            labels = labels
            optimizer.zero_grad()
            predicted_output = net(images)
            fit = error(predicted_output,labels)
            fit.backward()
            optimizer.step()
            train_loss += fit.item()
        for i, data in enumerate(test_loader):
            with torch.no_grad():
                images, labels = data
                images = images
                labels = labels
                predicted_output = net(images)
                _, predicted = torch.max(predicted_output, 1)
                fit = error(predicted_output,labels)
                test_loss += fit.item()
                test_correct += (predicted == labels).sum()
                test_total += labels.numel()
        train_loss = train_loss/len(train_loader)
        test_loss = test_loss/len(test_loader)
        test_acc = test_correct.item() / test_total
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        print('Epoch %s, Train loss %s, Test loss %s, Test Acc %s'%(epoch, train_loss, test_loss, test_acc))

train(model, optimizer, error, num_epochs=50)
