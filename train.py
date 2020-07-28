import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
from torchvision import models
import os
from torch.utils.tensorboard import SummaryWriter


## TENSORBOARD
logdir = "./Tensorboard/Experiment1_MobilenetV2_PretrainedFalse_Augmentation_LR0.01/"
writer = SummaryWriter(logdir)

## TRANSFORM
transform_ori = transforms.Compose([
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ColorJitter(brightness=0.2, contrast=0.25, saturation=0.2, hue=0.05),
                                    transforms.RandomPerspective(distortion_scale=0.04, p=0.4),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

transform_test = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

## LOAD DATASET
train_dataset = datasets.ImageFolder(root = './Dataset/train/',
                                     transform = transform_ori)

test_dataset = datasets.ImageFolder(root = './Dataset/test/',
                                    transform = transform_test)

## DATALOADER
batch_size = 64
train_load = torch.utils.data.DataLoader(dataset = train_dataset,
                                         batch_size = batch_size,
                                         shuffle = True)

test_load = torch.utils.data.DataLoader(dataset = test_dataset,
                                         batch_size = batch_size,
                                         shuffle = False)

## MODEL SUMMARY

## MOBILENETV2
model = models.mobilenet_v2(pretrained=False)
model.classifier = nn.Sequential(
                                nn.Dropout(0.2),
                                nn.Linear(1280, 7)
                                )

print(model)

CUDA = torch.cuda.is_available()
if CUDA:
    model = model.cuda()
loss_fn = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

## PREPARING FOR TRAINING
train_loss = []
test_loss_list = []
train_accuracy = []
test_accuracy = []

## TRAINING
num_epochs = 100
for epoch in range(num_epochs):

    start = time.time()
    correct = 0
    iterations = 0
    iter_loss = 0.0

    model.train()

    for i, (inputs, labels) in enumerate(train_load):

        inputs = Variable(inputs)
        labels = Variable(labels)

        CUDA = torch.cuda.is_available()
        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        iter_loss += loss.item()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()
        iterations += 1

    train_loss.append(iter_loss/iterations)

    train_accuracy.append((100 * correct / len(train_dataset)))

    writer.add_scalar("Loss/Train", train_loss[-1], epoch)
    writer.add_scalar("Accuracy/Train", train_accuracy[-1], epoch)

    #Testing
    test_loss = 0.0
    correct = 0
    iterations_test = 0

    model.eval()

    for i, (inputs, labels) in enumerate(test_load):

        inputs = Variable(inputs)
        labels = Variable(labels)

        CUDA = torch.cuda.is_available()
        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()

        iterations_test += 1

    test_loss_list.append(test_loss/iterations_test)

    test_accuracy.append((100 * correct / len(test_dataset)))
    stop = time.time()

    writer.add_scalar("Loss/Test", test_loss_list[-1], epoch)
    writer.add_scalar("Accuracy/Test", test_accuracy[-1], epoch)

    # SAVE MODEL
    torch.save(model.state_dict(), logdir + 'Hair%s.pth' % epoch)

    print ('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Acc: {:.3f}, Time: {}s'
           .format(epoch+1, num_epochs, train_loss[-1], train_accuracy[-1], test_loss_list[-1], test_accuracy[-1], round(stop-start, 4)))
