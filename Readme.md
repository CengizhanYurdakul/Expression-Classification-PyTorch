# Binary Image Classifier and Organize CelebA Dataset!

Hi! In this project, I will guide you to organize CK+ dataset for expression classification in PyTorch.


### Steps
I will follow 2 steps;
#### 1. Prepate Data
#### 2. Train Model

## Prepare Data

Firstly, we need to download CK+ dataset from [here.](https://www.kaggle.com/shawon10/ckplus) Then extracted CK+48 folder to main root. `split_dataset.py` will split dataset into classes. Here, the test size was selected as 0.15. You can modify with this value according to the database that you will use.

```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)
```

## Train Model

In the training part, MobilenetV2 were used. Larger models such as VGG and Resnet suitable for your own project can be used here. 

### Tensorboard

The tensorboard section is available in the code to examine the results during training. You can change the file extension according to which the results will be examined.

```
## TENSORBOARD
logdir = "./Tensorboard/Experiment1_MobilenetV2_PretrainedFalse_Augmentation_LR0.01/"
writer = SummaryWriter(logdir)
```

### Transform and Dataloader

Requirements for converting the images in dataset to tensor and getting them ready for the model are available in the code. In transform, data augmentation methods were used. You can remove this part or change the probabilities in accordance with your own project.

```
## TRANSFORM
transform_ori = transforms.Compose([
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ColorJitter(brightness=0.2, contrast=0.25, saturation=0.2, hue=0.05),
                                    transforms.RandomPerspective(distortion_scale=0.04, p=0.4),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
```

Since the image sizes are small, I set the batch size to 64. You can change this value according to the computer you use.

```
## DATASET
batch_size = 64
train_load = torch.utils.data.DataLoader(dataset = train_dataset,
                                         batch_size = batch_size,
                                         shuffle = True)
```
### Model Selection

Since we have 7 classes, we are reducing the last layer of the MobilenetV2 model to 7. If this number is different in your dataset, you can change it.

```
## MOBILENETV2
model = models.mobilenet_v2(pretrained=False)
model.classifier = nn.Sequential(
                                nn.Dropout(0.2),
                                nn.Linear(1280, 7)
                                )
```

### Other Parameters

Loss function and optimizer are as shown.

```
loss_fn = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```
### Training

Once all the preparations have been made we can run the code. After printing the layers of the model, we can see the results for each epoch. Then we can examine the results we got from the tensorboard.

```
Epoch 1/100, Training Loss: 2.971, Training Accuracy: 21.000, Testing Loss: 2.176, Testing Acc: 18.000, Time: 1.6267s
...
Epoch 25/100, Training Loss: 0.628, Training Accuracy: 93.000, Testing Loss: 0.453, Testing Acc: 84.000, Time: 1.5118s
...
Epoch 50/100, Training Loss: 0.272, Training Accuracy: 97.000, Testing Loss: 0.065, Testing Acc: 97.000, Time: 1.693s
```
![Tensorboard](/Results/tensorboard.png)

## Requirements
- Torch
- Torchvision
- Matplotlib
- Numpy
- MTCNN
- Opencv-Python
- Pillow
- Pandas

> pip install -r requirements.txt
