import torch
import torchvision.transforms as transforms
import timm

import torch.nn as nn

import torch.optim as optim

from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split

from tooth_crop_dataset import ToothCropClassDataset
from utils.vit import train, test

writer = SummaryWriter()

# Data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Preprocess
transform = transforms.Compose([
    transforms.ToTensor(),
    # (lambda image: padding_to_size(image, 224)),
    transforms.Resize(size=(224, 224)),
    transforms.Normalize(mean=0.5, std=0.5),
])
target_transform = transforms.Compose([
    (lambda y: torch.Tensor(y)),
])

# Hyperparameter
epoch_num = 120
batch_size = 16
num_workers = 0
train_test_split = 0.8

# train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                          download=True, transform=transform)
dataset = ToothCropClassDataset(root='../preprocess', transform=transform, target_transform=target_transform)

dataset_size = len(dataset)
train_size = int(train_test_split * dataset_size)
test_size = dataset_size - train_size

train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)

classes = dataset.mlb.classes_

# model
# model = ViT(
#     image_size=300,
#     patch_size=30,
#     num_classes=len(classes),
#     dim=1024,
#     depth=6,
#     heads=8,
#     mlp_dim=2048,
#     channels=1,
#     dropout=0.2,
# )
# model = timm.create_model('vit_base_patch16_224', num_classes=6, pretrained=True)
model = timm.create_model('swin_base_patch4_window7_224', num_classes=4, pretrained=True)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
SGD_optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

if __name__ == '__main__':
    # inputs, labels = next(iter(train_loader))
    # inputs, labels = inputs.to(device), labels.to(device)

    for t in range(epoch_num):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_loader, model, criterion, SGD_optimizer, writer=writer, epoch=t, device=device)
        test(test_loader, model, criterion, len(classes), device=device, writer=writer, epoch=t, classes=classes)

    writer.close()
    print("Done!")

    print('Finished Training')
    # save your improved network
    torch.save(model.state_dict(), './pretrained-net.pt')
